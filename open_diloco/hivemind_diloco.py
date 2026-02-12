from enum import Enum
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import os
from datetime import datetime
import torch

from hivemind.averaging.averager import DecentralizedAverager
from hivemind.averaging.control import StepControl
from hivemind.compression.base import CompressionBase, NoCompression
from hivemind.dht.dht import DHT
from hivemind.optim.optimizer import Optimizer
from hivemind.optim.progress_tracker import (
    GlobalTrainingProgress,
    ProgressTracker,
    TrainingProgressSchema,
)
from hivemind.optim.state_averager import (
    LRSchedulerBase,
    OptimizerFactory,
    Parameters,
    ParamGroups,
    SchedulerFactory,
    TorchOptimizer,
    TrainingStateAverager,
)
from hivemind.utils import get_dht_time
from hivemind.utils.timed_storage import DHTExpiration
from hivemind.optim.optimizer import logger
from hivemind.optim.progress_tracker import LocalTrainingProgress


def unwrap(v):
    """Helper function to unwrap DHT values"""
    return getattr(v, "value", v)


def wait_for_all_nodes_local_step_complete(
    dht: DHT, 
    epoch: int, 
    num_inner_steps: int,
    expected_num_peers: int,
    log_fn=None,
    timeout: float = 300.0,
    check_interval: float = 0.1,
) -> float:
    """
    각 노드가 local step을 완료했는지 DHT를 통해 동기화합니다.
    
    :param dht: DHT 인스턴스
    :param epoch: 현재 epoch 번호
    :param num_inner_steps: 각 노드가 완료해야 하는 local step 수
    :param expected_num_peers: 예상되는 peer 수 (galaxy_size)
    :param log_fn: 로깅 함수 (None이면 logger.info 사용)
    :param timeout: 최대 대기 시간 (초)
    :param check_interval: 확인 간격 (초)
    :returns: 동기화 대기 시간 (초) - inner_step 계산 후 동기화까지의 GPU idle 시간
    """
    if log_fn is None:
        log_fn = logger.info
    
    RUN_ID = "OpenDiLoCo"
    local_step_key = f"{RUN_ID}:local_step_complete:epoch_{epoch}"
    worker_id = f"{socket.gethostname()}-pid{os.getpid()}"
    
    # 현재 노드의 local step 완료 상태를 DHT에 publish
    now = get_dht_time()
    local_step_payload = {
        "epoch": epoch,
        "local_steps_completed": num_inner_steps,
        "completed": True,
        "ts": now,
        "host": socket.gethostname(),
    }
    exp = now + timeout
    dht.store(key=local_step_key, subkey=worker_id, value=local_step_payload, expiration_time=exp)
    
    # 모든 노드가 local step을 완료할 때까지 대기 (inner_step 계산 후 동기화까지의 대기 시간 측정)
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            break
        
        local_step_res = dht.get(local_step_key, latest=True)
        local_step_root = unwrap(local_step_res) if local_step_res else None
        completed_count = 0
        
        if isinstance(local_step_root, dict):
            for k, v in local_step_root.items():
                p = unwrap(v)
                if isinstance(p, dict) and p.get("completed") is True and p.get("epoch") == epoch:
                    completed_count += 1
        
        if completed_count >= expected_num_peers:
            break
        
        time.sleep(check_interval)
    
    wait_time = time.time() - start_time
    
    # 동기화 완료 후 상태를 짧은 시간 후 만료되도록 업데이트
    now = get_dht_time()
    local_step_payload = {
        "epoch": epoch,
        "local_steps_completed": num_inner_steps,
        "completed": True,
        "ts": now,
        "host": socket.gethostname(),
    }
    exp = now + 5.0  # 5초 후 만료
    dht.store(key=local_step_key, subkey=worker_id, value=local_step_payload, expiration_time=exp)
    
    return wait_time

try:
    from .utils import found_inf_grad
    from .token_weighted_aggregation import TokenWeightedAggregationMixin
except ImportError:
    try:
        from open_diloco.utils import found_inf_grad
        from open_diloco.token_weighted_aggregation import TokenWeightedAggregationMixin
    except ImportError:
        # Fallback for direct module execution (same directory)
        from utils import found_inf_grad
        from token_weighted_aggregation import TokenWeightedAggregationMixin
# cyshin
import logging
import socket
import os
import time
from pydantic.v1 import BaseModel, StrictBool, StrictFloat, confloat, conint


class DiLoCoStateAverager(TrainingStateAverager, TokenWeightedAggregationMixin):
    def __init__(
        self,
        *,
        num_inner_steps: int,
        inner_optimizer: TorchOptimizer,
        scheduler: Optional[SchedulerFactory] = None,
        token_weighted_aggregation: bool = False,
        **kwargs,
    ):
        self.inner_optimizer = inner_optimizer
        self.num_inner_steps = num_inner_steps

        # Token-weighted aggregation 지원 (mixin 초기화)
        self.token_weighted_aggregation = token_weighted_aggregation
        self.cumulative_tokens = 0
        self.token_count_key = None
        self.worker_id = None
        self.expected_num_peers = None
        
        # grad_averager 참조는 나중에 설정됨 (DiLoCoOptimizer에서)
        self.grad_averager = None

        super().__init__(
            **kwargs
        )  # we specifically don't pass the scheduler here, default TrainingStateAverager would use it with the outer optimizer and we w

        self.scheduler_inner_optimizer = scheduler(self.inner_optimizer) if scheduler is not None else None
        assert isinstance(self.scheduler_inner_optimizer, (LRSchedulerBase, type(None)))
        
        # Token-weighted aggregation 초기화 (DHT와 worker_id가 설정된 후)
        if self.token_weighted_aggregation:
            self._init_token_weighted_aggregation(key_suffix="_state")

    @torch.no_grad()
    def _apply_optimizer_parameters_(self):
        """Copy parameters from offloaded optimizer to the main model
        모든 파라미터를 업데이트합니다. Skip된 파라미터의 경우 gradient가 0이었으므로
        outer optimizer step 후에도 변화가 없어야 하지만, 안전을 위해 모든 파라미터를 업데이트합니다.
        """
        assert self.offload_optimizer, "Applying offloaded optimizer updates requires offloaded optimizer"
        offloaded_parameters = [param for group in self.optimizer.param_groups for param in group["params"]]
        assert len(offloaded_parameters) == len(self.main_parameters), "Optimizer parameters changed during training"
        
        # 모든 파라미터를 업데이트 (skip된 파라미터도 포함)
        # Skip된 파라미터의 경우 gradient가 0으로 설정되어 outer optimizer step 후에도
        # 변화가 없어야 하지만, 모든 파라미터를 일관되게 업데이트합니다.
        for main_param, offloaded_param in zip(self.main_parameters, offloaded_parameters):
            main_param.copy_(offloaded_param, non_blocking=True)


    @torch.no_grad()
    def _apply_averaging_results_(self):
        """Copy averaged tensors into their respective local tensors
        Skip 유무와 상관없이 모든 파라미터를 덮어씁니다.
        """
        assert not self.reuse_tensors, "No need to update averaged tensors since they reuse the same memory"
        if self.delta_rule_averaging and self._old_tensors is None:
            logger.warning("Using delta_rule_averaging, but old tensors were not found. Averaging may have failed")
        
        with self.get_tensors() as averaged_tensors:
            local_tensors = list(self._local_tensors())
            assert len(local_tensors) == len(averaged_tensors), "Tensor structure changed during training"
            
            # outer optimizer의 파라미터만 필터링 (optimizer statistics는 제외)
            # _local_tensors()는 [optimizer_params, optimizer_stats, extra_tensors] 순서로 반환
            num_optimizer_params = len(self.optimizer.param_groups[0]["params"]) if self.optimizer.param_groups else 0
            # 실제로는 모든 param_groups의 파라미터 수를 세어야 함
            total_optimizer_params = sum(len(group["params"]) for group in self.optimizer.param_groups)
            
            for idx, (local_tensor, averaged_tensor) in enumerate(zip(local_tensors, averaged_tensors)):
                # 모든 파라미터 업데이트 (skip 유무와 상관없이 덮어쓰기)
                if not self.delta_rule_averaging or self._old_tensors is None:
                    local_tensor.copy_(averaged_tensor, non_blocking=True)
                else:
                    # Delta rule averaging
                    old_tensor = self._old_tensors[idx]
                    delta = torch.sub(averaged_tensor, old_tensor, out=old_tensor)
                    local_tensor.add_(delta.to(device=local_tensor.device, dtype=local_tensor.dtype))

    def _update_scheduler(self):
        """Increase the scheduler state until it becomes synchronized with local epoch"""
        # TODO(sami) handle update scheduler
        # for now assuming that all scheduler are on time
        pass
    
    def _adjust_momentum_from_token_weight(self, token_weight: float, base_momentum: Optional[float] = None) -> Optional[float]:
        """
        Token weight를 기반으로 outer optimizer의 momentum을 조정.
        
        :param token_weight: Token weight (0.0 ~ 1.0, 자신의 token 수 / 전체 token 수)
        :param base_momentum: 기본 momentum 값 (None이면 optimizer에서 자동으로 가져옴)
        :returns: 조정된 momentum 값 (조정 실패 시 None)
        """
        if not self.token_weighted_aggregation or token_weight <= 0:
            return base_momentum
        
        # Outer optimizer의 momentum 변경 (SGD optimizer인 경우)
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return None
        
        # base_momentum이 없으면 optimizer에서 현재 momentum 값 가져오기
        if base_momentum is None:
            for param_group in self.optimizer.param_groups:
                if 'momentum' in param_group:
                    base_momentum = param_group['momentum']
                    break
            if base_momentum is None:
                logger.warning("StateAverager: optimizer에 'momentum' 파라미터가 없어 momentum 조정을 건너뜁니다.")
                return None
        
        # Token weight를 기반으로 momentum 조정
        # token_weight는 0~1 사이 값 (자신의 token 수 / 전체 token 수)
        # token_weight가 높을수록 (더 많은 token 처리) momentum을 높게 설정
        
        # 선형 보간: token_weight=0일 때 min_ratio, token_weight=1일 때 max_ratio
        # 예: min_ratio=0.8, max_ratio=1.1이면 base_momentum * 0.8 ~ base_momentum * 1.1 범위
        min_momentum_ratio = 0.5  # token_weight=0일 때 momentum 비율
        max_momentum_ratio = 1.0  # token_weight=1일 때 momentum 비율
        
        adjusted_momentum = min_momentum_ratio + token_weight * (max_momentum_ratio - min_momentum_ratio)
        
        # Outer optimizer의 momentum 변경
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                old_momentum = param_group['momentum']
                param_group['momentum'] = adjusted_momentum
                if abs(old_momentum - adjusted_momentum) > 1e-6:  # 변화가 있을 때만 로그
                    logger.info(
                        f"StateAverager momentum adjusted: token_weight={token_weight:.6f}, "
                        f"old_momentum={old_momentum:.6f}, new_momentum={adjusted_momentum:.6f}"
                    )
        
        return adjusted_momentum
    
    def _do(
        self,
        wait_for_trigger: Optional[Callable[[], Any]],
        optimizer_step: bool,
        zero_grad: bool,
        averaging_round: bool,
        averaging_control: Optional[StepControl],
        grad_scaler: Optional[Any],
        set_to_none: bool,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Run the optimizer step, followed by a scheduler step and an averaging round, each stage is optional.
        This method overrides TrainingStateAverager._do to add token-weighted aggregation support.
        """
        # Token-weighted aggregation: weight 계산 및 설정
        computed_token_weight = None
        sync_wait_time_state = 0.0  # StateAverager의 동기화 대기 시간
        if averaging_round and 'weight' not in kwargs:
            # Token weight 수집 전체 시간 측정 시작
            time_0_token_weight_state = time.perf_counter() if self.token_weighted_aggregation else None
            
            if averaging_control is not None:
                # 이미 생성된 control이 있으면 weight 설정
                # _compute_token_weight는 weight을 반환
                computed_token_weight = self._compute_token_weight(averaging_control, log_prefix="StateAverager Token-weighted aggregation")
            elif self.token_weighted_aggregation and self.cumulative_tokens > 0:
                # control이 None이면, weight를 계산해서 kwargs에 추가
                # 먼저 token 수를 publish하고 읽어옴
                self._publish_token_count_to_dht()
                token_counts = self._read_token_counts_from_dht(
                    wait_for_peers=True,
                    max_wait_time=300.0,
                    check_interval=0.2
                )
                total_tokens = sum(token_counts.values())
                if total_tokens > 0:
                    weight = self.cumulative_tokens / total_tokens
                    computed_token_weight = weight
                    kwargs['weight'] = weight
                    logger.info(
                        f"StateAverager Token-weighted aggregation: local_tokens={self.cumulative_tokens}, "
                        f"total_tokens={total_tokens}, weight={weight:.6f}, num_peers={len(token_counts)}"
                    )
            
            # Token weight 수집 전체 시간 측정 종료
            if time_0_token_weight_state is not None:
                time_1_token_weight_state = time.perf_counter()
                sync_wait_time_state = time_1_token_weight_state - time_0_token_weight_state
                if computed_token_weight is not None:
                    logger.info(
                        f"StateAverager Token-weighted aggregation sync_wait_time={sync_wait_time_state:.6f} sec (GPU idle time during synchronization)"
                    )
                else:
                    logger.warning(
                        f"StateAverager Token-weighted aggregation: total_tokens is 0, using uniform weight, "
                        f"sync_wait_time={sync_wait_time_state:.6f} sec (GPU idle time during synchronization)"
                    )
        
        # Token weight를 기반으로 momentum 조정 (optimizer step이 있을 때만)
        if optimizer_step and self.token_weighted_aggregation:
            # Token weight 가져오기 (computed_token_weight 또는 averaging_control.weight)
            token_weight = computed_token_weight
            if token_weight is None and averaging_control is not None and hasattr(averaging_control, 'weight'):
                token_weight = averaging_control.weight
            
            if token_weight is not None and token_weight > 0:
                self._adjust_momentum_from_token_weight(token_weight)
        
        # StateAverager의 동기화 대기 시간 출력 (GPU idle 시간)
        if sync_wait_time_state > 0.0:
            logger.info(
                f"StateAverager GPU idle time during synchronization (after inner steps, before all-reduce): {sync_wait_time_state:.6f} sec"
            )
        
        # 부모 클래스의 _do 메서드 호출
        result = super()._do(
            wait_for_trigger=wait_for_trigger,
            optimizer_step=optimizer_step,
            zero_grad=zero_grad,
            averaging_round=averaging_round,
            averaging_control=averaging_control,
            grad_scaler=grad_scaler,
            set_to_none=set_to_none,
            timeout=timeout,
            **kwargs,
        )
        
        return result


class DiLoCoGradAverager(DecentralizedAverager, TokenWeightedAggregationMixin):
    """ "
    DiLoCoGradAverager is meant to be used in pair with DiLoCoStateAverager. Specifically it takes as input the offloaded optimizer of DiLoCoStateAverager, and
    use the grad buffer of the offloaded param as averaged_tensors for the DecentralizedAverager. In other words the DiLoCoGradAverager makes sure that the grad of the offloaded optimizer
    are kept in sync between peers.
    """

    def __init__(
        self,
        main_parameters: List[torch.nn.Parameter],
        offloaded_optimizer: TorchOptimizer,
        *,
        dht: DHT,
        prefix: str,
        warn: bool = True,
        param_names: Optional[List[str]] = None,
        selective_layer_patterns: Optional[List[str]] = None,
        gradient_magnitude_threshold: Optional[float] = None,
        gradient_magnitude_top_k_ratio: Optional[float] = None,
        gradient_magnitude_top_k_ratio_by_size: bool = False,
        gradient_magnitude_selection_mode: str = "layer",  # "layer" or "parameter"
        gradient_importance_metric: str = "magnitude",  # "magnitude" or "taylor"
        token_weighted_aggregation: bool = False,
        residual_norm_threshold: Optional[float] = None,
        enable_update_logs: bool = False,
        # Max Staleness (강제 업데이트) 기능
        enable_max_staleness: bool = False,
        max_staleness: int = 100,  # N번 이상 선택되지 않으면 강제 포함
        # Warm-up (서서히 줄이기) 기능
        enable_warmup: bool = False,
        warmup_epochs: int = 5,  # 초기 N epoch 동안은 모든 파라미터 전송
        **kwargs,
    ):
        if "client_mode" in kwargs:
            if kwargs["client_mode"] is not None and kwargs["client_mode"]:
                raise KeyError("client_mode is not supported in DiLoCoGradAverager")
            else:
                kwargs.pop("client_mode")

        if "averaged_grads" in kwargs:
            raise KeyError(
                "DiLoCoGradAverager does not support averaged_grads since it use the offloaded optimizer gradients directly"
            )

        if not isinstance(main_parameters, (list, tuple)):
            raise ValueError(
                "main_parameters must be a list or tuple of torch.nn.Parameter and not an iterator otherwise parameters will be consumed"
            )
        self.main_parameters = list(main_parameters)
        self.offloaded_optimizer = offloaded_optimizer

        self.warn = warn
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0

        self._new_averaged_grads = False
        
        # Token-weighted aggregation 지원 (mixin 초기화)
        self.token_weighted_aggregation = token_weighted_aggregation
        self.cumulative_tokens = 0
        self.token_count_key = None
        self.worker_id = None
        self.expected_num_peers = None

        # Local step 동기화를 위한 정보 저장
        self.num_inner_steps = None  # 나중에 설정됨
        self._current_epoch = None
        self._current_step = None
        
        # 파라미터 선택 및 업데이트 추적을 위한 로그 디렉토리
        self.log_dir = kwargs.pop("log_dir", "./parameter_tracking_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self._parameter_selection_history = []  # 시간에 따른 선택 이력
        self._parameter_update_history = []  # 파라미터 업데이트 이력

        # Selective layer update 지원
        self.param_names = param_names
        self.selective_layer_patterns = selective_layer_patterns
        self.gradient_magnitude_threshold = gradient_magnitude_threshold
        self.gradient_magnitude_top_k_ratio = gradient_magnitude_top_k_ratio
        self.gradient_magnitude_top_k_ratio_by_size = gradient_magnitude_top_k_ratio_by_size
        
        # Selection mode: "layer" (레이어 단위) or "parameter" (파라미터 텐서 단위)
        if gradient_magnitude_selection_mode not in ["layer", "parameter"]:
            raise ValueError(f"gradient_magnitude_selection_mode must be 'layer' or 'parameter', got '{gradient_magnitude_selection_mode}'")
        self.gradient_magnitude_selection_mode = gradient_magnitude_selection_mode
        
        # Importance metric: "magnitude" (L2 norm) or "taylor" (|w * g|)
        if gradient_importance_metric not in ["magnitude", "taylor"]:
            raise ValueError(f"gradient_importance_metric must be 'magnitude' or 'taylor', got '{gradient_importance_metric}'")
        self.gradient_importance_metric = gradient_importance_metric
        
        # Residual norm threshold 설정
        self.residual_norm_threshold = residual_norm_threshold
        
        # 파라미터 업데이트 로그 활성화/비활성화 (파일 저장 및 상세 로깅)
        self.enable_update_logs = enable_update_logs
        
        # 파일 I/O 작업을 위한 ThreadPoolExecutor 초기화 (비동기 로깅용)
        self._log_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="diloco_logger") if enable_update_logs else None
        
        # Max Staleness (강제 업데이트) 기능 설정
        self.enable_max_staleness = enable_max_staleness
        self.max_staleness = max_staleness
        if self.enable_max_staleness and param_names is not None:
            if self.gradient_magnitude_selection_mode == "layer":
                # 레이어 단위 Staleness 카운터 초기화 (CPU에 저장하여 메모리 효율성)
                layer_to_indices = self._group_parameters_by_layer(param_names)
                self.layer_staleness_counters = {layer_name: 0 for layer_name in layer_to_indices.keys()}
                self.param_staleness_counters = None
                logger.info(f"Max Staleness enabled (layer-based): layers not selected for {max_staleness} steps will be forced to update")
                logger.info(f"  Total layers: {len(self.layer_staleness_counters)}")
            else:  # parameter mode
                # 파라미터 단위 Staleness 카운터 초기화
                self.param_staleness_counters = [0] * len(param_names)
                self.layer_staleness_counters = None
                logger.info(f"Max Staleness enabled (parameter-based): parameters not selected for {max_staleness} steps will be forced to update")
                logger.info(f"  Total parameters: {len(self.param_staleness_counters)}")
        else:
            self.layer_staleness_counters = None
            self.param_staleness_counters = None
        
        # Warm-up (서서히 줄이기) 기능 설정
        self.enable_warmup = enable_warmup
        self.warmup_epochs = warmup_epochs
        self.original_top_k_ratio = gradient_magnitude_top_k_ratio  # 원본 top_k_ratio 저장
        self.original_top_k_ratio_by_size = gradient_magnitude_top_k_ratio_by_size  # 원본 top_k_ratio_by_size 저장
        if self.enable_warmup:
            logger.info(f"Warm-up enabled: all parameters will be sent for first {warmup_epochs} epochs, then gradually reduce to target sparsity")
        
        # Gradient magnitude 기반 선택과 pattern 기반 선택은 동시에 사용 불가
        if (gradient_magnitude_threshold is not None or gradient_magnitude_top_k_ratio is not None) and selective_layer_patterns is not None:
            raise ValueError("gradient_magnitude_threshold/top_k_ratio와 selective_layer_patterns는 동시에 사용할 수 없습니다.")
        
        # Residual buffer는 더 이상 필요 없음
        # main_param을 리셋하지 않으면, opt_param - main_param이 이미 누적된 차이를 나타냄
        # 따라서 별도의 residual buffer를 유지할 필요가 없음 (Implicit Accumulation)
        self.residual_buffers = None
        
        # 초기 마스크 설정 (pattern 기반이면 초기화 시 생성, magnitude 기반이면 step에서 동적 생성)
        if selective_layer_patterns is not None and param_names is not None:
            # 파라미터 마스크 생성: 업데이트할 레이어 패턴에 매칭되는 파라미터만 True
            self.param_update_mask = self._create_param_mask(param_names, selective_layer_patterns)
            updated_count = sum(self.param_update_mask)
            total_count = len(self.param_update_mask)
            logger.info(f"Selective layer update (pattern-based) enabled: {updated_count}/{total_count} parameters will be updated")
            
            # 디버그: 업데이트되는 파라미터 이름 출력
            updated_params = [name for name, mask in zip(param_names, self.param_update_mask) if mask]
            skipped_params = [name for name, mask in zip(param_names, self.param_update_mask) if not mask]
            
            logger.info(f"[DEBUG] Parameters to be updated ({updated_count}):")
            for i, name in enumerate(updated_params[:20]):  # 처음 20개만 출력
                logger.info(f"  {i+1}. {name}")
            if len(updated_params) > 20:
                logger.info(f"  ... and {len(updated_params) - 20} more parameters")
            
            if len(skipped_params) > 0:
                logger.info(f"[DEBUG] Parameters skipped from update ({len(skipped_params)}):")
                for i, name in enumerate(skipped_params[:10]):  # 처음 10개만 출력
                    logger.info(f"  {i+1}. {name}")
                if len(skipped_params) > 10:
                    logger.info(f"  ... and {len(skipped_params) - 10} more parameters")
        elif gradient_magnitude_threshold is not None or gradient_magnitude_top_k_ratio is not None:
            # Gradient magnitude 기반 선택: 초기에는 모든 파라미터를 포함하는 더미 마스크 생성
            # step에서 동적으로 마스크를 업데이트하고 averaged_grads를 재생성함
            if param_names is not None:
                # 초기에는 모든 파라미터를 포함하는 더미 마스크 생성
                self.param_update_mask = [True] * len(param_names)
                logger.info(f"Selective layer update (gradient magnitude-based) enabled: threshold={gradient_magnitude_threshold}, top_k_ratio={gradient_magnitude_top_k_ratio}")
                logger.info(f"Initial mask: all {len(param_names)} parameters included (will be updated dynamically in step)")
            else:
                self.param_update_mask = None
                logger.warning("gradient_magnitude_* 설정이 있지만 param_names가 None입니다. 동적 마스크 업데이트가 불가능합니다.")
        else:
            self.param_update_mask = None
            if param_names is not None:
                logger.info(f"[DEBUG] All {len(param_names)} parameters will be updated (no selective layer patterns)")

        averaged_grads = tuple(grad for grad in self._grads_from_optimizer())

        super().__init__(
            averaged_tensors=averaged_grads,
            dht=dht,
            prefix=prefix,
            client_mode=False,
            classstr="gradaverager",
            **kwargs,
        )
        
        # Token-weighted aggregation 초기화 (DHT와 worker_id가 설정된 후)
        if self.token_weighted_aggregation:
            self._init_token_weighted_aggregation(key_suffix="_grad")

    def _create_param_mask(self, param_names: List[str], patterns: List[str]) -> List[bool]:
        """파라미터 이름이 패턴 중 하나라도 매칭되면 True인 마스크 생성
        Transformer 모델의 관례에 따라 embedding과 lm_head는 항상 포함 (Dense하게 유지)
        """
        mask = []
        for param_name in param_names:
            matched = False
            for pattern in patterns:
                # 패턴이 파라미터 이름의 시작 부분과 매칭되는지 확인
                if pattern in param_name or param_name.startswith(pattern):
                    matched = True
                    break
            
            # Transformer 모델의 관례: embedding과 lm_head는 항상 포함 (Dense하게 유지)
            if not matched:
                # embedding 레이어 체크 (embed_tokens, embeddings 등)
                if 'embed' in param_name.lower() or 'embed_tokens' in param_name:
                    matched = True
                # output head 레이어 체크 (lm_head, output 등)
                elif 'lm_head' in param_name.lower() or param_name.endswith('.head') or 'output' in param_name.lower():
                    matched = True
            
            mask.append(matched)
        return mask

    def _extract_layer_name(self, param_name: str) -> str:
        """파라미터 이름에서 레이어 이름을 추출
        
        예시:
        - "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0"
        - "model.embed_tokens.weight" -> "model.embed_tokens"
        - "lm_head.weight" -> "lm_head"
        - "transformer.h.0.attn.c_attn.weight" -> "transformer.h.0"
        """
        if "." not in param_name:
            return param_name
        
        parts = param_name.split(".")
        
        # "model.layers.X" 형태 (Llama, Mistral 등)
        if "layers" in parts:
            layer_idx = parts.index("layers")
            if layer_idx + 1 < len(parts):
                return ".".join(parts[:layer_idx + 2])  # "model.layers.0"
        
        # "transformer.h.X" 형태 (GPT-2 등)
        if len(parts) >= 3 and parts[0] == "transformer" and parts[1] == "h":
            try:
                # parts[2]가 숫자인지 확인
                int(parts[2])
                return ".".join(parts[:3])  # "transformer.h.0"
            except ValueError:
                pass
        
        # "blocks.X" 형태
        if "blocks" in parts:
            block_idx = parts.index("blocks")
            if block_idx + 1 < len(parts):
                return ".".join(parts[:block_idx + 2])  # "blocks.0"
        
        # "model.embed_tokens", "model.norm" 등 (root 레벨)
        if len(parts) >= 2:
            # 첫 두 부분이 모델의 루트 레벨 컴포넌트인 경우
            if parts[0] == "model" or parts[0] == "transformer":
                # embed_tokens, norm 등은 별도 레이어로 취급
                if len(parts) >= 2:
                    return ".".join(parts[:2])  # "model.embed_tokens"
        
        # "lm_head" 같은 경우
        if parts[0] == "lm_head" or (len(parts) >= 1 and "head" in parts[0].lower()):
            return parts[0]
        
        # 기본값: 첫 두 부분만 반환
        return ".".join(parts[:2]) if len(parts) >= 2 else param_name

    def _group_parameters_by_layer(self, param_names: List[str]) -> Dict[str, List[int]]:
        """파라미터를 레이어별로 그룹화
        
        Returns:
            layer_to_param_indices: {layer_name: [param_idx1, param_idx2, ...]}
        """
        layer_to_indices = {}
        
        for idx, param_name in enumerate(param_names):
            layer_name = self._extract_layer_name(param_name)
            if layer_name not in layer_to_indices:
                layer_to_indices[layer_name] = []
            layer_to_indices[layer_name].append(idx)
        
        return layer_to_indices

    def _is_input_output_layer(self, layer_name: str) -> bool:
        """Input/Output 레이어인지 확인
        
        Input 레이어: embed_tokens, embeddings 등
        Output 레이어: lm_head, output 등
        """
        layer_name_lower = layer_name.lower()
        
        # Input 레이어 체크
        if 'embed' in layer_name_lower or 'embed_tokens' in layer_name:
            return True
        
        # Output 레이어 체크
        if 'lm_head' in layer_name_lower or layer_name.endswith('.head') or 'output' in layer_name_lower:
            return True
        
        return False

    @torch.no_grad()
    def _compute_gradient_magnitudes(self) -> List[float]:
        """각 파라미터의 gradient magnitude를 계산 (L2 norm)
        opt_param과 main_param의 차이를 계산하여 magnitude를 구합니다.
        
        Note: 이제 모든 파라미터를 업데이트하므로, skip된 파라미터의 누적 차이 문제가 해결됩니다.
        하지만 magnitude 계산은 outer step 전에 이루어지므로, 이전 step에서 skip된 파라미터의
        누적 차이가 여전히 반영될 수 있습니다. 다음 step부터는 모든 파라미터가 업데이트되므로
        이 문제가 해결됩니다.
        """
        magnitudes = []
        param_groups = self.offloaded_optimizer.param_groups
        param_idx = 0
        
        # main_parameters를 리스트로 변환 (인덱스 접근을 위해)
        main_params_list = list(self.main_parameters)
        
        for param_group in param_groups:
            for param in param_group["params"]:
                # offloaded_optimizer의 파라미터와 main_parameters는 같은 순서로 구성되어 있음
                if param_idx < len(main_params_list):
                    main_param = main_params_list[param_idx]
                    opt_param = param  # offloaded_optimizer의 파라미터
                    
                    # opt_param과 main_param의 차이를 계산 (pseudo gradient)
                    # 이제 모든 파라미터를 업데이트하므로, 이 차이는 현재 step의 gradient를 나타냅니다.
                    # 하지만 magnitude 계산은 outer step 전에 이루어지므로, 이전 step에서 skip된
                    # 파라미터의 누적 차이가 여전히 반영될 수 있습니다.
                    pseudo_grad = opt_param.data - main_param.detach().to(opt_param.device)
                    
                    # NaN이나 Inf 체크
                    if torch.isfinite(pseudo_grad).all():
                        magnitude = pseudo_grad.norm(p=2).item()
                    else:
                        magnitude = 0.0
                else:
                    # 인덱스 범위를 벗어나면 0
                    magnitude = 0.0
                
                magnitudes.append(magnitude)
                param_idx += 1
                
        return magnitudes

    @torch.no_grad()
    def _compute_taylor_scores(self) -> List[float]:
        """각 파라미터의 Taylor 1st order saliency score를 계산 (|w * g|)
        Taylor Score = |w_i * g_i| = 파라미터를 제거했을 때 예상되는 Loss 증가량
        
        Returns:
            taylor_scores: 각 파라미터 텐서의 Taylor score 리스트
        """
        taylor_scores = []
        param_groups = self.offloaded_optimizer.param_groups
        param_idx = 0
        
        # main_parameters를 리스트로 변환 (인덱스 접근을 위해)
        main_params_list = list(self.main_parameters)
        
        for param_group in param_groups:
            for param in param_group["params"]:
                # offloaded_optimizer의 파라미터와 main_parameters는 같은 순서로 구성되어 있음
                if param_idx < len(main_params_list):
                    main_param = main_params_list[param_idx]
                    opt_param = param  # offloaded_optimizer의 파라미터
                    
                    # opt_param과 main_param의 차이를 계산 (pseudo gradient)
                    pseudo_grad = opt_param.data - main_param.detach().to(opt_param.device)
                    
                    # Taylor Score = |w * g| (절대값 내적)
                    # w: 현재 파라미터 값, g: gradient
                    if torch.isfinite(pseudo_grad).all() and torch.isfinite(opt_param.data).all():
                        # 각 요소별로 |w * g| 계산 후 합산
                        taylor_score = (opt_param.data * pseudo_grad).abs().sum().item()
                    else:
                        taylor_score = 0.0
                else:
                    # 인덱스 범위를 벗어나면 0
                    taylor_score = 0.0
                
                taylor_scores.append(taylor_score)
                param_idx += 1
                
        return taylor_scores

    def _compute_layer_magnitudes(self, param_magnitudes: List[float]) -> Dict[str, float]:
        """레이어별 magnitude 계산 (레이어 내 모든 파라미터의 합)
        
        Args:
            param_magnitudes: 각 파라미터 텐서의 magnitude 리스트
            
        Returns:
            layer_magnitudes: {layer_name: layer_magnitude}
        """
        if self.param_names is None:
            raise ValueError("param_names가 None입니다. 레이어 단위 계산을 위해 필요합니다.")
        
        layer_to_indices = self._group_parameters_by_layer(self.param_names)
        layer_magnitudes = {}
        
        for layer_name, param_indices in layer_to_indices.items():
            # 레이어 내 모든 파라미터의 magnitude 합계
            layer_mag = sum(param_magnitudes[i] for i in param_indices if i < len(param_magnitudes))
            layer_magnitudes[layer_name] = layer_mag
        
        return layer_magnitudes

    def _collect_and_average_gradient_magnitudes(
        self,
        local_magnitudes: List[float],
        epoch: int,
        timeout: float = 300.0,
        check_interval: float = 0.1
    ) -> List[float]:
        """
        모든 노드의 gradient importance scores를 DHT에 공유하고 평균을 계산합니다.
        이렇게 하면 모든 노드가 동일한 기준으로 파라미터를 선택할 수 있습니다.
        
        :param local_magnitudes: 현재 노드의 importance score 리스트 (magnitude 또는 taylor)
        :param epoch: 현재 epoch 번호
        :param timeout: 최대 대기 시간 (초)
        :param check_interval: 확인 간격 (초)
        :returns: 모든 노드의 평균 importance score 리스트
        """
        RUN_ID = "OpenDiLoCo"
        # Metric에 따라 DHT 키 분리 (magnitude와 taylor가 섞이지 않도록)
        metric_suffix = "magnitudes" if self.gradient_importance_metric == "magnitude" else "taylor"
        magnitude_key = f"{RUN_ID}:gradient_{metric_suffix}:epoch_{epoch}"
        worker_id = f"{socket.gethostname()}-pid{os.getpid()}"
        
        # 현재 노드의 gradient magnitude를 DHT에 publish
        now = get_dht_time()
        magnitude_payload = {
            "epoch": epoch,
            "magnitudes": local_magnitudes,
            "ts": now,
            "host": socket.gethostname(),
        }
        exp = now + timeout
        self.dht.store(key=magnitude_key, subkey=worker_id, value=magnitude_payload, expiration_time=exp)
        
        # 모든 노드의 gradient magnitude를 수집
        all_magnitudes = []
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.warning(f"Timeout waiting for all nodes to publish gradient magnitudes (timeout={timeout}s). Using local magnitudes only.")
                return local_magnitudes
            
            magnitude_res = self.dht.get(magnitude_key, latest=True)
            magnitude_root = unwrap(magnitude_res) if magnitude_res else None
            
            collected_count = 0
            all_magnitudes = []
            
            if isinstance(magnitude_root, dict):
                for k, v in magnitude_root.items():
                    p = unwrap(v)
                    if isinstance(p, dict) and p.get("epoch") == epoch and "magnitudes" in p:
                        magnitudes = p["magnitudes"]
                        if isinstance(magnitudes, list) and len(magnitudes) == len(local_magnitudes):
                            all_magnitudes.append(magnitudes)
                            collected_count += 1
            
            expected_count = self.expected_num_peers if self.expected_num_peers is not None else 2
            
            if collected_count >= expected_count and len(all_magnitudes) > 0:
                break
            
            time.sleep(check_interval)
        
        # 모든 노드의 magnitude를 평균
        if len(all_magnitudes) == 0:
            logger.warning("No gradient magnitudes collected from peers. Using local magnitudes only.")
            return local_magnitudes
        
        # numpy를 사용하여 평균 계산
        magnitudes_array = np.array(all_magnitudes)
        averaged_magnitudes = magnitudes_array.mean(axis=0).tolist()
        
        # 동기화 완료 후 상태를 짧은 시간 후 만료되도록 업데이트
        now = get_dht_time()
        magnitude_payload = {
            "epoch": epoch,
            "magnitudes": local_magnitudes,
            "ts": now,
            "host": socket.gethostname(),
        }
        exp = now + 5.0  # 5초 후 만료
        self.dht.store(key=magnitude_key, subkey=worker_id, value=magnitude_payload, expiration_time=exp)
        
        return averaged_magnitudes

    def _create_layer_based_mask(
        self,
        layer_magnitudes: Dict[str, float],
        threshold: Optional[float] = None,
        top_k_ratio: Optional[float] = None,
        top_k_ratio_by_size: bool = False,
        forced_layers: Optional[set] = None
    ) -> List[bool]:
        """레이어 단위로 마스크 생성
        - Input/Output 레이어는 항상 True
        - Transformer block 레이어들은 magnitude 기반으로 선택
        
        Args:
            layer_magnitudes: {layer_name: layer_magnitude}
            threshold: magnitude threshold 이상의 레이어만 선택
            top_k_ratio: top-k% 레이어 선택 (0.0 ~ 1.0)
            top_k_ratio_by_size: True면 레이어 크기 기준으로 비율 계산, False면 개수 기준
            forced_layers: Max Staleness로 강제 포함할 레이어 이름 집합
            
        Returns:
            mask: 파라미터별 마스크 리스트
        """
        if self.param_names is None:
            raise ValueError("param_names가 None입니다. 레이어 단위 마스크 생성을 위해 필요합니다.")
        
        if forced_layers is None:
            forced_layers = set()
        
        layer_to_indices = self._group_parameters_by_layer(self.param_names)
        mask = [False] * len(self.param_names)
        
        # 디버깅: 레이어 그룹화 정보 출력
        logger.info("=" * 80)
        logger.info("[DEBUG] Layer Grouping Information")
        logger.info("=" * 80)
        logger.info(f"Total layers identified: {len(layer_to_indices)}")
        logger.info(f"Total parameters: {len(self.param_names)}")
        
        # 레이어별 정보 출력
        input_output_layers = []
        transformer_layers = []
        for layer_name, param_indices in sorted(layer_to_indices.items()):
            is_io = self._is_input_output_layer(layer_name)
            layer_type = "Input/Output" if is_io else "Transformer Block"
            param_count = len(param_indices)
            total_size = sum(self.main_parameters[i].numel() for i in param_indices)
            magnitude = layer_magnitudes.get(layer_name, 0.0)
            
            logger.info(f"  Layer: {layer_name:<40} Type: {layer_type:<20} Params: {param_count:<5} Size: {total_size:<12} Magnitude: {magnitude:.6f}")
            
            if is_io:
                input_output_layers.append(layer_name)
            else:
                transformer_layers.append(layer_name)
        
        logger.info(f"\nInput/Output layers (always included): {len(input_output_layers)}")
        for layer_name in input_output_layers:
            logger.info(f"  - {layer_name}")
        
        logger.info(f"Transformer block layers (selective): {len(transformer_layers)}")
        for layer_name in transformer_layers[:10]:  # 처음 10개만 출력
            logger.info(f"  - {layer_name}")
        if len(transformer_layers) > 10:
            logger.info(f"  ... and {len(transformer_layers) - 10} more layers")
        logger.info("=" * 80)
        
        # 1. Input/Output 레이어는 항상 포함
        for layer_name in input_output_layers:
            for idx in layer_to_indices[layer_name]:
                mask[idx] = True
        
        # 2. Transformer block 레이어들만 magnitude 기반 선택
        transformer_layer_magnitudes = {
            name: mag for name, mag in layer_magnitudes.items()
            if name in transformer_layers
        }
        
        selected_transformer_layers = set()
        
        if threshold is not None:
            # Threshold 기반: threshold 이상의 magnitude를 가진 레이어만 선택
            selected_transformer_layers = {
                name for name, mag in transformer_layer_magnitudes.items()
                if mag >= threshold
            }
            logger.info(f"Layer-based threshold selection: {len(selected_transformer_layers)}/{len(transformer_layers)} transformer layers selected (threshold={threshold})")
        
        elif top_k_ratio is not None:
            # Top-k ratio 기반 선택 로직
            if not (0.0 <= top_k_ratio <= 1.0):
                raise ValueError(f"top_k_ratio must be between 0.0 and 1.0, got {top_k_ratio}")
            
            # Input/Output 레이어와 forced_layers는 이미 포함됨
            already_included_layers = set(input_output_layers) | forced_layers
            
            if top_k_ratio_by_size:
                # 크기 기준: 레이어 크기 합계 기준으로 비율 계산
                total_layer_sizes = {}
                for layer_name, param_indices in layer_to_indices.items():
                    total_layer_sizes[layer_name] = sum(self.main_parameters[i].numel() for i in param_indices)
                
                total_size = sum(total_layer_sizes.values())
                already_included_size = sum(total_layer_sizes.get(name, 0) for name in already_included_layers)
                target_size = total_size * top_k_ratio
                remaining_target_size = max(0, target_size - already_included_size)
                
                # Magnitude 내림차순 정렬
                sorted_transformer = sorted(
                    transformer_layer_magnitudes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                accumulated_size = 0
                for layer_name, mag in sorted_transformer:
                    if accumulated_size >= remaining_target_size:
                        break
                    if layer_name not in already_included_layers:
                        accumulated_size += total_layer_sizes.get(layer_name, 0)
                        selected_transformer_layers.add(layer_name)
                
                total_selected_size = already_included_size + accumulated_size
                logger.info(
                    f"Layer-based top-k selection (by size): {len(selected_transformer_layers)}/{len(transformer_layers)} transformer layers selected "
                    f"({total_selected_size}/{total_size} elements, {100.0*total_selected_size/total_size:.1f}%), "
                    f"includes {len(input_output_layers)} input/output layers + "
                    f"{len([l for l in forced_layers if l not in input_output_layers])} forced by staleness + "
                    f"{len([l for l in selected_transformer_layers if l not in forced_layers])} by magnitude"
                )
            else:
                # 개수 기준: 레이어 개수 기준으로 비율 계산
                total_layers = len(layer_to_indices)
                already_included_count = len(already_included_layers)
                target_count = max(1, int(total_layers * top_k_ratio))
                remaining_target_count = max(0, target_count - already_included_count)
                
                # Magnitude 내림차순 정렬
                sorted_transformer = sorted(
                    transformer_layer_magnitudes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                selected_transformer_layers = {
                    name for name, _ in sorted_transformer[:remaining_target_count]
                    if name not in already_included_layers
                }
                
                actual_selected = len(selected_transformer_layers) + len(already_included_layers)
                logger.info(
                    f"Layer-based top-k selection (by count): {actual_selected}/{total_layers} layers selected "
                    f"({100.0*actual_selected/total_layers:.1f}%), "
                    f"includes {len(input_output_layers)} input/output layers + "
                    f"{len([l for l in forced_layers if l not in input_output_layers])} forced by staleness + "
                    f"{len(selected_transformer_layers)} by magnitude"
                )
        else:
            raise ValueError("Either threshold or top_k_ratio must be provided")
        
        # 3. Forced layers 추가 (Max Staleness)
        for layer_name in forced_layers:
            if layer_name in layer_to_indices:
                selected_transformer_layers.add(layer_name)
        
        # 4. 선택된 레이어의 모든 파라미터를 마스크에 포함
        for layer_name in selected_transformer_layers:
            if layer_name in layer_to_indices:
                for idx in layer_to_indices[layer_name]:
                    mask[idx] = True
        
        # 디버깅: 선택된 레이어 정보 출력
        logger.info("\n[DEBUG] Selected Layers:")
        logger.info(f"  Input/Output layers (always included): {len(input_output_layers)}")
        for layer_name in input_output_layers:
            param_count = len(layer_to_indices[layer_name])
            logger.info(f"    - {layer_name} ({param_count} parameters)")
        
        logger.info(f"  Transformer layers selected: {len(selected_transformer_layers)}")
        for layer_name in sorted(selected_transformer_layers):
            param_count = len(layer_to_indices[layer_name])
            magnitude = layer_magnitudes.get(layer_name, 0.0)
            logger.info(f"    - {layer_name} ({param_count} parameters, magnitude: {magnitude:.6f})")
        
        skipped_layers = set(transformer_layers) - selected_transformer_layers
        if skipped_layers:
            logger.info(f"  Transformer layers skipped: {len(skipped_layers)}")
            for layer_name in sorted(list(skipped_layers)[:10]):  # 처음 10개만 출력
                param_count = len(layer_to_indices[layer_name])
                magnitude = layer_magnitudes.get(layer_name, 0.0)
                logger.info(f"    - {layer_name} ({param_count} parameters, magnitude: {magnitude:.6f})")
            if len(skipped_layers) > 10:
                logger.info(f"    ... and {len(skipped_layers) - 10} more layers")
        
        logger.info("=" * 80)
        
        return mask

    def _create_mask_from_magnitudes(
        self, 
        magnitudes: List[float], 
        param_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        top_k_ratio: Optional[float] = None,
        top_k_ratio_by_size: bool = False,
        forced_indices: Optional[set] = None
    ) -> List[bool]:
        """Gradient magnitude를 기준으로 마스크 생성
        Transformer 모델의 관례에 따라 embedding과 lm_head는 항상 포함 (Dense하게 유지)
        top_k_ratio의 경우, embedding/lm_head와 forced_indices를 포함한 전체 파라미터 중에서 비율을 계산합니다.
        
        Args:
            threshold: magnitude threshold 이상의 파라미터만 선택
            top_k_ratio: top-k% 파라미터 선택 (0.0 ~ 1.0)
            top_k_ratio_by_size: True면 파라미터 크기 기준으로 비율 계산, False면 개수 기준
            forced_indices: Max Staleness로 강제 포함할 파라미터 인덱스 집합 (통신 비율에 포함됨)
        """
        if forced_indices is None:
            forced_indices = set()
        
        # 먼저 embedding/lm_head 인덱스 식별 (제외할 파라미터)
        embedding_indices = set()
        lm_head_indices = set()
        if param_names is not None:
            for idx, param_name in enumerate(param_names):
                # embedding 레이어 체크 (embed_tokens, embeddings 등)
                if 'embed' in param_name.lower() or 'embed_tokens' in param_name:
                    embedding_indices.add(idx)
                # output head 레이어 체크 (lm_head, output 등)
                elif 'lm_head' in param_name.lower() or param_name.endswith('.head') or 'output' in param_name.lower():
                    lm_head_indices.add(idx)
        
        excluded_indices = embedding_indices | lm_head_indices
        
        if threshold is not None:
            # Threshold 기반: threshold 이상의 magnitude를 가진 파라미터만 선택
            mask = [mag >= threshold for mag in magnitudes]
            logger.info(f"Gradient magnitude threshold-based selection: {sum(mask)}/{len(mask)} parameters selected (threshold={threshold})")
        elif top_k_ratio is not None:
            # Top-k ratio 기반 선택 로직:
            # 1. embedding/lm_head와 forced_indices는 항상 포함 (크기 비율에 포함됨)
            # 2. 전체 파라미터 크기(또는 개수)의 k%에 도달할 때까지 magnitude가 높은 파라미터 선택
            if not (0.0 <= top_k_ratio <= 1.0):
                raise ValueError(f"top_k_ratio must be between 0.0 and 1.0, got {top_k_ratio}")
            
            # 파라미터 크기 계산 (크기 기준인 경우)
            param_sizes = []
            if top_k_ratio_by_size:
                for param in self.main_parameters:
                    size = param.numel()  # 요소 개수 기준 (gradient 크기와 동일)
                    param_sizes.append(size)
            
            mask = [False] * len(magnitudes)
            
            # 1. embedding/lm_head는 항상 포함 (크기 비율에 포함)
            for idx in excluded_indices:
                mask[idx] = True
            
            # 2. forced_indices는 항상 포함 (크기 비율에 포함)
            for idx in forced_indices:
                mask[idx] = True
            
            # 이미 포함된 파라미터의 크기(또는 개수) 계산
            already_included_indices = excluded_indices | forced_indices
            if top_k_ratio_by_size:
                already_included_size = sum(param_sizes[i] for i in already_included_indices)
                total_size = sum(param_sizes)
                target_size = total_size * top_k_ratio
                remaining_target_size = max(0, target_size - already_included_size)
            else:
                already_included_count = len(already_included_indices)
                total_count = len(magnitudes)
                target_count = max(1, int(total_count * top_k_ratio))
                remaining_target_count = max(0, target_count - already_included_count)
            
            # 3. 나머지 파라미터 중 magnitude 기준으로 선택
            eligible_indices = [i for i in range(len(magnitudes)) if i not in already_included_indices]
            eligible_magnitudes = [(i, magnitudes[i]) for i in eligible_indices]
            
            # Magnitude를 내림차순으로 정렬
            sorted_eligible = sorted(eligible_magnitudes, key=lambda x: x[1], reverse=True)
            
            if top_k_ratio_by_size:
                # 크기 기준: 목표 크기에 도달할 때까지 선택
                accumulated_size = 0
                selected_by_magnitude = []
                for idx, mag in sorted_eligible:
                    if accumulated_size >= remaining_target_size:
                        break
                    accumulated_size += param_sizes[idx]
                    selected_by_magnitude.append(idx)
                    mask[idx] = True
                
                # 로그 출력
                total_selected_size = already_included_size + accumulated_size
                excluded_size = sum(param_sizes[i] for i in excluded_indices)
                forced_size = sum(param_sizes[i] for i in forced_indices if i not in excluded_indices)
                selected_size = accumulated_size
                logger.info(
                    f"Gradient magnitude top-k selection (by size): {sum(mask)}/{len(magnitudes)} parameters selected "
                    f"({total_selected_size}/{total_size} elements, {100.0*total_selected_size/total_size:.1f}%), "
                    f"includes {len(excluded_indices)} embedding/lm_head ({excluded_size} elements) + "
                    f"{len([i for i in forced_indices if i not in excluded_indices])} forced by staleness ({forced_size} elements) + "
                    f"{len(selected_by_magnitude)} by magnitude ({selected_size} elements)"
                )
            else:
                # 개수 기준: 목표 개수만큼 선택
                selected_by_magnitude = [idx for idx, _ in sorted_eligible[:remaining_target_count]]
                for idx in selected_by_magnitude:
                    mask[idx] = True
                
                # 로그 출력
                excluded_count = len(excluded_indices)
                forced_count = len([i for i in forced_indices if i not in excluded_indices])
                actual_selected = sum(mask)
                logger.info(
                    f"Gradient magnitude top-k selection (by count): {actual_selected}/{len(magnitudes)} parameters selected "
                    f"({100.0*actual_selected/len(magnitudes):.1f}%), "
                    f"includes {excluded_count} embedding/lm_head + {forced_count} forced by staleness + {len(selected_by_magnitude)} by magnitude"
                )
        else:
            raise ValueError("Either threshold or top_k_ratio must be provided")
        
        # Transformer 모델의 관례: embedding과 lm_head는 항상 포함 (Dense하게 유지)
        # (top_k_ratio의 경우 이미 위에서 처리됨)
        if param_names is not None and threshold is not None:
            embedding_count = 0
            lm_head_count = 0
            for idx in embedding_indices:
                if not mask[idx]:
                    mask[idx] = True
                    embedding_count += 1
            for idx in lm_head_indices:
                if not mask[idx]:
                    mask[idx] = True
                    lm_head_count += 1
            
            if embedding_count > 0 or lm_head_count > 0:
                logger.info(f"Transformer convention: {embedding_count} embedding parameters and {lm_head_count} output head parameters are always included (dense)")
        
        return mask

    def _log_magnitude_based_selection_sync(self, magnitudes: List[float], layer_magnitudes: Optional[Dict[str, float]] = None, epoch: Optional[int] = None, step: Optional[int] = None, current_loss: Optional[float] = None):
        """매 outer optimization마다 magnitude 기반 선택 상세 정보 출력 및 파일 저장 (동기 버전)"""
        if not self.enable_update_logs:
            return
            
        if self.param_names is None:
            return
        
        if len(magnitudes) != len(self.param_names):
            return
        
        # step 정보 가져오기
        if step is None:
            step = getattr(self, '_current_step', None)
        timestamp = datetime.now().isoformat()
        
        epoch_str = f"epoch {epoch}" if epoch is not None else "unknown epoch"
        metric_str = "Taylor Score" if self.gradient_importance_metric == "taylor" else "Magnitude"
        mode_str = "Layer-based" if self.gradient_magnitude_selection_mode == "layer" else "Parameter-based"
        logger.info("=" * 80)
        logger.info(f"[DEBUG] {mode_str} Gradient {metric_str} Selection - {epoch_str}")
        logger.info("=" * 80)
        
        # 파라미터별 정보 수집
        param_info = []
        for idx, (name, mag, is_selected) in enumerate(zip(self.param_names, magnitudes, self.param_update_mask)):
            layer_name = self._extract_layer_name(name)
            param_info.append({
                'idx': idx,
                'name': name,
                'layer_name': layer_name,
                'magnitude': mag,
                'selected': is_selected
            })
        
        # 전체 통계
        total_params = len(param_info)
        selected_param_count = sum(1 for p in param_info if p['selected'])
        skipped_param_count = total_params - selected_param_count
        
        if self.gradient_magnitude_selection_mode == "layer" and layer_magnitudes is not None:
            # 레이어 단위 통계 출력
            # 레이어 단위 magnitude 계산 (아직 계산되지 않은 경우)
            if layer_magnitudes is None:
                layer_magnitudes = self._compute_layer_magnitudes(magnitudes)
            
            # 레이어별 정보 수집
            layer_to_indices = self._group_parameters_by_layer(self.param_names)
            layer_info = []
            
            for layer_name, param_indices in layer_to_indices.items():
                layer_mag = layer_magnitudes.get(layer_name, 0.0)
                # 레이어의 파라미터 중 하나라도 선택되었으면 레이어가 선택된 것으로 간주
                is_selected = any(self.param_update_mask[idx] for idx in param_indices if idx < len(self.param_update_mask))
                param_count = len(param_indices)
                total_size = sum(self.main_parameters[i].numel() for i in param_indices if i < len(self.main_parameters))
                is_io = self._is_input_output_layer(layer_name)
                
                layer_info.append({
                    'layer_name': layer_name,
                    'magnitude': layer_mag,
                    'selected': is_selected,
                    'param_count': param_count,
                    'total_size': total_size,
                    'is_input_output': is_io,
                    'param_indices': param_indices
                })
            
            # 레이어 단위 통계 출력
            layer_info_sorted = sorted(layer_info, key=lambda x: x['magnitude'], reverse=True)
            
            # Input/Output 레이어와 Transformer block 레이어 분리
            io_layers = [l for l in layer_info_sorted if l['is_input_output']]
            transformer_layers = [l for l in layer_info_sorted if not l['is_input_output']]
            
            selected_layers = [l for l in layer_info_sorted if l['selected']]
            skipped_layers = [l for l in layer_info_sorted if not l['selected']]
            
            # 전체 통계
            total_layers = len(layer_info_sorted)
            
            logger.info(f"Summary (Layer-based):")
            logger.info(f"  - Total layers: {total_layers} (IO: {len(io_layers)}, Transformer: {len(transformer_layers)})")
            logger.info(f"  - Selected layers: {len(selected_layers)}/{total_layers} ({100.0 * len(selected_layers) / total_layers:.2f}%)")
            logger.info(f"  - Skipped layers: {len(skipped_layers)}/{total_layers} ({100.0 * len(skipped_layers) / total_layers:.2f}%)")
            logger.info(f"  - Total parameters: {total_params}")
            logger.info(f"  - Selected parameters: {selected_param_count} ({100.0 * selected_param_count / total_params:.2f}%)")
            logger.info(f"  - Skipped parameters: {skipped_param_count} ({100.0 * skipped_param_count / total_params:.2f}%)")
            
            if layer_magnitudes:
                layer_mags = list(layer_magnitudes.values())
                logger.info(f"  - Layer magnitude statistics: min={min(layer_mags):.6f}, max={max(layer_mags):.6f}, avg={sum(layer_mags)/len(layer_mags):.6f}")
            
            # 선택된 레이어 상위 20개 출력
            logger.info(f"\nTop {min(20, len(selected_layers))} Selected Layers (by magnitude):")
            for i, layer in enumerate(selected_layers[:20], 1):
                status = "✓ SELECTED"
                layer_type = "IO" if layer['is_input_output'] else "TRANS"
                logger.info(f"  {i:2d}. [{status}] [{layer_type}] {layer['layer_name']}")
                logger.info(f"      Params: {layer['param_count']}, Size: {layer['total_size']}, Magnitude: {layer['magnitude']:.6f}")
            
            # 선택되지 않은 레이어 중 magnitude가 큰 상위 10개 출력
            if skipped_layers:
                logger.info(f"\nTop {min(10, len(skipped_layers))} Skipped Layers (by magnitude):")
                for i, layer in enumerate(skipped_layers[:10], 1):
                    logger.info(f"  {i:2d}. [✗ SKIPPED] {layer['layer_name']}")
                    logger.info(f"      Params: {layer['param_count']}, Size: {layer['total_size']}, Magnitude: {layer['magnitude']:.6f}")
            
            # 레이어별 상세 통계
            logger.info(f"\nLayer-wise Statistics (sorted by magnitude):")
            logger.info(f"{'Layer':<40} {'Type':<8} {'Params':<8} {'Size':<12} {'Selected':<10} {'Magnitude':<12}")
            logger.info("-" * 100)
            for layer in layer_info_sorted[:30]:  # 상위 30개 레이어만 출력
                layer_type = "IO" if layer['is_input_output'] else "TRANS"
                selected_str = "YES" if layer['selected'] else "NO"
                logger.info(f"{layer['layer_name']:<40} {layer_type:<8} {layer['param_count']:<8} {layer['total_size']:<12} {selected_str:<10} {layer['magnitude']:<12.6f}")
        else:
            # 파라미터 단위 통계 출력
            logger.info(f"Summary (Parameter-based):")
            logger.info(f"  - Total parameters: {total_params}")
            logger.info(f"  - Selected parameters: {selected_param_count} ({100.0 * selected_param_count / total_params:.2f}%)")
            logger.info(f"  - Skipped parameters: {skipped_param_count} ({100.0 * skipped_param_count / total_params:.2f}%)")
            
            if magnitudes:
                logger.info(f"  - Parameter magnitude statistics: min={min(magnitudes):.6f}, max={max(magnitudes):.6f}, avg={sum(magnitudes)/len(magnitudes):.6f}")
            
            # 선택된 파라미터 상위 20개 출력
            selected_params = sorted([p for p in param_info if p['selected']], key=lambda x: x['magnitude'], reverse=True)
            logger.info(f"\nTop {min(20, len(selected_params))} Selected Parameters (by magnitude):")
            for i, param in enumerate(selected_params[:20], 1):
                logger.info(f"  {i:2d}. [✓ SELECTED] {param['name']}")
                logger.info(f"      Magnitude: {param['magnitude']:.6f}")
            
            # 선택되지 않은 파라미터 중 magnitude가 큰 상위 10개 출력
            skipped_params = sorted([p for p in param_info if not p['selected']], key=lambda x: x['magnitude'], reverse=True)
            if skipped_params:
                logger.info(f"\nTop {min(10, len(skipped_params))} Skipped Parameters (by magnitude):")
                for i, param in enumerate(skipped_params[:10], 1):
                    logger.info(f"  {i:2d}. [✗ SKIPPED] {param['name']}")
                    logger.info(f"      Magnitude: {param['magnitude']:.6f}")
        
        logger.info("=" * 80)
        
        # Loss 대비 비율 출력 (current_loss가 제공된 경우)
        if current_loss is not None and current_loss > 0:
            self._log_loss_ratio_internal(magnitudes, layer_magnitudes, current_loss, epoch, step)
        
        # 파일로 저장: CSV 형식 (각 파라미터별 상세 정보)
        self._save_parameter_selection_to_file(param_info, magnitudes, epoch, step, timestamp)
        
        # 파일로 저장: JSON 형식 (시간에 따른 변화 추적용)
        self._save_selection_history_to_json(param_info, magnitudes, epoch, step, timestamp)
    
    def _log_loss_ratio(self, magnitudes: List[float], layer_magnitudes: Optional[Dict[str, float]], current_loss: float, epoch: Optional[int]):
        """Loss 대비 중요도 비율 로깅 (매 epoch마다 항상 출력)"""
        if current_loss <= 0:
            return
        
        metric_name = "Taylor Score" if self.gradient_importance_metric == "taylor" else "Magnitude"
        
        logger.info("=" * 80)
        logger.info(f"[Loss Ratio Analysis] Epoch {epoch if epoch is not None else 'unknown'}")
        logger.info("=" * 80)
        logger.info(f"Current Total Loss: {current_loss:.6f}")
        logger.info(f"Importance Metric: {metric_name}")
        logger.info("")
        
        if self.gradient_magnitude_selection_mode == "layer" and layer_magnitudes is not None:
            # 레이어 단위 Loss 대비 비율
            logger.info("Layer-wise Loss Ratio (Score / Loss):")
            logger.info(f"{'Layer':<40} {'Score':<15} {'Ratio':<12} {'Interpretation':<20}")
            logger.info("-" * 90)
            
            sorted_layers = sorted(layer_magnitudes.items(), key=lambda x: x[1], reverse=True)
            for layer_name, score in sorted_layers[:20]:  # 상위 20개 레이어
                ratio = score / current_loss if current_loss > 0 else 0.0
                
                # 해석
                if ratio >= 0.01:  # 1% 이상
                    interpretation = "매우 중요 (건드리면 안됨)"
                elif ratio >= 0.001:  # 0.1% ~ 1%
                    interpretation = "유의미함"
                elif ratio >= 0.0001:  # 0.01% ~ 0.1%
                    interpretation = "영향력 미미"
                else:  # 0.01% 미만
                    interpretation = "안전한 가지치기 대상"
                
                logger.info(f"{layer_name:<40} {score:<15.6f} {ratio:<12.6f} ({ratio*100:.4f}%) {interpretation:<20}")
            
            # 전체 통계
            total_score = sum(layer_magnitudes.values())
            total_ratio = total_score / current_loss if current_loss > 0 else 0.0
            logger.info("")
            logger.info(f"Total Layer Score: {total_score:.6f}")
            logger.info(f"Total Ratio (Score/Loss): {total_ratio:.6f} ({total_ratio*100:.4f}%)")
        else:
            # 파라미터 단위 Loss 대비 비율
            total_score = sum(magnitudes)
            total_ratio = total_score / current_loss if current_loss > 0 else 0.0
            
            logger.info("Parameter-wise Loss Ratio:")
            logger.info(f"Total Parameter Score: {total_score:.6f}")
            logger.info(f"Total Ratio (Score/Loss): {total_ratio:.6f} ({total_ratio*100:.4f}%)")
            
            # 상위 20개 파라미터의 비율
            sorted_params = sorted(enumerate(magnitudes), key=lambda x: x[1], reverse=True)
            logger.info("")
            logger.info(f"{'Param Index':<12} {'Score':<15} {'Ratio':<12} {'Interpretation':<20}")
            logger.info("-" * 70)
            for idx, score in sorted_params[:20]:
                ratio = score / current_loss if current_loss > 0 else 0.0
                
                if ratio >= 0.01:
                    interpretation = "매우 중요"
                elif ratio >= 0.001:
                    interpretation = "유의미함"
                elif ratio >= 0.0001:
                    interpretation = "영향력 미미"
                else:
                    interpretation = "안전한 가지치기"
                
                param_name = self.param_names[idx] if self.param_names and idx < len(self.param_names) else f"param_{idx}"
                logger.info(f"{param_name[:40]:<40} {score:<15.6f} {ratio:<12.6f} ({ratio*100:.4f}%) {interpretation:<20}")
        
        logger.info("=" * 80)
        logger.info("")
    
    def _log_loss_ratio_internal(self, magnitudes: List[float], layer_magnitudes: Optional[Dict[str, float]], current_loss: float, epoch: Optional[int], step: Optional[int]):
        """내부 함수: Loss 대비 비율 로깅 (enable_update_logs와 무관하게 항상 출력)"""
        self._log_loss_ratio(magnitudes, layer_magnitudes, current_loss, epoch)

    def _log_magnitude_based_selection(self, magnitudes: List[float], layer_magnitudes: Optional[Dict[str, float]] = None, epoch: Optional[int] = None, step: Optional[int] = None, current_loss: Optional[float] = None):
        """매 outer optimization마다 magnitude 기반 레이어 선택 상세 정보 출력 및 파일 저장 (비동기 버전) - 레이어 단위"""
        if not self.enable_update_logs:
            return
            
        if self._log_executor is None:
            return
            
        # 비동기로 실행 (메인 학습 루프를 블로킹하지 않음)
        self._log_executor.submit(self._log_magnitude_based_selection_sync, magnitudes, layer_magnitudes, epoch, step, current_loss)

    def _save_parameter_selection_to_file(
        self, 
        param_info: List[Dict], 
        magnitudes: List[float], 
        epoch: Optional[int], 
        step: Optional[int],
        timestamp: str
    ):
        """파라미터 선택 정보를 CSV 파일로 저장"""
        import csv
        
        csv_filename = os.path.join(self.log_dir, f"parameter_selection_epoch_{epoch if epoch is not None else 'unknown'}_step_{step if step is not None else 'unknown'}.csv")
        
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'step', 'timestamp', 'param_idx', 'param_name', 'layer_name', 'magnitude', 'selected', 'rank_by_magnitude']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                # Magnitude 기준으로 정렬하여 rank 계산
                sorted_params = sorted(enumerate(param_info), key=lambda x: x[1]['magnitude'], reverse=True)
                rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_params)}
                
                for param in param_info:
                    writer.writerow({
                        'epoch': epoch if epoch is not None else '',
                        'step': step if step is not None else '',
                        'timestamp': timestamp,
                        'param_idx': param['idx'],
                        'param_name': param['name'],
                        'layer_name': param['layer_name'],
                        'magnitude': f"{param['magnitude']:.8f}",
                        'selected': 'True' if param['selected'] else 'False',
                        'rank_by_magnitude': rank_map[param['idx']]
                    })
            
            logger.info(f"Saved parameter selection to CSV: {csv_filename}")
        except Exception as e:
            logger.warning(f"Failed to save parameter selection to CSV: {e}")

    def _save_selection_history_to_json(
        self,
        param_info: List[Dict],
        magnitudes: List[float],
        epoch: Optional[int],
        step: Optional[int],
        timestamp: str
    ):
        """파라미터 선택 이력을 JSON 파일로 저장 (시간에 따른 변화 추적용)"""
        import json
        
        # 현재 선택 정보 저장
        selection_record = {
            'epoch': epoch,
            'step': step,
            'timestamp': timestamp,
            'total_parameters': len(param_info),
            'selected_count': sum(1 for p in param_info if p['selected']),
            'skipped_count': sum(1 for p in param_info if not p['selected']),
            'magnitude_stats': {
                'min': float(min(magnitudes)) if magnitudes else 0.0,
                'max': float(max(magnitudes)) if magnitudes else 0.0,
                'mean': float(np.mean(magnitudes)) if magnitudes else 0.0,
                'median': float(np.median(magnitudes)) if magnitudes else 0.0,
                'std': float(np.std(magnitudes)) if magnitudes else 0.0
            },
            'selected_parameters': [
                {
                    'idx': p['idx'],
                    'name': p['name'],
                    'layer_name': p['layer_name'],
                    'magnitude': float(p['magnitude'])
                }
                for p in param_info if p['selected']
            ],
            'all_parameters': [
                {
                    'idx': p['idx'],
                    'name': p['name'],
                    'layer_name': p['layer_name'],
                    'magnitude': float(p['magnitude']),
                    'selected': p['selected']
                }
                for p in param_info
            ]
        }
        
        # 이력에 추가
        self._parameter_selection_history.append(selection_record)
        
        # JSON 파일로 저장 (전체 이력)
        json_filename = os.path.join(self.log_dir, "parameter_selection_history.json")
        try:
            with open(json_filename, 'w') as jsonfile:
                json.dump(self._parameter_selection_history, jsonfile, indent=2)
            logger.info(f"Saved parameter selection history to JSON: {json_filename}")
        except Exception as e:
            logger.warning(f"Failed to save parameter selection history to JSON: {e}")
        
        # 최근 선택 정보만 별도 파일로 저장 (빠른 접근용)
        recent_filename = os.path.join(self.log_dir, f"parameter_selection_epoch_{epoch if epoch is not None else 'unknown'}_step_{step if step is not None else 'unknown'}.json")
        try:
            with open(recent_filename, 'w') as jsonfile:
                json.dump(selection_record, jsonfile, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save recent parameter selection to JSON: {e}")

    def _grads_from_optimizer(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated optimizer
        주의: 이 함수는 초기화 및 마스크 업데이트 시에만 사용되며,
        실제 통신에 사용되는 값은 compute_and_load_pseudo_grad_into_averager에서 계산됩니다.
        """
        param_groups = self.offloaded_optimizer.param_groups
        param_idx = 0
        for param_group in param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)                
                grad = param.grad
                
                # Selective layer update가 활성화된 경우, 마스크에 따라 gradient 반환
                if self.param_update_mask is not None:
                    if self.param_update_mask[param_idx]:
                        yield grad
                    # skipped params는 yield하지 않음 - 통신에서 완전히 제외
                else:
                    yield grad
                param_idx += 1

    def schedule_step(self, scheduled_time: Optional[DHTExpiration] = None, **kwargs) -> StepControl:

        """
        Begin matchmaking: look for a group of peers and prepare for averaging gradients at a specified time.

        :param scheduled_time: expected time when to perform all-reduce. Can be changed using control.scheduled_time
        :param kwargs: any additional keyword args from DecentralizedAverager.step, such as gather, allow_retries, etc
        :note: setting weight at this stage is not supported, please leave this parameter as None
        :returns: step_control - a handle that can be passed into GradientAverager.step to use the pre-scheduled group
        :note: in the current implementation, each step_control can only be used in one step.
        """
        
        assert kwargs.get("weight") is None, "setting weight in schedule_step is not supported"
        return super().step(scheduled_time=scheduled_time, wait=False, require_trigger=True, **kwargs)
    def step(
        self,
        control: Optional[StepControl] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        epoch: Optional[int] = None,
        current_loss: Optional[float] = None,
        **kwargs,        
    ):
        """
        Average accumulated gradients with peers, optionally load averaged gradients and reset accumulators

        :param weight: overrides the averaging weight; by default, weight equals the number of accumulated samples
        :param reset_accumulators: by default, set local gradient accumulators to zeros after averaging succeeds
        :param control: reuse a pre-arranged group of peers (or a matchmaking in progress) from averager.schedule_step
        :param timeout: if specified, await for averaging round for at most this number of seconds (if wait=True)
        :param wait: if True, await for the step to finish (or fail), otherwise run all-reduce in background
        :param epoch: current epoch number for debugging
        """
        # epoch 정보 저장 (디버그용)
        if epoch is not None:
            self._current_epoch = epoch
        
        # Residual norm threshold 체크는 더 이상 필요 없음
        # 이제 모든 파라미터를 업데이트하므로, skip된 파라미터의 누적 차이 문제가 해결됩니다.
        # residual_forced_params는 빈 집합으로 유지 (하위 호환성)
        residual_forced_params = set()
        
        # Gradient magnitude 기반 tensor 선택 및 Token weight 계산을 병렬로 처리
        has_magnitude_config = (self.gradient_magnitude_threshold is not None or self.gradient_magnitude_top_k_ratio is not None)
        has_param_names = self.param_names is not None
        
        magnitudes = None
        new_mask = None
        
        # Gradient importance scores 수집 함수 (병렬 실행용, 내부에서 시간 측정)
        def collect_magnitudes():
            if has_magnitude_config and has_param_names:
                time_start = time.perf_counter()
                # Metric에 따라 magnitude 또는 taylor 계산
                if self.gradient_importance_metric == "taylor":
                    local_scores = self._compute_taylor_scores()
                else:  # magnitude (default)
                    local_scores = self._compute_gradient_magnitudes()
                
                magnitudes = self._collect_and_average_gradient_magnitudes(
                    local_scores, 
                    epoch=epoch if epoch is not None else 0,
                    timeout=timeout if timeout is not None else 300.0
                )
                time_end = time.perf_counter()
                execution_time = time_end - time_start
                return magnitudes, execution_time
            return None, 0.0
        
        # Token weight 계산 함수 (병렬 실행용, 내부에서 시간 측정)
        def compute_token_weight():
            if self.token_weighted_aggregation and control is not None:
                time_start = time.perf_counter()
                weight = self._compute_token_weight(control, log_prefix="GradAverager Token-weighted aggregation")
                time_end = time.perf_counter()
                execution_time = time_end - time_start
                return weight, execution_time
            return None, 0.0
        
        # 병렬로 실행
        sync_wait_time = 0.0
        token_weight_result = None
        token_weight_wait_time = 0.0
        magnitude_selection_time = 0.0
        
        if (has_magnitude_config and has_param_names) and self.token_weighted_aggregation:
            # 두 작업 모두 병렬 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                magnitude_future = executor.submit(collect_magnitudes)
                token_weight_future = executor.submit(compute_token_weight)
                
                magnitudes, magnitude_selection_time = magnitude_future.result()
                token_weight_result, token_weight_wait_time = token_weight_future.result()
        elif has_magnitude_config and has_param_names:
            # Gradient magnitude만 실행
            magnitudes, magnitude_selection_time = collect_magnitudes()
        elif self.token_weighted_aggregation:
            # Token weight만 실행
            token_weight_result, token_weight_wait_time = compute_token_weight()
        
        # Warm-up: 초기 epoch 동안은 모든 레이어 전송
        effective_top_k_ratio = self.gradient_magnitude_top_k_ratio
        if self.enable_warmup and epoch is not None and self.original_top_k_ratio is not None:
            if epoch < self.warmup_epochs:
                # Warm-up 기간: 모든 레이어 전송 (top_k_ratio = 1.0)
                effective_top_k_ratio = 1.0
                if epoch == 0:
                    logger.info(f"Warm-up phase: sending all layers for first {self.warmup_epochs} epochs")
            else:
                # Warm-up 이후: 원본 top_k_ratio 사용
                effective_top_k_ratio = self.original_top_k_ratio
        
        # Magnitude 기반 마스크 생성 (모드에 따라 레이어 단위 또는 파라미터 단위)
        layer_magnitudes = None
        new_mask = None
        
        if magnitudes is not None and has_param_names:
            if self.gradient_magnitude_selection_mode == "layer":
                # 레이어 단위 선택 모드
                # 파라미터 단위 magnitude를 레이어 단위로 집계
                layer_magnitudes = self._compute_layer_magnitudes(magnitudes)
                
                # Max Staleness: 먼저 강제 포함할 레이어 식별 (레이어 단위)
                forced_layers_set = set()
                if self.enable_max_staleness and self.layer_staleness_counters is not None:
                    forced_layers_set = {
                        layer_name for layer_name, staleness in self.layer_staleness_counters.items()
                        if staleness >= self.max_staleness
                    }
                    if len(forced_layers_set) > 0:
                        logger.info(f"Max Staleness (layer-based): {len(forced_layers_set)} layers will be forced to update (staleness >= {self.max_staleness})")
                        for layer_name in sorted(forced_layers_set):
                            logger.info(f"  - {layer_name} (staleness: {self.layer_staleness_counters[layer_name]})")
                
                # 레이어 단위 마스크 생성
                new_mask = self._create_layer_based_mask(
                    layer_magnitudes,
                    threshold=self.gradient_magnitude_threshold,
                    top_k_ratio=effective_top_k_ratio,
                    top_k_ratio_by_size=self.gradient_magnitude_top_k_ratio_by_size,
                    forced_layers=forced_layers_set if self.enable_max_staleness else None
                )
                
                # Max Staleness: 레이어 단위 카운터 업데이트
                if self.enable_max_staleness and self.layer_staleness_counters is not None:
                    layer_to_indices = self._group_parameters_by_layer(self.param_names)
                    
                    # 선택된 레이어 확인
                    selected_layers = set()
                    for layer_name, param_indices in layer_to_indices.items():
                        # 레이어의 파라미터 중 하나라도 선택되었으면 레이어가 선택된 것으로 간주
                        if any(new_mask[idx] for idx in param_indices):
                            selected_layers.add(layer_name)
                    
                    # Staleness 카운터 업데이트
                    forced_count = 0
                    for layer_name in self.layer_staleness_counters.keys():
                        if layer_name in selected_layers:
                            self.layer_staleness_counters[layer_name] = 0  # 선택됐으면 초기화
                        else:
                            self.layer_staleness_counters[layer_name] += 1  # 선택 안 됐으면 +1
                            if layer_name in forced_layers_set:
                                forced_count += 1
                    
                    if forced_count > 0:
                        logger.info(f"Max Staleness (layer-based): {forced_count} layers forced to update (staleness >= {self.max_staleness})")
                    
                    # 디버깅: 레이어별 staleness 정보 출력
                    logger.info("\n[DEBUG] Layer Staleness Counters:")
                    sorted_layers = sorted(self.layer_staleness_counters.items(), key=lambda x: x[1], reverse=True)
                    for layer_name, staleness in sorted_layers[:20]:  # 상위 20개만 출력
                        status = "SELECTED" if layer_name in selected_layers else "SKIPPED"
                        forced = "FORCED" if layer_name in forced_layers_set else ""
                        logger.info(f"  {layer_name:<40} Staleness: {staleness:<5} {status} {forced}")
                    if len(sorted_layers) > 20:
                        logger.info(f"  ... and {len(sorted_layers) - 20} more layers")
                    logger.info("")
            else:  # parameter mode
                # 파라미터 단위 선택 모드
                # Max Staleness: 먼저 강제 포함할 파라미터 식별 (파라미터 단위)
                forced_indices_set = set()
                if self.enable_max_staleness and self.param_staleness_counters is not None:
                    forced_indices_set = {
                        idx for idx, staleness in enumerate(self.param_staleness_counters)
                        if staleness >= self.max_staleness
                    }
                    if len(forced_indices_set) > 0:
                        logger.info(f"Max Staleness (parameter-based): {len(forced_indices_set)} parameters will be forced to update (staleness >= {self.max_staleness})")
                
                # 파라미터 단위 마스크 생성
                new_mask = self._create_mask_from_magnitudes(
                    magnitudes,
                    param_names=self.param_names,
                    threshold=self.gradient_magnitude_threshold,
                    top_k_ratio=effective_top_k_ratio,
                    top_k_ratio_by_size=self.gradient_magnitude_top_k_ratio_by_size,
                    forced_indices=forced_indices_set if self.enable_max_staleness else None
                )
                
                # Max Staleness: 파라미터 단위 카운터 업데이트
                if self.enable_max_staleness and self.param_staleness_counters is not None:
                    # Staleness 카운터 업데이트
                    forced_count = 0
                    for idx in range(len(self.param_staleness_counters)):
                        if new_mask[idx]:
                            self.param_staleness_counters[idx] = 0  # 선택됐으면 초기화
                        else:
                            self.param_staleness_counters[idx] += 1  # 선택 안 됐으면 +1
                            if idx in forced_indices_set:
                                forced_count += 1
                    
                    if forced_count > 0:
                        logger.info(f"Max Staleness (parameter-based): {forced_count} parameters forced to update (staleness >= {self.max_staleness})")
            
        # 마스크 업데이트 및 averaged_grads 재생성
        if new_mask is not None:
            self.param_update_mask = new_mask
            
            # 로깅 (enable_update_logs가 활성화된 경우)
            if self.enable_update_logs and magnitudes is not None:
                if self.gradient_magnitude_selection_mode == "layer" and layer_magnitudes is not None:
                    # 레이어 단위 로깅
                    self._log_magnitude_based_selection(magnitudes, layer_magnitudes, epoch, step=getattr(self, '_current_step', None), current_loss=current_loss)
                elif self.gradient_magnitude_selection_mode == "parameter":
                    # 파라미터 단위 로깅 (레이어 단위 집계 없이)
                    self._log_magnitude_based_selection(magnitudes, None, epoch, step=getattr(self, '_current_step', None), current_loss=current_loss)
            
            # Loss 대비 비율 로깅 (매 epoch마다 항상 출력)
            if magnitudes is not None and current_loss is not None and current_loss > 0:
                self._log_loss_ratio(magnitudes, layer_magnitudes, current_loss, epoch)
            
            with self.lock_averaged_tensors:
                new_averaged_grads = tuple(grad for grad in self._grads_from_optimizer())
                self._averaged_tensors = tuple(new_averaged_grads)
                for tensor in self._averaged_tensors:
                    tensor.share_memory_()
                self.total_size = sum(map(torch.Tensor.numel, self._averaged_tensors))
                from hivemind.averaging.averager import compute_schema_hash
                self.schema_hash = compute_schema_hash(self._averaged_tensors)
            
            # averager 프로세스에 업데이트 전달
            if self.is_alive():
                try:
                    self._outer_pipe.send(("_update_averaged_tensors", [list(new_averaged_grads)], {
                        "total_size": self.total_size,
                        "schema_hash": self.schema_hash
                    }))
                except Exception as e:
                    logger.warning(f"Failed to send _update_averaged_tensors: {e}")
        
        # Token weight 설정
        if token_weight_result is not None:
            sync_wait_time = token_weight_wait_time
        elif 'weight' in kwargs:
            control.weight = kwargs['weight']
            kwargs.pop('weight', None)
        
        if control is None:
            control = self.schedule_step(timeout=timeout, **kwargs)
        
        time_0_compute_pseudo_grad = time.perf_counter()
        self.compute_and_load_pseudo_grad_into_averager()
        time_1_compute_pseudo_grad = time.perf_counter()
        compute_pseudo_grad_time = time_1_compute_pseudo_grad - time_0_compute_pseudo_grad

        control.allow_allreduce()

        # Local step 동기화 (token_weighted_aggregation이 비활성화된 경우만)
        if (not self.token_weighted_aggregation and 
            wait and self.num_inner_steps is not None and epoch is not None and self.expected_num_peers is not None):
            try:
                sync_wait_time = wait_for_all_nodes_local_step_complete(
                    dht=self.dht,
                    epoch=epoch,
                    num_inner_steps=self.num_inner_steps,
                    expected_num_peers=self.expected_num_peers,
                    log_fn=lambda msg: None,  # 로그 제거
                    timeout=timeout if timeout is not None else 300.0,
                )
            except Exception as e:
                logger.warning(f"Failed to synchronize local step completion via DHT: {e}. Proceeding with control.result() anyway.")

        # 시간 측정 로그 출력
        logger.log(
            logging.INFO,
            f"[TIMING] Gradient magnitude selection: {magnitude_selection_time:.6f} sec, "
            f"Token weight sync: {token_weight_wait_time:.6f} sec, "
            f"Sync wait (GPU idle): {sync_wait_time:.6f} sec"
        )
        
        # return control.result(timeout) if wait else control
        if wait:
            time_0_allreduce = time.perf_counter()
            return_value = control.result(timeout)
            time_1_allreduce = time.perf_counter()
            allreduce_time = time_1_allreduce - time_0_allreduce
            logger.log(
                logging.INFO,
                f"[TIMING] All-reduce networking time: {allreduce_time:.6f} sec"
            )
            return return_value
        else: 
            return control
        

    @torch.no_grad()
    def compute_and_load_pseudo_grad_into_averager(self):
        """compute pseudo gradient by subtracting the offloaded optimizer parameters with the main parameters and load them in the averager"""
        opt_parameters = [param for group in self.offloaded_optimizer.param_groups for param in group["params"]]
        
        # 파라미터 업데이트 추적을 위한 정보 수집 (enable_update_logs가 True인 경우에만)
        epoch = getattr(self, '_current_epoch', None)
        step = getattr(self, '_current_step', None)
        timestamp = datetime.now().isoformat() if self.enable_update_logs else None
        update_records = [] if self.enable_update_logs else None
        
        with self.get_tensors() as averaged_grads:
            param_idx = 0
            grad_idx = 0  # averaged_grads의 인덱스
            updated_param_names = []
            skipped_param_names = []
            
            # param_update_mask가 있을 때, 선택된 파라미터 수 확인
            if self.param_update_mask is not None:
                expected_grad_count = sum(self.param_update_mask)
                if len(averaged_grads) != expected_grad_count:
                    raise RuntimeError(
                        f"averaged_grads 크기 불일치: averaged_grads에는 {len(averaged_grads)}개의 텐서가 있지만, "
                        f"param_update_mask에 따르면 {expected_grad_count}개의 텐서가 필요합니다. "
                        f"이는 gradient_magnitude 기반 선택을 사용할 때 초기화 시점과 step 시점의 마스크가 다를 수 있기 때문입니다. "
                        f"초기화 시점에 param_update_mask를 설정하거나, averaged_grads를 동적으로 재생성해야 합니다."
                    )
            
            for opt_param, main_param in zip(opt_parameters, self.main_parameters):
                param_name = self.param_names[param_idx] if self.param_names and param_idx < len(self.param_names) else f"param_{param_idx}"
                
                # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                if self.param_update_mask is not None:
                    if self.param_update_mask[param_idx]:
                        # 선택된 파라미터만 pseudo gradient 계산
                        # opt_param과 main_param의 차이가 곧 누적된 gradient (Implicit Accumulation)
                        grad = opt_param.data - main_param.detach().to(opt_param.device)
                        
                        if grad_idx >= len(averaged_grads):
                            raise RuntimeError(
                                f"grad_idx ({grad_idx})가 averaged_grads의 길이 ({len(averaged_grads)})를 초과했습니다. "
                                f"param_idx={param_idx}, param_name={param_name}"
                            )
                        if averaged_grads[grad_idx].shape != grad.shape:
                            raise RuntimeError(
                                f"텐서 크기 불일치: averaged_grads[{grad_idx}]의 크기는 {averaged_grads[grad_idx].shape}이지만, "
                                f"grad의 크기는 {grad.shape}입니다. param_idx={param_idx}, param_name={param_name}. "
                                f"이는 초기화 시점과 step 시점의 param_update_mask가 다르거나, "
                                f"opt_parameters와 main_parameters의 순서가 일치하지 않을 수 있습니다."
                            )
                        # Gradient 정보 기록 (enable_update_logs가 True인 경우에만)
                        averaged_grads[grad_idx].copy_(grad, non_blocking=True)
                        if self.enable_update_logs and update_records:
                            # 업데이트 정보 기록 (gradient norm 및 선택 여부 포함)
                            update_records.append({
                                'param_idx': param_idx,
                                'param_name': param_name,
                                'grad_norm': float(grad.norm().item()),
                                'was_selected': True,  # 이 블록에 들어왔으므로 선택됨
                                'was_forced_by_residual': False,  # residual은 더 이상 사용하지 않음
                            })
                        
                        updated_param_names.append(param_name)
                        grad_idx += 1
                    else:
                        # 선택되지 않은 레이어는 통신에서 제외됨
                        # 이제 모든 파라미터를 업데이트하므로, skip된 파라미터의 gradient는 0으로 설정되어
                        # outer optimizer step 후에도 변화가 없습니다.
                        skipped_param_names.append(param_name)
                else:
                    # 모든 파라미터 업데이트
                    # opt_param과 main_param의 차이가 곧 누적된 gradient (Implicit Accumulation)
                    grad = opt_param.data - main_param.detach().to(opt_param.device)
                    
                    if grad_idx >= len(averaged_grads):
                        raise RuntimeError(
                            f"grad_idx ({grad_idx})가 averaged_grads의 길이 ({len(averaged_grads)})를 초과했습니다. "
                            f"param_idx={param_idx}, param_name={param_name}"
                        )
                    if averaged_grads[grad_idx].shape != grad.shape:
                        raise RuntimeError(
                            f"텐서 크기 불일치: averaged_grads[{grad_idx}]의 크기는 {averaged_grads[grad_idx].shape}이지만, "
                            f"grad의 크기는 {grad.shape}입니다. param_idx={param_idx}, param_name={param_name}. "
                            f"opt_parameters와 main_parameters의 순서가 일치하지 않을 수 있습니다."
                        )
                    # Gradient 정보 기록 (enable_update_logs가 True인 경우에만)
                    averaged_grads[grad_idx].copy_(grad, non_blocking=True)
                    if self.enable_update_logs and update_records:
                        # 업데이트 정보 기록 (gradient norm 및 선택 여부 포함)
                        update_records.append({
                            'param_idx': param_idx,
                            'param_name': param_name,
                            'grad_norm': float(grad.norm().item()),
                            'was_selected': True,  # 이 블록에 들어왔으므로 선택됨
                            'was_forced_by_residual': False,  # residual은 더 이상 사용하지 않음
                        })
                    
                    if not updated_param_names:  # 첫 번째 호출 시에만 로깅
                        updated_param_names.append(param_name)
                    grad_idx += 1
                param_idx += 1
            
            
            # 파라미터 업데이트 이력 저장 (enable_update_logs가 True인 경우에만)
            if self.enable_update_logs and update_records:
                self._save_parameter_updates_to_file(update_records, epoch, step, timestamp)

    def _save_parameter_updates_to_file_sync(
        self,
        update_records: List[Dict],
        epoch: Optional[int],
        step: Optional[int],
        timestamp: str
    ):
        """파라미터 업데이트 정보를 파일로 저장 (동기 버전)"""
        import csv
        import json
        
        # CSV 파일로 저장
        csv_filename = os.path.join(self.log_dir, f"parameter_updates_epoch_{epoch if epoch is not None else 'unknown'}_step_{step if step is not None else 'unknown'}.csv")
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'step', 'timestamp', 'param_idx', 'param_name', 'grad_norm', 'was_selected', 'was_forced_by_residual']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in update_records:
                    writer.writerow({
                        'epoch': epoch if epoch is not None else '',
                        'step': step if step is not None else '',
                        'timestamp': timestamp,
                        **{k: f"{v:.8f}" if isinstance(v, float) else v for k, v in record.items()}
                    })
            
            logger.info(f"Saved parameter updates to CSV: {csv_filename}")
        except Exception as e:
            logger.warning(f"Failed to save parameter updates to CSV: {e}")
        
        # JSON 파일로 저장 (이력 추적용)
        update_record = {
            'epoch': epoch,
            'step': step,
            'timestamp': timestamp,
            'total_updates': len(update_records),
            'updates': update_records
        }
        
        self._parameter_update_history.append(update_record)
        
        json_filename = os.path.join(self.log_dir, "parameter_update_history.json")
        try:
            with open(json_filename, 'w') as jsonfile:
                json.dump(self._parameter_update_history, jsonfile, indent=2)
            logger.info(f"Saved parameter update history to JSON: {json_filename}")
        except Exception as e:
            logger.warning(f"Failed to save parameter update history to JSON: {e}")
    
    def _save_parameter_updates_to_file(
        self,
        update_records: List[Dict],
        epoch: Optional[int],
        step: Optional[int],
        timestamp: str
    ):
        """파라미터 업데이트 정보를 파일로 저장 (비동기 버전)"""
        if not self.enable_update_logs:
            return
            
        if self._log_executor is None:
            return
            
        # 비동기로 실행 (메인 학습 루프를 블로킹하지 않음)
        self._log_executor.submit(self._save_parameter_updates_to_file_sync, update_records, epoch, step, timestamp)

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for
        Residual buffer는 더 이상 사용하지 않으므로 초기화 로직이 필요 없음.
        """
        self._new_averaged_grads = False
    
    async def _update_averaged_tensors(self, new_tensors: List[torch.Tensor], total_size: int, schema_hash: int):
        """
        averager 프로세스에서 _averaged_tensors를 업데이트합니다.
        
        메인 프로세스에서 새로운 텐서 리스트를 받아서 averager 프로세스의 _averaged_tensors를 업데이트합니다.
        텐서는 이미 share_memory_()를 호출했으므로 공유 메모리를 통해 접근 가능합니다.
        """
        old_tensor_count = len(self._averaged_tensors)
        old_total_size = self.total_size
        
        with self.lock_averaged_tensors:
            self._averaged_tensors = tuple(new_tensors)
            self.total_size = total_size
            self.schema_hash = schema_hash
        
        logger.info(
            f"[DEBUG averager process] Updated _averaged_tensors: "
            f"tensor count {old_tensor_count} -> {len(self._averaged_tensors)}, "
            f"total_size {old_total_size} -> {self.total_size}"
        )


class DiloCoProgressTracker(ProgressTracker):
    global_progress: GlobalTrainingProgress
    local_progress: LocalTrainingProgress

    def __init__(self, batch_size: int, num_inner_steps: int, **kwargs):
        self.batch_size = batch_size
        self.num_inner_steps = num_inner_steps
        super().__init__(**kwargs)

    @property
    def ready_to_update_epoch(self) -> bool:
        """Whether or not this peer can increment epoch right away."""
        return (
            self.global_epoch > self.local_progress.epoch
            or self.local_progress.samples_accumulated
            >= self.target_batch_size  # here we track local progress as each diloco worker need to do num_inner_steps (for now)
            # or get_dht_time() >= self.global_progress.eta_next_epoch # disabled for our test
        )

    @property
    def estimated_next_update_time(self) -> DHTExpiration:
        """Estimate (absolute) time when this peer should increment epoch"""
        if self.ready_to_update_epoch:
            return get_dht_time()

        samples_remaining_to_next_epoch = max(0, self.target_batch_size - self.local_progress.samples_accumulated)
        return samples_remaining_to_next_epoch / self.performance_ema.samples_per_second

    @property
    def local_step(self) -> int:
        return self.local_progress.samples_accumulated // self.batch_size

    @property
    def real_step(self) -> int:
        return self.local_step + self.local_progress.epoch * self.batch_size
    
    def _parse_swarm_progress_data(self, metadata: TrainingProgressSchema) -> GlobalTrainingProgress:
        """Read performance statistics reported by peers, estimate progress towards next batch
        This function is copy paste from hivemind. Only difference is that if fix the ETA estimation.
        """
        current_time = get_dht_time()

        if not isinstance(metadata, dict) or len(metadata) == 0:
            logger.log(self.status_loglevel, f"Found no active peers: {metadata}")
            samples_remaining_to_next_epoch = max(0, self.target_batch_size - self.local_progress.samples_accumulated)
            local_eta_next_epoch = samples_remaining_to_next_epoch / self.performance_ema.samples_per_second

            return GlobalTrainingProgress(
                self.local_progress.epoch,
                self.local_progress.samples_accumulated,
                self.target_batch_size,
                num_peers=0,
                num_clients=0,
                eta_next_epoch=current_time + local_eta_next_epoch,
                next_fetch_time=current_time + self.default_refresh_period,
            )

        valid_peer_entries = [
            LocalTrainingProgress.parse_obj(peer_state.value)
            for peer_state in metadata.values()
            if peer_state.value is not None
        ]

        num_peers = len(valid_peer_entries)
        num_clients = sum(peer.client_mode for peer in valid_peer_entries)

        global_epoch = self.local_progress.epoch
        for peer in valid_peer_entries:
            if not peer.client_mode:
                global_epoch = max(global_epoch, peer.epoch)

        total_samples_accumulated = 0
        total_samples_per_second = self.performance_ema.eps
        
        estimated_time_to_next_epoch = 0

        for peer in valid_peer_entries:
            total_samples_per_second += peer.samples_per_second
            if peer.epoch == global_epoch:
                samples_remaining_to_next_epoch = max(0, peer.target_batch_size - peer.samples_accumulated)
                local_eta_next_epoch = samples_remaining_to_next_epoch / peer.samples_per_second

                estimated_time_to_next_epoch = max(estimated_time_to_next_epoch, local_eta_next_epoch)

            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        time_to_next_fetch = float(
            np.clip(
                a=estimated_time_to_next_epoch,
                a_min=self.min_refresh_period,
                a_max=self.max_refresh_period,
            )
        )

        logger.log(
            self.status_loglevel,
            f"{self.prefix} has taken {self.local_step} local steps. Peers: {num_peers}, epoch: {self.local_progress.epoch}, steps: {self.real_step}. ETA: {estimated_time_to_next_epoch:.2f}",
        )

        return GlobalTrainingProgress(
            global_epoch,
            total_samples_accumulated,
            target_batch_size=self.target_batch_size,
            num_peers=num_peers,
            num_clients=num_clients,
            eta_next_epoch=current_time + estimated_time_to_next_epoch,
            next_fetch_time=current_time + time_to_next_fetch,
        )


class AllReduceStrategy(Enum):
    """
    DiLoCo support multiple strategy to trigger the pseudo gradient averaging step.

    stregy:
        * WAIT_FOR_ALL: DiLoCo will wait for all peers to finish their local updates before triggering the all reduce step
            use this strategy when you trust all of your peers
        * NO_WAIT: The fastest peer will trigger the all reduce as soon as it reach its local steps (modulo the amount of time it need to wait because of the `matchmaking_time`)
            use this strategy when some of your peers are unreliable
    """

    WAIT_FOR_ALL = "WAIT_FOR_ALL"
    NO_WAIT = "NO_WAIT"


DEFAULT_TIMEOUT_WAITING_FOR_PEERS = 600


class DiLoCoOptimizer(Optimizer):
    """
    DiLoCo optimizer extend Hivemind's Optimizer to support DiLoCo training with local updates, requiring less bandwidth to train
    and still converge.

    Pseudo gradient is the difference between the weight before and after the multiple local update of the inner optimizer.

    Paper:  https://arxiv.org/abs/2311.08105

    :param: outer_optimizer: Callable to an optimizer to update the pseudo gradient, this optimizer is shared between peers. (DiLoCo used the Nesterov opt)
    :param: inner_optimizer: Callable to an optimizer to update the model parameter locally, this optimizer is not shared between peers (DiLoCo used the AdamW opt)
    :param: scheduler: callable to a learning rate scheduler to update the inner optimizer lr.
    :param: num_inner_steps: number of inner optimizer updates per outer optimizer update
    :param: batch_size: number of samples in a single batch

    the rest of parameters are the same as Hivemind's Optimizer, expect `optimizer` that is override by `outer_optimizer`.
    """

    state_averager: DiLoCoStateAverager
    inner_optimizer: TorchOptimizer
    tracker: DiloCoProgressTracker
    diloco_grad_averager: DiLoCoGradAverager

    def __init__(
        self,
        *,
        dht: DHT,
        run_id: str,
        batch_size: int,
        num_inner_steps: int,
        outer_optimizer: OptimizerFactory,
        inner_optimizer: OptimizerFactory,
        params: Optional[Union[Parameters, ParamGroups]] = None,
        scheduler: Optional[SchedulerFactory] = None,
        averager_opts: Optional[dict] = None,
        grad_compression: CompressionBase = NoCompression(),
        tracker_opts: Optional[dict] = None,
        all_reduce_strategy: AllReduceStrategy = AllReduceStrategy.WAIT_FOR_ALL,
        timeout_waiting_for_peers: float | None = None,
        matchmaking_time: Optional[float] = 15.0,
        lora: bool | None = False,
        selective_layer_patterns: Optional[List[str]] = None,
        gradient_magnitude_threshold: Optional[float] = None,
        gradient_magnitude_top_k_ratio: Optional[float] = None,
        gradient_magnitude_top_k_ratio_by_size: bool = False,
        gradient_magnitude_selection_mode: str = "layer",  # "layer" or "parameter"
        gradient_importance_metric: str = "magnitude",  # "magnitude" or "taylor"
        param_names: Optional[List[str]] = None,
        token_weighted_aggregation: bool = False,
        residual_norm_threshold: Optional[float] = None,
        enable_update_logs: bool = False,
        # Max Staleness (강제 업데이트) 기능
        enable_max_staleness: bool = False,
        max_staleness: int = 100,
        # Warm-up (서서히 줄이기) 기능
        enable_warmup: bool = False,
        warmup_epochs: int = 5,
        # Gradient Clipping 기능
        enable_gradient_clipping: bool = False,
        gradient_clip_norm: float = 1.0,
        **kwargs,
    ):
        self._check_kwargs(kwargs)
        
        # Selective layer update를 위한 파라미터 이름 수집
        self.selective_layer_patterns = selective_layer_patterns
        self.gradient_magnitude_threshold = gradient_magnitude_threshold
        self.gradient_magnitude_top_k_ratio = gradient_magnitude_top_k_ratio
        self.gradient_magnitude_top_k_ratio_by_size = gradient_magnitude_top_k_ratio_by_size
        self.gradient_magnitude_selection_mode = gradient_magnitude_selection_mode
        self.gradient_importance_metric = gradient_importance_metric
        self.token_weighted_aggregation = token_weighted_aggregation
        self.residual_norm_threshold = residual_norm_threshold
        self.enable_update_logs = enable_update_logs
        # Max Staleness 및 Warm-up 설정
        self.enable_max_staleness = enable_max_staleness
        self.max_staleness = max_staleness
        self.enable_warmup = enable_warmup
        self.warmup_epochs = warmup_epochs
        # Gradient Clipping 설정
        self.enable_gradient_clipping = enable_gradient_clipping
        self.gradient_clip_norm = gradient_clip_norm
        if selective_layer_patterns is not None or gradient_magnitude_threshold is not None or gradient_magnitude_top_k_ratio is not None:
            if param_names is None:
                # param_names가 제공되지 않으면 모델에서 직접 추출 시도
                # 하지만 여기서는 모델에 직접 접근할 수 없으므로, 나중에 grad_averager 생성 시 전달
                logger.warning("selective_layer_patterns is provided but param_names is None. Parameter names will be extracted from optimizer.")
            self.param_names = param_names
        else:
            self.param_names = None

        if timeout_waiting_for_peers is not None:
            if all_reduce_strategy == AllReduceStrategy.NO_WAIT:
                raise ValueError(
                    "You cannot use timeout_waiting_for_peers with NO_WAIT strategy, use WAIT_FOR_ALL instead"
                )

        if timeout_waiting_for_peers is not None and timeout_waiting_for_peers < matchmaking_time:
            raise ValueError("timeout_waiting_for_peers must be greater than matchmaking_time")

        if all_reduce_strategy == AllReduceStrategy.WAIT_FOR_ALL:
            if timeout_waiting_for_peers is None:
                timeout_waiting_for_peers = DEFAULT_TIMEOUT_WAITING_FOR_PEERS

        self.all_reduce_strategy = all_reduce_strategy
        self.timeout_waiting_for_peers = timeout_waiting_for_peers

        params = list(params)
        # cyshin
        if lora:
            params = [p for p in params if p.requires_grad]
        
        # if params is a generator (like model.parameters()) it would be consumed by the first optimizer
        # since we have two optimizers, we need to persist the params to a list
        self.num_inner_steps = num_inner_steps

        for opt_or_scheduler in [outer_optimizer, scheduler]:
            if not (callable(opt_or_scheduler) or opt_or_scheduler is None):
                raise TypeError("You need to pass inner and outer optimizer as well as scheduler as callable")

        if isinstance(inner_optimizer, TorchOptimizer):
            self.inner_optimizer = inner_optimizer
        elif isinstance(inner_optimizer, Callable):
            self.inner_optimizer = inner_optimizer(params=params)
            # cyshin
            # called here
        else:
            raise TypeError(
                f"Expected inner_optimizer to be TorchOptimizer or OptimizerFactory, got {type(inner_optimizer)}"
            )

        if tracker_opts is None:
            tracker_opts = {}

        tracker_opts.update(dict(batch_size=batch_size, num_inner_steps=num_inner_steps))

        if "max_refresh_period" not in tracker_opts:
            tracker_opts["max_refresh_period"] = 2

        self.scheduled_diloco_grads: Optional[StepControl] = None

        super().__init__(
            optimizer=outer_optimizer,
            dht=dht,
            run_id=run_id,
            target_batch_size=batch_size * num_inner_steps,
            batch_size_per_step=batch_size,
            params=params,
            scheduler=scheduler,
            use_local_updates=True,  # we are handling grad scaler ourself
            offload_optimizer=True,  # DiLoCo is always offloading optimizers bc of the pseudo gradient
            averager_opts=averager_opts,
            tracker_opts=tracker_opts,
            matchmaking_time=matchmaking_time,
            **kwargs,
        )
        self.diloco_grad_averager = self._make_gradient_averager(compression=grad_compression)
        
        # state_averager에 grad_averager 참조 설정 (skip된 파라미터 보호를 위해)
        self.state_averager.grad_averager = self.diloco_grad_averager

    def _check_kwargs(self, kwargs) -> None:
        """DiLoCo Optimizer only support a subset of Hivemind Optimizer kwargs.
        This function raise an error if some kwargs are not supported"""

        if "optimizer" in kwargs:
            raise KeyError("optimizer should not be passed to DiLoCoOptimizer, pass rather to outer_optimizer")

        if "use_local_updates" in kwargs:
            if kwargs["use_local_updates"] is False:
                raise ValueError(
                    "You cannot use DiLoCo without local updates, please use normal Optimizer if you don't want local updates"
                )
            else:
                kwargs.pop("use_local_updates")

        if "offload_optimizer" in kwargs:
            if kwargs["offload_optimizer"] is False:
                raise ValueError("offload_optimizer=False, is not supported in DiLoCo for now")
            else:
                kwargs.pop("offload_optimizer")

        for arg_name in (
            "delay_state_averaging",
            "delay_grad_averaging",
            "delay_optimizer_step",
        ):
            if arg_name in kwargs:
                if kwargs[arg_name] is True:
                    raise ValueError(f"{arg_name} is not supported in DiLoCo for now")

        if "target_batch_size" in kwargs:
            raise KeyError(
                "DiLoCo does not have a target_batch_size, use batch_size instead in combination with num_inner_steps"
            )

        if "batch_size_per_step" in kwargs:
            raise KeyError("DiLoCo does not have a batch_size_per_step, use batch_size instead")

    def _make_gradient_averager(self, **kwargs) -> DiLoCoGradAverager:
        assert hasattr(self, "state_averager"), "must initialize state averager first"
        print("call _make_gradient_averager")
        
        # 파라미터 이름 추출 (selective_layer_patterns 또는 gradient magnitude 기반 선택이 있는 경우)
        param_names = None
        if self.selective_layer_patterns is not None or self.gradient_magnitude_threshold is not None or self.gradient_magnitude_top_k_ratio is not None:
            if self.param_names is not None:
                param_names = self.param_names
            else:
                # offloaded optimizer에서 파라미터 이름 추출 시도
                # 주의: optimizer의 파라미터는 순서가 보장되지만 이름은 직접 저장되지 않음
                # 따라서 외부에서 param_names를 제공해야 함
                logger.warning("Cannot extract parameter names from optimizer. Please provide param_names when using selective layer update.")
        
        grad_averager = DiLoCoGradAverager(
            dht=self.dht,
            prefix=f"{self.run_id}_grad_averager",
            # cyshin
            target_group_size=4,
            bandwidth=100, 
            main_parameters=self.state_averager.main_parameters,
            offloaded_optimizer=self.state_averager.optimizer,
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            param_names=param_names,
            selective_layer_patterns=self.selective_layer_patterns,
            gradient_magnitude_threshold=self.gradient_magnitude_threshold,
            gradient_magnitude_top_k_ratio=self.gradient_magnitude_top_k_ratio,
            gradient_magnitude_top_k_ratio_by_size=self.gradient_magnitude_top_k_ratio_by_size,
            gradient_magnitude_selection_mode=self.gradient_magnitude_selection_mode,
            gradient_importance_metric=getattr(self, 'gradient_importance_metric', 'magnitude'),
            token_weighted_aggregation=self.token_weighted_aggregation,
            residual_norm_threshold=self.residual_norm_threshold,
            enable_update_logs=self.enable_update_logs,
            enable_max_staleness=self.enable_max_staleness,
            max_staleness=self.max_staleness,
            enable_warmup=self.enable_warmup,
            warmup_epochs=self.warmup_epochs,
            **kwargs,
        )
        
        # num_inner_steps를 grad_averager에 전달 (local step 동기화용)
        grad_averager.num_inner_steps = self.num_inner_steps
        
        # Token-weighted aggregation을 위한 설정 (활성화된 경우에만)
        if self.token_weighted_aggregation:
            RUN_ID = "OpenDiLoCo"
            grad_averager.token_count_key = f"{RUN_ID}:tokens"
            grad_averager.worker_id = f"{socket.gethostname()}-pid{os.getpid()}"
            grad_averager.cumulative_tokens = 0
            # expected_num_peers는 step() 호출 시점에 tracker에서 동적으로 설정됨
        
        return grad_averager

    def _make_state_averager(self, **kwargs) -> DiLoCoStateAverager:
        return DiLoCoStateAverager(
            dht=self.dht,
            prefix=f"{self.run_id}_state_averager",
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            offload_optimizer=self.offload_optimizer,
            custom_gradients=self.offload_optimizer,
            status_loglevel=self.status_loglevel,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            num_inner_steps=self.num_inner_steps,
            inner_optimizer=self.inner_optimizer,
            token_weighted_aggregation=False,  # State averager에서는 token-weighted aggregation 비활성화
            **kwargs,
        )

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        current_loss: Optional[float] = None,
        **kwargs,
    ):
        """
        Note: code is is copied from Hivemind's Optimizer.step, the main change is that the local step is used with the **iner optimizer**, only
        the global step that sync data via all reduce is using the **outer optimizer** states.

        Note: There is no gradient accumulation in our DiLoCo implementation since we use local updates.

        Note2: the gradient scaler is only apply to the inner optimizer step. The outer optimizer step is working on pseudo gradient
        that don't need to be scaled.

        Note3: You should not call scaler.step(optimizer) but rather optimizer.step(scaler=scaler) otherwise the scaler will not work as expected because of the outer step.

        Update training. Depending on the configuration, this will
        report progress to peers, run global or local optimizer step, average parameters or schedule background tasks.

        Grad scaler must be pass to use mixed precision with the inner optimizer. One can call unscale_ before tho.

        :param closure: A closure that reevaluates the model and returns the loss.
        :param batch_size: optional override for batch_size_per_step from init.
        :param scaler: a scaler from torch.cuda.amp.GradScaler, if provided, the scaler will be used to scale the inner optimizer step but not the outer optimizer step.
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        ### OG HIVEMIND CODE START ###
        if self.batch_size_per_step is None and batch_size is None and not self.auxiliary:
            raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
        if self.auxiliary and (closure is not None or batch_size is not None):
            raise ValueError("Auxiliary peers should not have batch size, run closures, or use grad_scaler")
        if scaler is not None and closure is not None:
            raise ValueError("You cannot use closure and scaler at the same time")

        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        # if delayed updates finished before step, apply these updates; otherwise do nothing
        # self.state_averager.step(apply_delayed_updates=True)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Loss 저장 (grad_averager에 전달용)
        # train_fsdp.py에서 optimizer.step(scaler=scaler, current_loss=...)로 전달됨
        if current_loss is not None:
            self._current_loss = float(current_loss)
        elif loss is not None:
            self._current_loss = float(loss.detach().item())
        # loss와 current_loss가 모두 None이면 이전 값 유지 (outer step에서는 loss가 없을 수 있음)

        if not self.auxiliary and self._should_load_state_from_peers():
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return loss  # local gradients were computed with out-of-sync parameters, must start over

        ### OG HIVEMIND CODE END ###

        # this code is similar to the hivemind.Optimizer.step when `use_local_updates` is True
        # at the difference that it call the inner optimizer step as well.

        if not self.auxiliary:
            new_samples_accumulated = self.tracker.local_progress.samples_accumulated + batch_size
            self.tracker.report_local_progress(self.local_epoch, samples_accumulated=new_samples_accumulated)

            self._maybe_schedule_state_averaging()
            print("call self._maybe_schedule_gradient_averaging()")
            self._maybe_schedule_gradient_averaging()

            if scaler is not None:
                print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f]"), "call scaler.step(self.inner_optimizer)")
                scaler.step(self.inner_optimizer)
                if found_inf_grad(self.inner_optimizer, scaler):
                    logger.log(self.status_loglevel, f"Found inf grad at step {self.tracker.real_step}")
            else:
                print("call self.inner_optimizer.step(closure=closure)")
                self.inner_optimizer.step(closure=closure)

            if self.state_averager.scheduler_inner_optimizer:
                print("call self.state_averager.scheduler_inner_optimizer.step()")
                self.state_averager.scheduler_inner_optimizer.step()

        if self.tracker.ready_to_update_epoch:
            print("call  self._update_global_epoch()")
            self._update_global_epoch()

        return loss

    def _compute_schema_hash(self) -> int:
        """this function is similar to hivemind.Optimizer._compute_schema_hash
        but disregard the gradient buffers of the offloaded optimizer
        """
        optimized_param_groups = self.state_averager.optimizer.param_groups
        optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
        param_shapes = tuple(tuple(param.shape) for param in optimized_parameters)
        grad_ids = None
        return hash((grad_ids, param_shapes))

    def _update_global_epoch(self) -> None:
        """Depending on the configuration: aggregate gradients and/or parameters, perform global optimizer step

        NOTE: this has been mostly copied from hivemind.Optimizer._update_global_epoch, except highlighted lines
        """
        assert self._schema_hash == self._compute_schema_hash(), "parameters changed during iteration"
        _epoch_start_time = time.perf_counter()

        if self.tracker.global_progress.num_peers > 1:
            if self.all_reduce_strategy == AllReduceStrategy.WAIT_FOR_ALL:
                if self.scheduled_diloco_grads is None:
                    init_time_waiting = time.perf_counter()

                    timeout_triggered = False

                    while time.perf_counter() - init_time_waiting < self.timeout_waiting_for_peers:
                        eta_next_epoch = self.tracker.global_progress.eta_next_epoch - get_dht_time()
                        if eta_next_epoch > self.matchmaking_time:
                            time_to_wait = max(0.1, self.tracker.global_progress.next_fetch_time - get_dht_time())
                            logger.log(
                                self.status_loglevel,
                                f"ETA next epoch {eta_next_epoch}, refresh in {time_to_wait}",
                            )
                            time.sleep(time_to_wait)
                        else:
                            logger.log(
                                self.status_loglevel,
                                f"Pre-scheduling gradient averaging round in {self.matchmaking_time:.2f} sec",
                            )
                            break
                    else:
                        timeout_triggered = True

                    if timeout_triggered:
                        logger.log(
                            self.status_loglevel,
                            "Timeout waiting for peers all-reduce was triggered. Going to skip slowest peers",
                        )
                        # todo(sami) in this case we still will have to wait for min_matchmaking_time, this could be optimized

        with self.tracker.pause_updates():
            assert not self.delay_optimizer_step, "delay_optimizer_step must be False in DiLoCo"

            if self.tracker.global_progress.num_peers > 1:
                # epoch 정보를 grad averager에 전달
                if hasattr(self.diloco_grad_averager, '_current_epoch'):
                    self.diloco_grad_averager._current_epoch = self.local_epoch
                self.diloco_grad_averager._current_step = self.tracker.real_step
                
                # Token-weighted aggregation 및 local step 동기화를 위한 expected_num_peers 설정
                self.diloco_grad_averager.expected_num_peers = max(
                    self.tracker.global_progress.num_peers,
                    2  # 최소 2개 노드는 있어야 함
                )
                
                self.diloco_grad_averager.step(
                    wait=True, timeout=self.averaging_timeout, control=self.scheduled_diloco_grads, 
                    epoch=self.local_epoch, current_loss=getattr(self, '_current_loss', None)
                )

                self.diloco_grad_averager.notify_used_averaged_gradients()
                
                # Token 수 리셋 (outer step 완료 후)
                self.diloco_grad_averager.reset_token_count()
                
                self.scheduled_diloco_grads = None
            else:
                self.diloco_grad_averager.compute_and_load_pseudo_grad_into_averager()

            next_epoch = max(self.local_epoch + 1, self.tracker.global_epoch)
            swarm_not_empty = self.tracker.global_progress.num_peers > 1
            should_perform_optimizer_step = True  # different from hivemind.Optimizer
            should_average_state = (
                swarm_not_empty
                and next_epoch % self.average_state_every == 0
                and not self.state_averager.averaging_in_progress
            )

            if should_average_state and self.scheduled_state is not None:
                if self.scheduled_state.triggered or self.scheduled_state.done():
                    logger.log(
                        self.status_loglevel,
                        f"Not using pre-scheduled group for state averaging because it"
                        f"was already used elsewhere: {self.scheduled_state}",
                    )
                    self.scheduled_state = None
                self.delay_before_state_averaging.update(task_size=1, interval=time.perf_counter() - _epoch_start_time)

            assert self.state_averager.custom_gradients, "custom gradient must be enable for syncing pseudo gradients"

            # 통신된 averaged gradient를 outer optimizer에 로드
            # custom_gradients=True이므로 수동으로 averaged gradient를 로드해야 합니다.
            if self.tracker.global_progress.num_peers > 1:
                self._load_averaged_gradients_into_outer_optimizer()

            # Token-weighted aggregation을 위한 expected_num_peers 설정 (state_averager)
            if self.state_averager.token_weighted_aggregation:
                # tracker에서 현재 peer 수를 가져와서 설정 (최소한의 예상 값)
                self.state_averager.expected_num_peers = max(
                    self.tracker.global_progress.num_peers,
                    2  # 최소 2개 노드는 있어야 함
                )

            logger.info(f"Try outer optimizer step at  {self.tracker.real_step} step")
            
            # Gradient Clipping: outer optimizer 업데이트 전에 gradient clipping 적용
            if self.enable_gradient_clipping and should_perform_optimizer_step:
                # outer optimizer의 파라미터에 대한 gradient clipping
                outer_params = [param for group in self.state_averager.optimizer.param_groups for param in group["params"]]
                # gradient가 있는 파라미터만 필터링
                params_with_grad = [p for p in outer_params if p.grad is not None]
                if len(params_with_grad) > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=self.gradient_clip_norm)
                    logger.log(
                        self.status_loglevel,
                        f"Gradient clipping applied: total_norm={total_norm:.6f}, max_norm={self.gradient_clip_norm}"
                    )
            
            time_0_state_averager_step = time.perf_counter()
            self.state_averager.step(
                increment_epoch=True,
                wait_for_trigger=None,
                optimizer_step=should_perform_optimizer_step,
                delay_optimizer_step=self.delay_optimizer_step and should_perform_optimizer_step,
                grad_scaler=None,
                averaging_round=should_average_state,
                delay_averaging=self.delay_state_averaging and not self.auxiliary,
                averaging_control=(self.scheduled_state if should_average_state else None),
                averaging_opts=(dict(timeout=self.averaging_timeout) if should_average_state else None),
                zero_grad=False,  # zero grad should be done outside of diloco
            )
            time_1_state_averager_step = time.perf_counter()
            logger.log(
                self.status_loglevel,
                f"Time taken for state_averager_step: {time_1_state_averager_step - time_0_state_averager_step} sec",
            )

            if not should_average_state and self.scheduled_state is not None and not self.scheduled_state.done():
                self.scheduled_state.cancel()
            self.scheduled_state = None

            self.tracker.update_epoch(new_epoch=self.state_averager.local_epoch)
            self._should_check_synchronization_on_update = True
            # the above line ensures that peers check for *strict* synchronization once per epoch

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            # Token 수 리셋 (state_averager, outer step 완료 후)
            if self.state_averager.token_weighted_aggregation:
                self.state_averager.reset_token_count()

            # cyshin
            time_0_update_main_param = time.perf_counter()
            self.update_main_param_after_outer_step()
            time_1_update_main_param = time.perf_counter()
            logger.log(
                self.status_loglevel,
                f"Time taken for update_main_param: {time_1_update_main_param - time_0_update_main_param} sec",
            )
            logger.log(self.status_loglevel, f"Transitioning to epoch {self.local_epoch}")

    def _make_progress_tracker(self, target_batch_size: int, **kwargs) -> DiloCoProgressTracker:
        return DiloCoProgressTracker(
            dht=self.dht,
            prefix=self.run_id,
            target_batch_size=target_batch_size,
            client_mode=self.client_mode,
            status_loglevel=self.status_loglevel,
            start=True,
            **kwargs,
        )

    @property
    def param_groups(self) -> ParamGroups:
        """Inner optimizer is the main optimizer"""
        return self.inner_optimizer.param_groups

    def state_dict(self) -> dict:
        """we save both inner and outer optimizer states, and the local epoch"""
        state_dict_outer = self.state_averager.optimizer.state_dict()
        state_dict_outer["state"]["local_epoch"] = self.local_epoch

        state_dict_inner = self.inner_optimizer.state_dict()

        return {
            "state_dict_outer": state_dict_outer,
            "state_dict_inner": state_dict_inner,
        }

    def load_state_dict(self, state_dict: dict):
        if "local_epoch" in state_dict["state_dict_outer"]["state"]:
            self.state_averager.local_epoch = state_dict["state_dict_outer"]["state"].pop("local_epoch")

        self.state_averager.optimizer.load_state_dict(state_dict["state_dict_outer"])
        self.inner_optimizer.load_state_dict(state_dict["state_dict_inner"])

    def update_num_inner_steps(self, new_num_inner_steps: int):
        """Update num_inner_steps across all components (optimizer, state_averager, and tracker)"""
        if new_num_inner_steps <= 0:
            raise ValueError(f"num_inner_steps must be positive, got {new_num_inner_steps}")
        
        logger.info(f"Updating num_inner_steps from {self.num_inner_steps} to {new_num_inner_steps}")
        
        # Update DiLoCoOptimizer's num_inner_steps
        self.num_inner_steps = new_num_inner_steps
        
        # Update DiLoCoStateAverager's num_inner_steps
        self.state_averager.num_inner_steps = new_num_inner_steps
        
        # Update DiloCoProgressTracker's num_inner_steps and target_batch_size
        self.tracker.num_inner_steps = new_num_inner_steps
        new_target_batch_size = self.tracker.batch_size * new_num_inner_steps
        self.tracker.target_batch_size = new_target_batch_size
        
        # Update GlobalTrainingProgress's target_batch_size
        self.tracker.global_progress.target_batch_size = new_target_batch_size
        
        logger.info(f"Updated target_batch_size to {new_target_batch_size} (batch_size={self.tracker.batch_size} * num_inner_steps={new_num_inner_steps})")

    @torch.no_grad()
    def _load_averaged_gradients_into_outer_optimizer(self):
        """Load averaged gradients from diloco_grad_averager into outer optimizer's gradient buffers
        
        custom_gradients=True이므로 통신된 averaged gradient를 수동으로 outer optimizer에 로드합니다.
        """
        outer_params = [param for group in self.state_averager.optimizer.param_groups for param in group["params"]]
        
        # param_update_mask 확인 (selective layer update가 활성화된 경우)
        param_update_mask = getattr(self.diloco_grad_averager, 'param_update_mask', None)
        
        # diloco_grad_averager에서 averaged gradient 가져오기
        with self.diloco_grad_averager.get_tensors() as averaged_grads:
            param_idx = 0
            grad_idx = 0
            
            for opt_param in outer_params:
                if param_update_mask is not None:
                    if param_update_mask[param_idx]:
                        # 선택된 파라미터: averaged gradient를 opt_param.grad에 설정
                        if grad_idx >= len(averaged_grads):
                            raise RuntimeError(
                                f"grad_idx ({grad_idx})가 averaged_grads의 길이 ({len(averaged_grads)})를 초과했습니다."
                            )
                        averaged_grad = averaged_grads[grad_idx]
                        if opt_param.grad is None:
                            opt_param.grad = averaged_grad.clone()
                        else:
                            opt_param.grad.copy_(averaged_grad, non_blocking=True)
                        grad_idx += 1
                    else:
                        # Skip된 파라미터: gradient를 0으로 설정
                        if opt_param.grad is None:
                            opt_param.grad = torch.zeros_like(opt_param)
                        else:
                            opt_param.grad.zero_()
                else:
                    # 모든 파라미터 업데이트: averaged gradient를 opt_param.grad에 설정
                    if grad_idx >= len(averaged_grads):
                        raise RuntimeError(
                            f"grad_idx ({grad_idx})가 averaged_grads의 길이 ({len(averaged_grads)})를 초과했습니다."
                        )
                    averaged_grad = averaged_grads[grad_idx]
                    if opt_param.grad is None:
                        opt_param.grad = averaged_grad.clone()
                    else:
                        opt_param.grad.copy_(averaged_grad, non_blocking=True)
                    grad_idx += 1
                param_idx += 1


    def update_main_param_after_outer_step(self):
        """Update the inner optimizer parameters with the main parameters after outer step
        
        흐름:
        1. Outer step: state_averager.step() 내부에서 outer optimizer.step() 호출
           → offloaded optimizer의 파라미터가 pseudo gradient로 업데이트됨
        2. _apply_optimizer_parameters_(): offloaded optimizer → main_parameters 복사
           → 모든 파라미터 업데이트 (skip된 파라미터도 포함, gradient가 0이므로 변화 없음)
        3. 이 함수: main_parameters → inner optimizer 파라미터 복사 (Main → Inner)
        
        Note: inner optimizer의 파라미터는 main_parameters와 같은 객체를 참조하므로,
        실제로는 복사가 필요 없지만, 명시적으로 동기화를 보장하기 위해 유지됩니다.
        PyTorch Optimizer는 파라미터 객체 자체를 참조하므로 별도 복사가 필요 없으나,
        안전을 위해 방향을 Main → Inner로 명확히 합니다.
        """
        opt_parameters = [param for group in self.inner_optimizer.param_groups for param in group["params"]]
        
        for main_param, opt_param in zip(self.state_averager.main_parameters, opt_parameters):
            # inner optimizer의 파라미터와 main_parameters는 같은 객체를 참조하므로,
            # _apply_optimizer_parameters_()에서 이미 선택적 업데이트가 처리되었습니다.
            # 여기서는 모든 파라미터를 복사해도 됩니다 (같은 객체이므로 결과는 동일).
            main_param.data.copy_(opt_param.data, non_blocking=True)

    def _maybe_schedule_gradient_averaging(self) -> None:
        """If next epoch is coming soon, schedule the next gradient averaging round at the estimated end of epoch"""

        if self.all_reduce_strategy == AllReduceStrategy.WAIT_FOR_ALL:
            eta_seconds = self.tracker.global_progress.eta_next_epoch - get_dht_time()
        else:
            eta_seconds = self.tracker.estimated_next_update_time

        if eta_seconds <= self.matchmaking_time:
            if ( 
                self.scheduled_diloco_grads is None
                or self.scheduled_diloco_grads.triggered
                or self.scheduled_diloco_grads.done()
            ):
                eta_seconds = max(eta_seconds, self.diloco_grad_averager.matchmaking_kwargs["min_matchmaking_time"])
                logger.log(self.status_loglevel, f"May be Pre-scheduling gradient averaging round in {eta_seconds:.2f} sec")
                # cyshin
                self.scheduled_diloco_grads = self.diloco_grad_averager.schedule_step(timeout=self.averaging_timeout)
