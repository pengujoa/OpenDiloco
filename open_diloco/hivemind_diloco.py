from enum import Enum
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import numpy as np

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
    check_interval: float = 1.0,
):
    """
    각 노드가 local step을 완료했는지 DHT를 통해 동기화합니다.
    
    :param dht: DHT 인스턴스
    :param epoch: 현재 epoch 번호
    :param num_inner_steps: 각 노드가 완료해야 하는 local step 수
    :param expected_num_peers: 예상되는 peer 수 (galaxy_size)
    :param log_fn: 로깅 함수 (None이면 logger.info 사용)
    :param timeout: 최대 대기 시간 (초)
    :param check_interval: 확인 간격 (초)
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
    log_fn(f"Published local step completion for epoch {epoch}: worker_id={worker_id}, local_steps={num_inner_steps}")
    
    # 모든 노드가 local step을 완료할 때까지 대기
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            log_fn(f"Timeout waiting for all nodes to complete local steps (timeout={timeout}s)")
            break
        
        local_step_res = dht.get(local_step_key, latest=True)
        local_step_root = unwrap(local_step_res) if local_step_res else None
        completed_count = 0
        
        if isinstance(local_step_root, dict):
            for k, v in local_step_root.items():
                p = unwrap(v)
                if isinstance(p, dict) and p.get("completed") is True and p.get("epoch") == epoch:
                    completed_count += 1
        
        log_fn(f"Local step completion status for epoch {epoch}: {completed_count}/{expected_num_peers} nodes completed")
        
        if completed_count >= expected_num_peers:
            log_fn(f"All {expected_num_peers} nodes completed local steps for epoch {epoch}!")
            break
        
        time.sleep(check_interval)
    
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
    log_fn(f"Updated local step completion expiration to 5 seconds for epoch {epoch}, worker_id={worker_id}")

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

        super().__init__(
            **kwargs
        )  # we specifically don't pass the scheduler here, default TrainingStateAverager would use it with the outer optimizer and we w

        self.scheduler_inner_optimizer = scheduler(self.inner_optimizer) if scheduler is not None else None
        assert isinstance(self.scheduler_inner_optimizer, (LRSchedulerBase, type(None)))
        
        # Token-weighted aggregation 초기화 (DHT와 worker_id가 설정된 후)
        if self.token_weighted_aggregation:
            self._init_token_weighted_aggregation(key_suffix="_state")

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
        if averaging_round and 'weight' not in kwargs:
            if averaging_control is not None:
                # 이미 생성된 control이 있으면 weight 설정
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
        
        # Token weight를 기반으로 momentum 조정 (optimizer step이 있을 때만)
        if optimizer_step and self.token_weighted_aggregation:
            # Token weight 가져오기 (computed_token_weight 또는 averaging_control.weight)
            token_weight = computed_token_weight
            if token_weight is None and averaging_control is not None and hasattr(averaging_control, 'weight'):
                token_weight = averaging_control.weight
            
            if token_weight is not None and token_weight > 0:
                self._adjust_momentum_from_token_weight(token_weight)
        
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
        token_weighted_aggregation: bool = False,
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

        # Selective layer update 지원
        self.param_names = param_names
        self.selective_layer_patterns = selective_layer_patterns
        if selective_layer_patterns is not None and param_names is not None:
            # 파라미터 마스크 생성: 업데이트할 레이어 패턴에 매칭되는 파라미터만 True
            self.param_update_mask = self._create_param_mask(param_names, selective_layer_patterns)
            updated_count = sum(self.param_update_mask)
            total_count = len(self.param_update_mask)
            logger.info(f"Selective layer update enabled: {updated_count}/{total_count} parameters will be updated")
            
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
        """파라미터 이름이 패턴 중 하나라도 매칭되면 True인 마스크 생성"""
        mask = []
        for param_name in param_names:
            matched = False
            for pattern in patterns:
                # 패턴이 파라미터 이름의 시작 부분과 매칭되는지 확인
                if pattern in param_name or param_name.startswith(pattern):
                    matched = True
                    break
            mask.append(matched)
        return mask

    def _grads_from_optimizer(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated optimizer"""
        param_groups = self.offloaded_optimizer.param_groups
        param_idx = 0
        for param_group in param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                # Selective layer update가 활성화된 경우, 마스크에 따라 gradient 반환
                if self.param_update_mask is not None:
                    if self.param_update_mask[param_idx]:
                        yield param.grad
                    # skipped params는 yield하지 않음 - 통신에서 완전히 제외
                else:
                    yield param.grad
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
        
        if control is None:
            # cyshin
            time_0_schedule_step = time.perf_counter()
            control = self.schedule_step(timeout=timeout, **kwargs)
            time_1_schedule_step = time.perf_counter()
            logger.log(
                logging.INFO,
                f"Time taken for schedule_step: {time_1_schedule_step - time_0_schedule_step} sec",
            )
        # cyshin
        time_0_compute_and_load_pseudo_grad = time.perf_counter()
        self.compute_and_load_pseudo_grad_into_averager()
        time_1_compute_and_load_pseudo_grad = time.perf_counter()
        logger.log(
            logging.INFO,
            f"Time taken for compute_and_load_pseudo_grad: {time_1_compute_and_load_pseudo_grad - time_0_compute_and_load_pseudo_grad} sec",
        )

        # Token-weighted aggregation: weight 계산 및 설정
        # weight가 kwargs에 명시적으로 지정되지 않은 경우에만 token 기반 weight 계산
        if 'weight' in kwargs:
            # 명시적으로 weight가 지정된 경우 사용 (token weight 덮어쓰기)
            control.weight = kwargs['weight']
            # kwargs에서 weight 제거하여 나중에 다시 사용되지 않도록 함
            kwargs.pop('weight', None)
        elif self.token_weighted_aggregation:
            # Mixin의 _compute_token_weight 메서드 사용
            self._compute_token_weight(control, log_prefix="GradAverager Token-weighted aggregation")
            # token weight 사용 후 kwargs에 weight가 있으면 덮어쓰기 방지
            if 'weight' in kwargs:
                kwargs.pop('weight', None)

        control.allow_allreduce()

        # control.result() 호출 전에 각 노드가 local step을 완료했는지 DHT를 통해 동기화
        if wait and self.num_inner_steps is not None and epoch is not None and self.expected_num_peers is not None:
            try:
                logger.log(
                    logging.INFO,
                    f"Synchronizing local step completion via DHT before control.result() call (epoch={epoch}, num_inner_steps={self.num_inner_steps}, expected_peers={self.expected_num_peers})",
                )
                wait_for_all_nodes_local_step_complete(
                    dht=self.dht,
                    epoch=epoch,
                    num_inner_steps=self.num_inner_steps,
                    expected_num_peers=self.expected_num_peers,
                    log_fn=lambda msg: logger.log(logging.INFO, msg),
                    timeout=timeout if timeout is not None else 300.0,
                )
                logger.log(
                    logging.INFO,
                    f"Local step synchronization completed for epoch {epoch}",
                )
            except Exception as e:
                logger.warning(f"Failed to synchronize local step completion via DHT: {e}. Proceeding with control.result() anyway.")

        # return control.result(timeout) if wait else control
        if wait:
            time_0_control_result = time.perf_counter()
            return_value = control.result(timeout)
            time_1_control_result = time.perf_counter()
            logger.log(
                logging.INFO,
                f"Time taken for control_result: {time_1_control_result - time_0_control_result} sec",
            )
            return return_value
        else: 
            return control
        

    @torch.no_grad()
    def compute_and_load_pseudo_grad_into_averager(self):
        """compute pseudo gradient by subtracting the offloaded optimizer parameters with the main parameters and load them in the averager"""
        opt_parameters = [param for group in self.offloaded_optimizer.param_groups for param in group["params"]]
        with self.get_tensors() as averaged_grads:
            param_idx = 0
            grad_idx = 0  # averaged_grads의 인덱스
            updated_param_names = []
            skipped_param_names = []
            
            for opt_param, main_param in zip(opt_parameters, self.main_parameters):
                param_name = self.param_names[param_idx] if self.param_names and param_idx < len(self.param_names) else f"param_{param_idx}"
                
                # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                if self.param_update_mask is not None:
                    if self.param_update_mask[param_idx]:
                        # 선택된 레이어만 pseudo gradient 계산
                        grad = opt_param.data - main_param.detach().to(opt_param.device)
                        averaged_grads[grad_idx].copy_(grad, non_blocking=True)
                        updated_param_names.append(param_name)
                        grad_idx += 1
                    else:
                        # 선택되지 않은 레이어는 통신에서 제외됨 (averaged_grads에 포함되지 않음)
                        skipped_param_names.append(param_name)
                else:
                    # 모든 레이어 업데이트
                    grad = opt_param.data - main_param.detach().to(opt_param.device)
                    averaged_grads[grad_idx].copy_(grad, non_blocking=True)
                    if not updated_param_names:  # 첫 번째 호출 시에만 로깅
                        updated_param_names.append(param_name)
                    grad_idx += 1
                param_idx += 1
            
            # 디버그: 실제로 통신되는 파라미터 로깅
            current_epoch = getattr(self, '_current_epoch', 0)
            if not hasattr(self, '_last_logged_epoch'):
                self._last_logged_epoch = -1
            
            # 매 epoch마다 한 번만 또는 매번 상세 로깅 (처음 몇 번만)
            should_log = current_epoch != self._last_logged_epoch or current_epoch < 3
            
            if should_log:
                logger.info(f"[DEBUG] Computing pseudo gradients (epoch {current_epoch}, step {getattr(self, '_current_step', '?')})")
                if self.param_update_mask is not None:
                    logger.info(f"[DEBUG]   - Parameters to be synced: {len(updated_param_names)}/{param_idx}")
                    logger.info(f"[DEBUG]   - Parameters skipped: {len(skipped_param_names)}/{param_idx}")
                    
                    if updated_param_names:
                        logger.info(f"[DEBUG]   - Synced parameter names:")
                        for i, name in enumerate(updated_param_names[:30]):  # 처음 30개
                            logger.info(f"     {i+1}. {name}")
                        if len(updated_param_names) > 30:
                            logger.info(f"     ... and {len(updated_param_names) - 30} more parameters")
                    
                    if skipped_param_names and len(skipped_param_names) <= 10:
                        logger.info(f"[DEBUG]   - Skipped parameter names:")
                        for i, name in enumerate(skipped_param_names[:10]):
                            logger.info(f"     {i+1}. {name}")
                else:
                    logger.info(f"[DEBUG]   - All {param_idx} parameters will be synced")
                
                self._last_logged_epoch = current_epoch

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False


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
        param_names: Optional[List[str]] = None,
        token_weighted_aggregation: bool = False,
        **kwargs,
    ):
        self._check_kwargs(kwargs)
        
        # Selective layer update를 위한 파라미터 이름 수집
        self.selective_layer_patterns = selective_layer_patterns
        self.token_weighted_aggregation = token_weighted_aggregation
        if selective_layer_patterns is not None:
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
        
        # 파라미터 이름 추출 (selective_layer_patterns가 있는 경우)
        param_names = None
        if self.selective_layer_patterns is not None:
            if self.param_names is not None:
                param_names = self.param_names
            else:
                # offloaded optimizer에서 파라미터 이름 추출 시도
                # 주의: optimizer의 파라미터는 순서가 보장되지만 이름은 직접 저장되지 않음
                # 따라서 외부에서 param_names를 제공해야 함
                logger.warning("Cannot extract parameter names from optimizer. Please provide param_names when using selective_layer_patterns.")
        
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
            token_weighted_aggregation=self.token_weighted_aggregation,
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
                logger.log(self.status_loglevel, f"Beginning optimizer step #{self.local_epoch}")
                time_0 = time.perf_counter()

                # epoch 정보를 grad averager에 전달
                if hasattr(self.diloco_grad_averager, '_current_epoch'):
                    self.diloco_grad_averager._current_epoch = self.local_epoch
                self.diloco_grad_averager._current_step = self.tracker.real_step
                
                # Token-weighted aggregation 및 local step 동기화를 위한 expected_num_peers 설정
                # tracker에서 현재 peer 수를 가져와서 설정 (최소한의 예상 값)
                self.diloco_grad_averager.expected_num_peers = max(
                    self.tracker.global_progress.num_peers,
                    2  # 최소 2개 노드는 있어야 함
                )
                
                self.diloco_grad_averager.step(
                    wait=True, timeout=self.averaging_timeout, control=self.scheduled_diloco_grads, epoch=self.local_epoch
                )
                time_1 = time.perf_counter()
                logger.log(
                    self.status_loglevel,
                    f"Time taken for gradient all reduce: {time_1 - time_0} sec",
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

            # Token-weighted aggregation을 위한 expected_num_peers 설정 (state_averager)
            if self.state_averager.token_weighted_aggregation:
                # tracker에서 현재 peer 수를 가져와서 설정 (최소한의 예상 값)
                self.state_averager.expected_num_peers = max(
                    self.tracker.global_progress.num_peers,
                    2  # 최소 2개 노드는 있어야 함
                )

            logger.info(f"Try outer optimizer step at  {self.tracker.real_step} step")
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

    def update_main_param_after_outer_step(self):
        """Update the main parameters with the inner optimizer step"""
        opt_parameters = [param for group in self.inner_optimizer.param_groups for param in group["params"]]
        for main_param, opt_param in zip(self.state_averager.main_parameters, opt_parameters):
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
