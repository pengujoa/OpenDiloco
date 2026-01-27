"""
Token-weighted aggregation mixin for DiLoCo averagers.

This mixin provides token-weighted aggregation functionality that can be used
by both DiLoCoGradAverager and DiLoCoStateAverager to avoid code duplication.
"""
import time
from typing import Dict, Optional

from hivemind.averaging.control import StepControl
from hivemind.dht.dht import DHT
from hivemind.utils import get_dht_time
from hivemind.utils.timed_storage import ValueWithExpiration
from hivemind.optim.optimizer import logger

import socket


class TokenWeightedAggregationMixin:
    """
    Mixin class that provides token-weighted aggregation functionality.
    
    Classes using this mixin should:
    1. Set self.token_weighted_aggregation = True/False in __init__
    2. Set self.dht, self.prefix, and self.worker_id after super().__init__()
    3. Call self._init_token_weighted_aggregation() after DHT is available
    """
    
    def _init_token_weighted_aggregation(self, key_suffix: str = ""):
        """
        Initialize token-weighted aggregation after DHT is available.
        
        :param key_suffix: Optional suffix to add to the DHT key (e.g., "_state" or "_grad")
        """
        if self.token_weighted_aggregation and self.dht is not None:
            self.worker_id = str(self.dht.peer_id)
            self.token_count_key = f"{self.prefix}{key_suffix}_token_counts"
            logger.info(f"Token-weighted aggregation initialized with key: {self.token_count_key}")
    
    def _read_token_counts_from_dht(
        self, 
        wait_for_peers: bool = True, 
        max_wait_time: float = 300.0, 
        check_interval: float = 0.1
    ) -> tuple[Dict[str, float], float]:
        """
        DHT에서 모든 노드의 token 수를 읽어옴 (token_weighted_aggregation이 활성화된 경우에만)
        
        :param wait_for_peers: True면 expected_num_peers만큼의 노드가 token 수를 업데이트할 때까지 대기
        :param max_wait_time: 최대 대기 시간 (초)
        :param check_interval: 대기 중 확인 간격 (초)
        :returns: (token_counts, wait_time) 튜플 - token_counts는 노드별 token 수 딕셔너리, wait_time은 항상 0.0 (시간 측정은 _do 함수에서 수행)
        """
        if not self.token_weighted_aggregation or self.token_count_key is None or self.dht is None:
            return {}, 0.0
        
        token_counts = {}
        start_time = time.time()  # 타임아웃 체크용으로만 사용
        
        while True:
            # DHT에서 모든 token count 읽기
            all_counts = self.dht.get(self.token_count_key, latest=True)
            if all_counts is not None:
                # all_counts는 ValueWithExpiration 객체이므로 .value로 접근
                if isinstance(all_counts, ValueWithExpiration):
                    counts_dict = all_counts.value
                else:
                    counts_dict = all_counts
                
                if isinstance(counts_dict, dict):
                    for subkey, value in counts_dict.items():
                        # 각 value도 ValueWithExpiration일 수 있음
                        if isinstance(value, ValueWithExpiration):
                            value_dict = value.value
                        else:
                            value_dict = value
                        
                        if isinstance(value_dict, dict) and "tokens" in value_dict:
                            token_counts[subkey] = value_dict["tokens"]
            
            if not wait_for_peers:
                return token_counts, 0.0
            
            # expected_num_peers가 설정되어 있으면 그만큼 기다림
            if self.expected_num_peers is not None:
                num_peers_with_tokens = len(token_counts)
                if num_peers_with_tokens >= self.expected_num_peers:
                    return token_counts, 0.0
            
            # 타임아웃 확인
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_wait_time:
                num_peers_with_tokens = len(token_counts)
                logger.warning(
                    f"Token-weighted aggregation: Timeout waiting for token counts. "
                    f"Got {num_peers_with_tokens}/{self.expected_num_peers}"
                )
                return token_counts, 0.0
            
            # 일정 간격마다 재확인
            time.sleep(check_interval)
    
    def _publish_token_count_to_dht(self, ttl: float = 300.0):
        """DHT에 현재 노드의 token 수를 publish (token_weighted_aggregation이 활성화된 경우에만)"""
        if not self.token_weighted_aggregation or self.token_count_key is None or self.worker_id is None or self.dht is None:
            return
        
        now = get_dht_time()
        payload = {
            "tokens": float(self.cumulative_tokens),
            "ts": now,
            "host": socket.gethostname()
        }
        exp = now + ttl
        self.dht.store(key=self.token_count_key, subkey=self.worker_id, value=payload, expiration_time=exp)
    
    def accumulate_tokens(self, token_count: int):
        """Token 수를 누적 (token_weighted_aggregation이 활성화된 경우에만)"""
        if self.token_weighted_aggregation:
            self.cumulative_tokens += token_count
    
    def reset_token_count(self):
        """Token 수를 리셋 및 DHT에서 이전 값 삭제 (outer step 완료 후 호출, token_weighted_aggregation이 활성화된 경우에만)"""
        if self.token_weighted_aggregation:
            # DHT에서 이전 token 수 값을 명시적으로 삭제 (다음 iteration에서 혼선 방지)
            self._delete_token_count_from_dht()
            self.cumulative_tokens = 0
    
    def _delete_token_count_from_dht(self):
        """DHT에서 현재 노드의 token 수 값을 삭제 (token_weighted_aggregation이 활성화된 경우에만)"""
        if not self.token_weighted_aggregation or self.token_count_key is None or self.worker_id is None or self.dht is None:
            return
        
        # DHT에서 해당 키 삭제 (expiration_time을 과거로 설정하여 삭제 효과)
        now = get_dht_time()
        self.dht.store(key=self.token_count_key, subkey=self.worker_id, value=None, expiration_time=now - 1)
    
    def _compute_token_weight(
        self, 
        averaging_control: Optional[StepControl],
        log_prefix: str = "Token-weighted aggregation"
    ) -> tuple[Optional[float], float]:
        """
        Token 수를 기반으로 weight를 계산하고 averaging_control에 설정.
        
        :param averaging_control: StepControl 객체 (weight를 설정할 대상)
        :param log_prefix: 로그 메시지에 사용할 prefix
        :returns: (weight, wait_time) 튜플 - weight는 계산된 weight 값 (설정 실패 시 None), wait_time은 항상 0.0 (시간 측정은 _do 함수에서 수행)
        """
        if not self.token_weighted_aggregation or averaging_control is None or self.cumulative_tokens <= 0:
            return None, 0.0
        
        # 자신의 token 수를 DHT에 publish (먼저 publish해서 다른 노드들이 읽을 수 있도록)
        self._publish_token_count_to_dht()
        
        # DHT에서 모든 노드의 token 수 조회 (모든 노드가 업데이트할 때까지 대기)
        # 시간 측정은 _do 함수에서 수행되므로 여기서는 측정하지 않음
        token_counts, _ = self._read_token_counts_from_dht(
            wait_for_peers=True,
            max_wait_time=300.0,  # 최대 5분 대기
            check_interval=0.1  # 0.1초마다 확인
        )
        
        # 모든 노드의 token 수 합산
        total_tokens = sum(token_counts.values())
        
        # Weight 계산: 자신의 token 수 / 전체 token 수
        if total_tokens > 0:
            weight = self.cumulative_tokens / total_tokens
            averaging_control.weight = weight
            logger.info(
                f"{log_prefix}: local_tokens={self.cumulative_tokens}, "
                f"total_tokens={total_tokens}, weight={weight:.6f}, num_peers={len(token_counts)}"
            )
            # Weight 계산 후 DHT에서 token count 삭제 (다음 iteration에서 혼선 방지)
            self._delete_token_count_from_dht()
            return weight, 0.0
        else:
            return None, 0.0

