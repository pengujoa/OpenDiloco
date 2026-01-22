"""
데이터셋을 각 노드별 batch size에 비례하여 분할하는 모듈
"""
import hashlib
from typing import Dict, List
from hivemind.dht.dht import DHT
from hivemind.utils import get_dht_time


def unwrap(v):
    """DHT value를 unwrap합니다."""
    return getattr(v, "value", v)


def publish_node_info(
    dht: DHT,
    key: str,
    worker_id: str,
    world_rank: int,
    gpu_count: int,
    per_device_batch_size: int,
    ttl: float = 300.0,
):
    """
    DHT에 현재 노드의 GPU 개수와 batch size 정보를 publish합니다.
    
    Args:
        dht: DHT 인스턴스
        key: DHT 키
        worker_id: 현재 노드의 worker ID
        world_rank: 현재 노드의 world rank
        gpu_count: 현재 노드의 GPU 개수
        per_device_batch_size: 디바이스당 batch size
        ttl: Time to live (초)
    """
    now = get_dht_time()
    payload = {
        "world_rank": int(world_rank),
        "gpu_count": int(gpu_count),
        "per_device_batch_size": int(per_device_batch_size),
        "node_batch_size": int(gpu_count * per_device_batch_size),
        "ts": now,
        "host": worker_id,
    }
    exp = now + ttl
    dht.store(key=key, subkey=worker_id, value=payload, expiration_time=exp)


def read_node_info(dht: DHT, key: str) -> Dict[str, Dict]:
    """
    DHT에서 모든 노드의 GPU 개수와 batch size 정보를 읽어옵니다.
    
    Args:
        dht: DHT 인스턴스
        key: DHT 키
    
    Returns:
        {worker_id: {"gpu_count": int, "per_device_batch_size": int, "node_batch_size": int}} 형태의 딕셔너리
    """
    res = dht.get(key, latest=True)
    root = unwrap(res) if res else None
    node_info: Dict[str, Dict] = {}
    
    if isinstance(root, dict):
        for k, v in root.items():
            p = unwrap(v)
            if isinstance(p, dict):
                if "gpu_count" in p and "per_device_batch_size" in p:
                    node_info[k] = {
                        "world_rank": int(p.get("world_rank", -1)),
                        "gpu_count": int(p["gpu_count"]),
                        "per_device_batch_size": int(p["per_device_batch_size"]),
                        "node_batch_size": int(p.get("node_batch_size", p["gpu_count"] * p["per_device_batch_size"])),
                    }
    
    return node_info


def get_node_batch_sizes_from_dht(
    dht: DHT,
    run_id: str,
    galaxy_size: int,
    timeout: float = 60.0,
) -> tuple[List[int], List[int], List[int]]:
    """
    DHT에서 모든 노드의 batch size, GPU worker 수, per_device_batch_size 정보를 읽어와서 리스트로 반환합니다.
    
    Args:
        dht: DHT 인스턴스
        run_id: 실행 ID
        galaxy_size: 전체 노드 개수
        timeout: 타임아웃 (초)
    
    Returns:
        (각 노드의 batch size 리스트, 각 노드의 GPU worker 수 리스트, 각 노드의 per_device_batch_size 리스트) (world_rank 순서대로 정렬)
    """
    import time
    
    key = f"{run_id}:node_info"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        node_info = read_node_info(dht, key)
        
        if len(node_info) >= galaxy_size:
            # world_rank 순서대로 정렬
            sorted_nodes = sorted(node_info.items(), key=lambda x: x[1].get("world_rank", -1))
            
            # world_rank가 유효한지 확인
            world_ranks = [info.get("world_rank", -1) for _, info in sorted_nodes]
            if all(wr >= 0 for wr in world_ranks) and len(set(world_ranks)) == galaxy_size:
                # world_rank 순서대로 batch size, GPU worker 수, per_device_batch_size 리스트 생성
                batch_sizes = [info["node_batch_size"] for _, info in sorted_nodes]
                gpu_counts = [info["gpu_count"] for _, info in sorted_nodes]
                per_device_batch_sizes = [info["per_device_batch_size"] for _, info in sorted_nodes]
                
                if len(batch_sizes) == galaxy_size and len(gpu_counts) == galaxy_size and len(per_device_batch_sizes) == galaxy_size:
                    return batch_sizes, gpu_counts, per_device_batch_sizes
        
        time.sleep(2.0)
    
    raise TimeoutError(f"Could not collect node info from all {galaxy_size} nodes within {timeout} seconds")


def split_dataset_by_worker_batch_size(
    dataset,
    node_batch_sizes: List[int],
    node_gpu_counts: List[int],
    node_per_device_batch_sizes: List[int],
    world_rank: int,
    local_rank: int,
):
    """
    각 GPU worker별 batch size에 비례하여 데이터셋을 분할합니다.
    각 worker는 독립적인 dataloader를 가지므로, worker 단위로 분할합니다.
    노드마다 per_device_batch_size가 다를 수 있으므로, 각 worker의 batch size에 비례하여 분할합니다.
    
    Args:
        dataset: 분할할 데이터셋
        node_batch_sizes: 각 노드의 batch size 리스트
        node_gpu_counts: 각 노드의 GPU worker 수 리스트
        node_per_device_batch_sizes: 각 노드의 per_device_batch_size 리스트
        world_rank: 현재 노드의 world rank (0부터 시작)
        local_rank: 현재 노드 내의 local rank (0부터 시작)
    
    Returns:
        분할된 데이터셋
    """
    # 현재 worker의 global rank 계산
    # global_rank = 이전 노드들의 GPU worker 수 합 + 현재 노드의 local_rank
    global_rank = sum(node_gpu_counts[:world_rank]) + local_rank
    
    # 현재 worker의 batch size 계산 (현재 노드의 per_device_batch_size)
    current_worker_batch_size = node_per_device_batch_sizes[world_rank]
    
    # 각 worker의 batch size를 개별적으로 계산하여 전체 batch size 계산
    # 각 노드의 worker 수를 고려하여 각 worker의 batch size를 리스트로 생성
    worker_batch_sizes = []
    for node_idx in range(len(node_gpu_counts)):
        per_device_batch_size = node_per_device_batch_sizes[node_idx]
        for _ in range(node_gpu_counts[node_idx]):
            worker_batch_sizes.append(per_device_batch_size)
    
    # 전체 batch size = 모든 worker의 batch size 합
    total_batch_size = sum(worker_batch_sizes)
    
    # 각 worker의 batch size에 비례하여 데이터 비율 계산
    # 각 worker가 받을 데이터의 비율 = worker의 batch size / 전체 batch size
    worker_ratio = current_worker_batch_size / total_batch_size
    
    # 이전 worker들의 batch size 합 계산 (현재 worker 이전의 모든 worker들)
    # worker_batch_sizes 리스트를 사용하여 정확하게 계산
    previous_workers_batch_size = sum(worker_batch_sizes[:global_rank])
    
    # 전체 GPU worker 수 계산 (마지막 worker 체크용)
    total_gpu_workers = sum(node_gpu_counts)
    
    # 현재 worker의 시작 비율과 끝 비율 계산
    worker_start_ratio = previous_workers_batch_size / total_batch_size
    worker_end_ratio = (previous_workers_batch_size + current_worker_batch_size) / total_batch_size
    
    def hash_assign(example):
        """
        샘플을 해시하여 현재 worker에 할당합니다.
        """
        # 샘플의 내용을 해시하여 일관된 할당
        if 'input_ids' in example:
            sample_str = str(example['input_ids'])
        else:
            sample_str = str(example)
        
        sample_hash = int(hashlib.md5(sample_str.encode()).hexdigest(), 16)
        # 0~1 범위로 정규화
        hash_normalized = (sample_hash % 1000000) / 1000000.0
        
        # 현재 worker 범위 내에 있는지 확인
        if worker_start_ratio <= hash_normalized < worker_end_ratio:
            return True
        
        # 부동소수점 오차로 인한 경계 케이스 처리 (마지막 worker)
        if global_rank == total_gpu_workers - 1 and hash_normalized >= worker_end_ratio - 1e-10:
            return True
        
        return False
    
    return dataset.filter(hash_assign)

