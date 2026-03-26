from typing import Optional, Union, Any
from pathlib import Path
import os
import sys

try:
    import torch
except ImportError:
    torch = None

# max_bs 모듈을 찾기 위해 프로젝트 루트를 sys.path에 추가
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# max_bs 디렉토리도 sys.path에 추가 (max_bs/max_bs/run.py에서 사용하는 방식과 동일)
_max_bs_dir = _project_root / "max_bs"
if str(_max_bs_dir) not in sys.path:
    sys.path.insert(0, str(_max_bs_dir))

try:
    from max_bs import BatchEstimator, EstimationConfig, ModelConfig
    from max_bs.model_io import resolve_model_dir, extract_model_specs
    from max_bs.utils import bytes_to_gb
except ImportError:
    # fallback: max_bs.max_bs에서 직접 import
    from max_bs.max_bs import BatchEstimator, EstimationConfig, ModelConfig
    from max_bs.max_bs.model_io import resolve_model_dir, extract_model_specs
    from max_bs.max_bs.utils import bytes_to_gb


def _convert_precision(precision: str) -> str:
    """train_fsdp.py의 precision 형식을 max_bs 형식으로 변환"""
    if precision == "fp16-mixed":
        return "fp16"
    elif precision == "bf16-mixed":
        return "bf16"
    elif precision == "32-true":
        return "fp32"
    else:
        # 이미 변환된 형식이거나 기본값인 경우 그대로 반환
        return precision


def _detect_gpu_memory_gb() -> float:
    """torch.cuda API로 현재 디바이스의 GPU 메모리(GB)를 감지."""
    if torch is not None and torch.cuda.is_available():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device_id = local_rank
        except (ValueError, RuntimeError):
            device_id = torch.cuda.current_device()

        total_memory_bytes = torch.cuda.get_device_properties(device_id).total_memory
        gb = bytes_to_gb(total_memory_bytes)
        print(f"GPU 메모리 자동 감지: {gb:.2f} GB (device: cuda:{device_id})")
        return gb

    print("GPU 메모리 자동 감지 실패, 기본값 사용: 80.00 GB")
    return 80.0


def _extract_from_config(config: Any) -> dict:
    """Config 객체에서 추정에 필요한 필드를 추출."""
    precision = _convert_precision(config.precision)
    optimizer = getattr(config, "inner_optimizer_type", "adamw")

    sharding = getattr(config, "sharding_strategy", "NO_SHARD")
    if getattr(config, "hv", None) is not None:
        sharding = "NO_SHARD"

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))

    return {
        "model_path": config.path_model,
        "precision": precision,
        "optimizer": optimizer,
        "sharding_strategy": sharding,
        "num_devices": world_size,
        "local_num_devices": local_world_size,
    }


def find_max_batch_size_for_model(
    config_or_model_path: Union[Any, str],
    gpu_mem_per_device_gb: Optional[float] = None,
    sequence_length: Optional[int] = None,
    precision: Optional[str] = None,
    optimizer: Optional[str] = None,
    sharding_strategy: str = "NO_SHARD",
    num_devices: int = 1,
    local_num_devices: int = 1,
    activation_coefficient: float = 24.0,
    framework_overhead_fraction: float = 0.10,
    extra_inference_like_overhead_fraction: float = 0.15,
    micro_batch_search_upper_bound: int = 16384,
) -> int:
    """
    모델의 최대 배치 크기를 찾는 함수.

    Args:
        config_or_model_path: Config 객체 또는 모델 디렉토리 경로/이름.
            Config 객체 전달 시 precision, optimizer, sharding_strategy 등을 자동 추출.
        gpu_mem_per_device_gb: 디바이스당 GPU 메모리 (GB). None이면 자동 감지.
        sequence_length: 시퀀스 길이. None이면 모델 config에서 가져옴.
        precision: "fp16", "bf16", "fp32". Config 전달 시 무시됨.
        optimizer: "adamw" 또는 "lion". Config 전달 시 무시됨.
        sharding_strategy: FSDP 샤딩 전략. Config 전달 시 무시됨.
        num_devices: world_size. Config 전달 시 환경변수에서 자동 추출.
        local_num_devices: 노드 내 GPU 수. Config 전달 시 자동 추출.
        activation_coefficient: 활성화 메모리 추정 계수.
        framework_overhead_fraction: 프레임워크 오버헤드 비율.
        extra_inference_like_overhead_fraction: 추가 추론 오버헤드 비율.
        micro_batch_search_upper_bound: 마이크로 배치 크기 검색 상한.

    Returns:
        int: 디바이스당 최대 마이크로 배치 크기.
    """
    is_config = hasattr(config_or_model_path, "path_model")

    if is_config:
        extracted = _extract_from_config(config_or_model_path)
        model_path = extracted["model_path"]
        precision = extracted["precision"]
        optimizer = extracted["optimizer"]
        sharding_strategy = extracted["sharding_strategy"]
        num_devices = extracted["num_devices"]
        local_num_devices = extracted["local_num_devices"]
    else:
        model_path = config_or_model_path
        if precision is None:
            precision = "fp16"
        if optimizer is None:
            optimizer = "adamw"

    if gpu_mem_per_device_gb is None:
        gpu_mem_per_device_gb = _detect_gpu_memory_gb()

    param_count, hidden_size, num_layers, seq_len_cfg = extract_model_specs(model_path)

    if sequence_length is None:
        sequence_length = seq_len_cfg

    model = ModelConfig(
        parameter_count=param_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    estimate = EstimationConfig(
        precision=precision,
        optimizer=optimizer,
        sequence_length=sequence_length,
        activation_coefficient=activation_coefficient,
        framework_overhead_fraction=framework_overhead_fraction,
        extra_inference_like_overhead_fraction=extra_inference_like_overhead_fraction,
        sharding_strategy=sharding_strategy,
        num_devices=num_devices,
        local_num_devices=local_num_devices,
    )

    estimator = BatchEstimator(model=model, estimate=estimate)
    result = estimator.simulate_max_batch_size(
        gpu_mem_per_device_gb=gpu_mem_per_device_gb,
        micro_batch_search_upper_bound=micro_batch_search_upper_bound,
    )

    print("\n" + "=" * 60)
    print("Batch Size Estimation Results")
    print("=" * 60)
    print(f"  Precision:         {precision}")
    print(f"  Optimizer:         {optimizer}")
    print(f"  Sharding:          {sharding_strategy}")
    print(f"  Devices (world):   {num_devices}")
    print(f"  Devices (local):   {local_num_devices}")
    print("-" * 60)
    print(f"  Max micro-batch/device: {result.max_micro_batch_per_device}")
    print(f"  Memory used:       {bytes_to_gb(result.per_device_memory_used_bytes):.2f} GB")
    print(f"  Memory available:  {bytes_to_gb(result.per_device_memory_available_bytes):.2f} GB")
    print(f"  Headroom:          {bytes_to_gb(result.headroom_bytes):.2f} GB")
    bd = result.breakdown_bytes
    print("  Memory Breakdown:")
    print(f"    Model weights (fp32 master): {bytes_to_gb(bd.get('model_weights_bytes', 0)):.2f} GB")
    print(f"    Gradients:                   {bytes_to_gb(bd.get('gradients_bytes', 0)):.2f} GB")
    print(f"    Optimizer states ({optimizer}): {bytes_to_gb(bd.get('optimizer_bytes', 0)):.4f} GB")
    print(f"    Activations:                 {bytes_to_gb(bd.get('activations_bytes', 0)):.2f} GB")
    print(f"    Extra overhead:              {bytes_to_gb(bd.get('extra_overhead_bytes', 0)):.2f} GB")
    print("=" * 60 + "\n")

    return result.max_micro_batch_per_device

