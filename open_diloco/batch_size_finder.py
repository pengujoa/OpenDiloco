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


def find_max_batch_size_for_model(
    config_or_model_path: Union[Any, str],
    gpu_mem_per_device_gb: Optional[float] = None,
    sequence_length: Optional[int] = None,
    precision: Optional[str] = None,
    activation_coefficient: float = 24.0,
    framework_overhead_fraction: float = 0.10,
    extra_inference_like_overhead_fraction: float = 0.15,
    micro_batch_search_upper_bound: int = 16384,
) -> int:
    """
    모델의 최대 배치 크기를 찾는 함수.
    
    Args:
        config_or_model_path: Config 객체 또는 모델 디렉토리 경로/이름
            - Config 객체인 경우: path_model, precision을 사용
            - str인 경우: 모델 디렉토리 경로 또는 이름 (예: "llama-150m-fresh")
        gpu_mem_per_device_gb: 디바이스당 GPU 메모리 (GB). None이면 torch.cuda API로 자동 확인
        sequence_length: 시퀀스 길이. None이면 모델 config에서 가져옴
        precision: 정밀도 ("fp16", "bf16", "fp32"). Config 객체인 경우 precision을 사용하며, 이 값은 무시됨
        activation_coefficient: 활성화 메모리 추정 계수, 기본값: 24.0
        framework_overhead_fraction: 프레임워크 오버헤드 비율, 기본값: 0.10
        extra_inference_like_overhead_fraction: 추가 추론 오버헤드 비율, 기본값: 0.15
        micro_batch_search_upper_bound: 마이크로 배치 크기 검색 상한, 기본값: 16384
        
    Returns:
        int: 디바이스당 최대 마이크로 배치 크기
    """
    # Config 객체인지 확인 (hasattr로 주요 속성 확인)
    is_config = hasattr(config_or_model_path, 'path_model')
    
    if is_config:
        # Config 객체에서 값 추출
        config = config_or_model_path
        model_path = config.path_model
        # Config 객체인 경우 항상 config.precision을 사용 (precision 파라미터 무시)
        precision = _convert_precision(config.precision)
        # sequence_length는 config에서 가져오지 않고 모델 config에서 가져옴
    else:
        # 기존 방식: 문자열로 모델 경로 전달
        model_path = config_or_model_path
        if precision is None:
            precision = "fp16"
    
    # GPU 메모리 확인: torch.cuda API 사용 또는 수동 지정 값 사용
    if gpu_mem_per_device_gb is None:
        if torch is not None and torch.cuda.is_available():
            # LOCAL_RANK 환경 변수에서 현재 디바이스 확인
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                # 디바이스 설정
                torch.cuda.set_device(local_rank)
                device_id = local_rank
            except (ValueError, RuntimeError):
                # LOCAL_RANK가 유효하지 않거나 디바이스 설정 실패 시 현재 디바이스 사용
                device_id = torch.cuda.current_device()
            
            # GPU의 총 메모리 가져오기 (GB)
            total_memory_bytes = torch.cuda.get_device_properties(device_id).total_memory
            gpu_mem_per_device_gb = bytes_to_gb(total_memory_bytes)
            print(f"GPU 메모리 자동 감지: {gpu_mem_per_device_gb:.2f} GB (device: cuda:{device_id})")
        else:
            # CUDA가 없거나 torch가 없으면 기본값 사용
            gpu_mem_per_device_gb = 80.0
            print(f"GPU 메모리 자동 감지 실패, 기본값 사용: {gpu_mem_per_device_gb:.2f} GB")
    
    # 모델 스펙 추출 (model_path를 직접 전달, Hugging Face 모델 ID 또는 로컬 경로 모두 처리)
    param_count, hidden_size, num_layers, seq_len_cfg = extract_model_specs(model_path)
    
    # 시퀀스 길이 결정
    if sequence_length is None:
        sequence_length = seq_len_cfg
    
    # ModelConfig 생성
    model = ModelConfig(
        parameter_count=param_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    
    # EstimationConfig 생성
    estimate = EstimationConfig(
        precision=precision,
        optimizer="adam",
        sequence_length=sequence_length,
        activation_coefficient=activation_coefficient,
        framework_overhead_fraction=framework_overhead_fraction,
        extra_inference_like_overhead_fraction=extra_inference_like_overhead_fraction,
    )
    
    # BatchEstimator 생성 및 시뮬레이션 실행
    estimator = BatchEstimator(model=model, estimate=estimate)
    result = estimator.simulate_max_batch_size(
        gpu_mem_per_device_gb=gpu_mem_per_device_gb,
        micro_batch_search_upper_bound=micro_batch_search_upper_bound,
    )
    
    # Debug 정보 출력
    print("\n" + "="*60)
    print("Batch Size Estimation Results (DEBUG)")
    print("="*60)
    print(f"Max micro batch per device: {result.max_micro_batch_per_device}")
    print(f"Per-device memory used: {bytes_to_gb(result.per_device_memory_used_bytes):.2f} GB")
    print(f"Per-device memory available: {bytes_to_gb(result.per_device_memory_available_bytes):.2f} GB")
    print(f"Headroom: {bytes_to_gb(result.headroom_bytes):.2f} GB")
    print("\nMemory Breakdown:")
    breakdown = result.breakdown_bytes
    print(f"  Model weights: {bytes_to_gb(breakdown.get('model_weights_bytes', 0)):.2f} GB")
    print(f"  Gradients: {bytes_to_gb(breakdown.get('gradients_bytes', 0)):.2f} GB")
    print(f"  Optimizer: {bytes_to_gb(breakdown.get('optimizer_bytes', 0)):.2f} GB")
    print(f"  Activations: {bytes_to_gb(breakdown.get('activations_bytes', 0)):.2f} GB")
    print(f"  Extra overhead: {bytes_to_gb(breakdown.get('extra_overhead_bytes', 0)):.2f} GB")
    print("="*60 + "\n")
    
    # 최대 배치 크기만 반환
    return result.max_micro_batch_per_device

