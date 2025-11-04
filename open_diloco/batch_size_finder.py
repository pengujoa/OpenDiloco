"""
OoM을 발생시키지 않는 최대 batch_size를 탐색하는 모듈
Accelerate의 메모리 추정기를 사용하여 batch size를 자동으로 찾습니다.
"""

from dataclasses import dataclass
from typing import Optional

try:
    # Accelerate의 estimate 모듈에서 필요한 함수들을 import
    from accelerate.commands.estimate import gather_data
except ImportError:
    try:
        # 대안: 다른 경로에서 시도
        from accelerate.estimator import gather_data
    except ImportError:
        gather_data = None


@dataclass
class MemoryEstimate:
    """메모리 추정 결과를 저장하는 클래스"""
    dtype: str
    largest_layer_mb: float
    total_size_mb: float
    training_adam_mb: float


def get_available_gpu_memory_gb(device_id: int = 0) -> float:
    """
    GPU의 사용 가능한 메모리를 GB 단위로 반환합니다.
    
    Args:
        device_id: GPU 디바이스 ID (기본값: 0)
    
    Returns:
        사용 가능한 GPU 메모리 (GB)
    
    Raises:
        RuntimeError: GPU를 사용할 수 없거나 메모리 정보를 가져올 수 없는 경우
    """
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    
    # 사용 가능한 메모리 가져오기 (free, total)
    free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(device_id)
    # 총 메모리를 GB로 변환하여 가용 메모리로 사용
    available_memory_gb = total_memory_bytes / (1024 ** 3)
    
    if available_memory_gb <= 0:
        raise RuntimeError(f"GPU {device_id}의 사용 가능한 메모리가 0입니다.")
    
    return available_memory_gb


def bytes_to_mb(bytes_value: int | float) -> float:
    """바이트를 MB로 변환"""
    return bytes_value / (1024 ** 2)


def estimate_model_memory(model_name: str, library_name: Optional[str] = None) -> dict[str, MemoryEstimate]:
    """
    Accelerate의 gather_data를 사용하여 모델의 메모리 사용량을 추정합니다.
    
    Args:
        model_name: 모델 이름 (예: "bert-base-cased", "PrimeIntellect/llama-150m-fresh")
        library_name: 모델 라이브러리 (예: "transformers", "timm")
    
    Returns:
        dtype별 메모리 추정 결과 딕셔너리
    
    Raises:
        ImportError: Accelerate가 설치되어 있지 않거나 gather_data를 import할 수 없는 경우
        RuntimeError: 메모리 추정에 실패한 경우
    """
    if gather_data is None:
        raise ImportError("Accelerate의 gather_data를 import할 수 없습니다. Accelerate가 설치되어 있는지 확인하세요.")
    
    try:
        # gather_data 호출하여 메모리 정보 수집
        memory_data = gather_data(
            model_name=model_name,
            library_name=library_name,
            trust_remote_code=False,
        )
        
        if not memory_data:
            raise ValueError("메모리 추정 데이터가 비어있습니다.")
        
        # gather_data가 반환하는 데이터 구조를 파싱
        estimates = {}
        
        # gather_data의 반환 구조에 따라 처리
        # 일반적으로 리스트나 딕셔너리 형태로 반환됩니다
        if isinstance(memory_data, dict):
            # 딕셔너리인 경우: dtype을 키로 사용
            for dtype, data in memory_data.items():
                if isinstance(data, dict):
                    estimates[dtype] = MemoryEstimate(
                        dtype=dtype,
                        largest_layer_mb=bytes_to_mb(data.get("largest_layer", data.get("largest_layer_bytes", 0))),
                        total_size_mb=bytes_to_mb(data.get("total_size", data.get("total_size_bytes", 0))),
                        training_adam_mb=bytes_to_mb(data.get("training_adam", data.get("training_adam_bytes", 0))),
                    )
                elif hasattr(data, 'largest_layer') or hasattr(data, 'largest_layer_bytes'):
                    # dataclass나 namedtuple인 경우
                    largest = getattr(data, 'largest_layer', None) or getattr(data, 'largest_layer_bytes', 0)
                    total = getattr(data, 'total_size', None) or getattr(data, 'total_size_bytes', 0)
                    training = getattr(data, 'training_adam', None) or getattr(data, 'training_adam_bytes', 0)
                    estimates[dtype] = MemoryEstimate(
                        dtype=dtype,
                        largest_layer_mb=bytes_to_mb(largest),
                        total_size_mb=bytes_to_mb(total),
                        training_adam_mb=bytes_to_mb(training),
                    )
        elif isinstance(memory_data, list):
            # 리스트인 경우: 각 요소를 처리
            for item in memory_data:
                dtype = item.get("dtype") if isinstance(item, dict) else getattr(item, "dtype", "float32")
                if isinstance(item, dict):
                    estimates[dtype] = MemoryEstimate(
                        dtype=dtype,
                        largest_layer_mb=bytes_to_mb(item.get("largest_layer", item.get("largest_layer_bytes", 0))),
                        total_size_mb=bytes_to_mb(item.get("total_size", item.get("total_size_bytes", 0))),
                        training_adam_mb=bytes_to_mb(item.get("training_adam", item.get("training_adam_bytes", 0))),
                    )
                else:
                    largest = getattr(item, 'largest_layer', None) or getattr(item, 'largest_layer_bytes', 0)
                    total = getattr(item, 'total_size', None) or getattr(item, 'total_size_bytes', 0)
                    training = getattr(item, 'training_adam', None) or getattr(item, 'training_adam_bytes', 0)
                    estimates[dtype] = MemoryEstimate(
                        dtype=dtype,
                        largest_layer_mb=bytes_to_mb(largest),
                        total_size_mb=bytes_to_mb(total),
                        training_adam_mb=bytes_to_mb(training),
                    )
        
        if not estimates:
            raise ValueError("메모리 추정 결과를 파싱할 수 없습니다. gather_data의 반환 구조를 확인하세요.")
        
        return estimates
        
    except Exception as e:
        raise RuntimeError(f"메모리 추정 실패: {e}")


def calculate_max_batch_size(
    model_memory_mb: float,
    available_gpu_memory_gb: float,
    seq_length: int,
    num_gpus: int = 1,
    precision_bits: int = 16,
    safety_margin: float = 0.05
) -> int:
    """
    주어진 GPU 메모리에서 실행 가능한 최대 batch size를 계산합니다.
    
    Args:
        model_memory_mb: 모델 메모리 사용량 (MB)
        available_gpu_memory_gb: 사용 가능한 GPU 메모리 (GB)
        seq_length: 시퀀스 길이
        num_gpus: GPU 개수
        precision_bits: 정밀도 비트 (16 또는 32)
        safety_margin: 안전 여유율 (기본 5%)
    
    Returns:
        최대 per-device batch size
    """
    # GPU 메모리를 MB로 변환
    available_memory_mb = available_gpu_memory_gb * 1024
    
    # 안전 여유를 고려한 사용 가능 메모리
    usable_memory_mb = available_memory_mb * (1 - safety_margin)
    
    # 모델과 optimizer가 차지하는 메모리
    optimizer_memory_mb = model_memory_mb  # Adam의 경우 모델 크기의 약 2배지만 보수적으로 1배로 계산
    
    # batch에 할당 가능한 메모리
    batch_memory_mb = usable_memory_mb - model_memory_mb - optimizer_memory_mb
    
    # 각 샘플의 메모리 계산
    bytes_per_param = precision_bits / 8
    estimated_memory_per_sample_mb = (model_memory_mb * bytes_per_param * seq_length) / (1000 * 1000)
    
    # 한 샘플당 약 2-4배의 활성화 메모리 고려
    activation_memory_multiplier = 3.0
    total_memory_per_sample_mb = estimated_memory_per_sample_mb * activation_memory_multiplier
    
    # 최대 batch size 계산
    if total_memory_per_sample_mb <= 0:
        return 1
    
    max_batch_size = int(batch_memory_mb / total_memory_per_sample_mb)
    max_batch_size = max(1, max_batch_size)  # 최소값은 1
    
    return max_batch_size


def find_max_batch_size_for_model(config) -> int | None:
    """
    OoM 없이 실행 가능한 최대 batch size를 탐색하고 반환합니다.
    
    Args:
        config: Config 설정 객체
    
    Returns:
        추정된 최대 per-device batch size. 실패 시 None 반환
    """
    print("=" * 80)
    print("최대 Batch Size 탐색 도구")
    print("=" * 80)
    print(f"모델: {config.path_model}")
    
    # GPU 메모리 자동 감지
    if config.available_gpu_memory_gb is None:
        print("\nGPU 메모리 자동 감지 중...")
        try:
            available_gpu_memory_gb = get_available_gpu_memory_gb(device_id=0)
            print(f"감지된 사용 가능한 GPU 메모리: {available_gpu_memory_gb:.2f} GB")
        except RuntimeError as e:
            print(f"오류: GPU 메모리 자동 감지 실패: {e}")
            print("--available-gpu-memory-gb 옵션을 수동으로 지정하세요.")
            print("예: --available-gpu-memory-gb 24.0")
            return None
    else:
        available_gpu_memory_gb = config.available_gpu_memory_gb
    
    print(f"사용 가능한 GPU 메모리: {available_gpu_memory_gb:.2f} GB")
    print(f"시퀀스 길이: {config.seq_length}")
    print(f"정밀도: {config.precision}")
    print("=" * 80)
    
    # 1. 모델 메모리 추정
    print("\n1단계: 모델 메모리 추정 중...")
    try:
        estimates = estimate_model_memory(config.path_model)
    except (ImportError, RuntimeError) as e:
        print(f"메모리 추정에 실패했습니다: {e}")
        return None
    
    if not estimates:
        print("메모리 추정 결과가 비어있습니다.")
        return None
    
    # 2. precision에 맞는 dtype 선택
    dtype_map = {
        "32-true": "float32",
        "fp16-mixed": "float16",
        "bf16-mixed": "float16",  # bf16도 float16과 크기 동일
    }
    selected_dtype = dtype_map.get(config.precision, "float16")
    
    if selected_dtype not in estimates:
        selected_dtype = "float16"  # 기본값
    
    print(f"\n선택된 dtype: {selected_dtype}")
    model_memory = estimates[selected_dtype]
    
    # 3. bits 계산
    precision_bits = 32 if config.precision == "32-true" else 16
    
    # 4. 최대 batch size 계산
    print("\n2단계: 최대 batch size 계산 중...")
    max_batch = calculate_max_batch_size(
        model_memory_mb=model_memory.training_adam_mb,
        available_gpu_memory_gb=available_gpu_memory_gb,
        seq_length=config.seq_length,
        num_gpus=1,  # per-device 기준으로 계산
        precision_bits=precision_bits,
        safety_margin=0.2
    )
    
    # 5. 결과 출력
    print("\n" + "=" * 80)
    print("추정 결과")
    print("=" * 80)
    print(f"모델 메모리 (총): {model_memory.total_size_mb:.2f} MB")
    print(f"훈련 메모리 (Adam): {model_memory.training_adam_mb:.2f} MB ({model_memory.training_adam_mb / 1024:.2f} GB)")
    print(f"최대 레이어 크기: {model_memory.largest_layer_mb:.2f} MB")
    print("\n권장 batch size 설정:")
    print(f"  per_device_train_batch_size: {max_batch}")
    print("=" * 80)
    
    return max_batch
