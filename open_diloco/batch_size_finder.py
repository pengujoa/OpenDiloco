"""
OoM을 발생시키지 않는 최대 batch_size를 탐색하는 모듈
Accelerate의 메모리 추정기를 사용하여 batch size를 자동으로 찾습니다.
"""

import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryEstimate:
    """메모리 추정 결과를 저장하는 클래스"""
    dtype: str
    largest_layer_mb: float
    total_size_mb: float
    training_adam_mb: float


def parse_size_to_mb(size_str: str) -> float:
    """크기 문자열을 MB 단위로 변환 (예: "418.18 MB" -> 418.18)"""
    size_str = size_str.strip()
    
    # 숫자 부분 추출
    match = re.search(r'([\d.]+)', size_str)
    if not match:
        return 0.0
    
    value = float(match.group(1))
    
    # 단위 변환
    if 'GB' in size_str.upper():
        return value * 1024
    elif 'TB' in size_str.upper():
        return value * 1024 * 1024
    elif 'KB' in size_str.upper():
        return value / 1024
    else:  # MB
        return value


def estimate_model_memory(model_name: str, library_name: Optional[str] = None) -> dict[str, MemoryEstimate]:
    """
    Accelerate를 사용하여 모델의 메모리 사용량을 추정합니다.
    
    Args:
        model_name: 모델 이름 (예: "bert-base-cased", "PrimeIntellect/llama-150m-fresh")
        library_name: 모델 라이브러리 (예: "transformers", "timm")
    
    Returns:
        dtype별 메모리 추정 결과 딕셔너리
    """
    cmd = ["accelerate", "estimate-memory", model_name]
    
    if library_name:
        cmd.extend(["--library_name", library_name])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        print(f"Accelerate 출력:\n{output}\n")
        
        # 결과 파싱
        estimates = {}
        lines = output.split('\n')
        
        # 헤더 라인 찾기
        header_idx = None
        for i, line in enumerate(lines):
            if '| dtype' in line and 'Largest Layer' in line:
                header_idx = i
                break
        
        if header_idx is None:
            raise ValueError("메모리 추정 결과를 찾을 수 없습니다")
        
        # 데이터 라인 파싱
        for i in range(header_idx + 2, len(lines)):
            line = lines[i].strip()
            if not line or not line.startswith('|'):
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 5:
                continue
            
            dtype = parts[1].strip()
            if dtype not in ['float32', 'float16', 'int8', 'int4']:
                continue
            
            estimates[dtype] = MemoryEstimate(
                dtype=dtype,
                largest_layer_mb=parse_size_to_mb(parts[2]),
                total_size_mb=parse_size_to_mb(parts[3]),
                training_adam_mb=parse_size_to_mb(parts[4])
            )
        
        return estimates
    
    except subprocess.CalledProcessError as e:
        print(f"Accelerate 실행 오류: {e}")
        print(f"출력: {e.stdout}")
        print(f"오류: {e.stderr}")
        return {}
    except Exception as e:
        print(f"메모리 추정 중 오류 발생: {e}")
        return {}


def calculate_max_batch_size(
    model_memory_mb: float,
    available_gpu_memory_gb: float,
    seq_length: int,
    num_gpus: int = 1,
    precision_bits: int = 16,
    safety_margin: float = 0.2
) -> int:
    """
    주어진 GPU 메모리에서 실행 가능한 최대 batch size를 계산합니다.
    
    Args:
        model_memory_mb: 모델 메모리 사용량 (MB)
        available_gpu_memory_gb: 사용 가능한 GPU 메모리 (GB)
        seq_length: 시퀀스 길이
        num_gpus: GPU 개수
        precision_bits: 정밀도 비트 (16 또는 32)
        safety_margin: 안전 여유율 (기본 20%)
    
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
    
    # 각 샘플의 메모리 계산 (중복 계산으로 추정)
    # 활성화 메모리: (hidden_size * num_layers * seq_length * batch_size)
    # 정확한 계산을 위해 추정치 사용
    bytes_per_param = precision_bits / 8
    estimated_memory_per_sample_mb = (model_memory_mb * bytes_per_param * seq_length) / (1000 * 1000)
    
    # 한 샘플당 약 2-4배의 활성화 메모리 고려
    # LLM의 경우 forward pass에서 여러 중간 활성화가 필요하므로
    activation_memory_multiplier = 3.0
    total_memory_per_sample_mb = estimated_memory_per_sample_mb * activation_memory_multiplier
    
    # 최대 batch size 계산
    if total_memory_per_sample_mb <= 0:
        return 1
    
    max_batch_size = int(batch_memory_mb / total_memory_per_sample_mb)
    max_batch_size = max(1, max_batch_size)  # 최소값은 1
    
    return max_batch_size


def find_max_batch_size_for_model(config) -> None:
    """
    OoM 없이 실행 가능한 최대 batch size를 탐색하고 출력합니다.
    
    Args:
        config: Config 설정 객체
    """
    print("=" * 80)
    print("최대 Batch Size 탐색 도구")
    print("=" * 80)
    print(f"모델: {config.path_model}")
    
    if config.available_gpu_memory_gb is None:
        print("오류: --available-gpu-memory-gb 옵션을 지정해야 합니다.")
        print("예: --available-gpu-memory-gb 24.0")
        return
    
    print(f"사용 가능한 GPU 메모리: {config.available_gpu_memory_gb} GB")
    print(f"시퀀스 길이: {config.seq_length}")
    print(f"정밀도: {config.precision}")
    print("=" * 80)
    
    # 1. 모델 메모리 추정
    print("\n1단계: 모델 메모리 추정 중...")
    estimates = estimate_model_memory(config.path_model)
    
    if not estimates:
        print("메모리 추정에 실패했습니다. Accelerate가 설치되어 있는지 확인하세요.")
        print("설치: pip install accelerate")
        return
    
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
        available_gpu_memory_gb=config.available_gpu_memory_gb,
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

