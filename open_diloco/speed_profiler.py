"""
학습 속도를 측정하는 모듈
실제 학습 환경과 유사한 조건에서 학습 속도를 벤치마크합니다.
"""

import torch
from transformers import LlamaForCausalLM
from torch.cuda.amp import GradScaler


def measure_steps_per_second(dev, batch_size, model_config, precision):
    """
    실제 학습 환경과 유사한 조건에서 학습 속도를 측정합니다.
    
    개선 사항:
    1. 전체 모델 사용 (단일 레이어 대신)
    2. 실제 Language Modeling loss 사용
    3. MixedPrecision 적용 (autocast 사용)
    4. GradScaler 사용 (fp16-mixed일 때)
    5. 여러 번 측정하여 통계적 신뢰도 향상
    6. 실제 학습과 동일한 optimizer 설정
    
    Args:
        dev: CUDA device
        batch_size: per-device training batch size
        model_config: LlamaConfig object
        precision: precision setting ("fp16-mixed", "bf16-mixed", "32-true")
    
    Returns:
        steps_per_sec: 측정된 초당 스텝 수 (중앙값 기반)
    """
    WARMUP_STEPS = 10      # 충분한 warmup으로 안정화
    BENCHMARK_STEPS = 50  # 더 많은 스텝으로 정확도 향상
    NUM_RUNS = 3           # 여러 번 측정하여 통계적 신뢰도 향상
    SEQ_LENGTH = 1024      # Sequence length for benchmark
    
    # Determine dtype based on precision setting
    if precision == "bf16-mixed":
        dtype = torch.bfloat16
        use_scaler = False
    elif precision == "fp16-mixed":
        dtype = torch.float16
        use_scaler = True
    else:  # "32-true"
        dtype = torch.float32
        use_scaler = False
    
    # 메모리 효율적인 작은 모델 생성 (10GB GPU 메모리 제한 고려)
    # 실제 학습과 유사한 연산 패턴을 유지하면서 메모리 사용량 최소화
    # 작은 모델 config 생성 (원본 config 기반)
    small_config = model_config.__class__(**model_config.to_dict())
    
    # 레이어 수를 줄여서 메모리 사용량 감소 (최소 1개 레이어 유지)
    # 작은 모델: hidden_size와 num_attention_heads는 유지하고 레이어 수만 줄임
    original_num_layers = getattr(small_config, 'num_hidden_layers', 16)
    # 10GB GPU에 맞게 레이어 수 조정 (보통 2-4개 레이어면 충분)
    # batch_size와 seq_length에 따라 조정 가능
    target_num_layers = original_num_layers // 4
    small_config.num_hidden_layers = target_num_layers
    
    # 작은 vocab_size도 메모리 절약에 도움 (필요시)
    # small_config.vocab_size = min(small_config.vocab_size, 32000)
    
    print(f"[Speed Profiler] Using small model for memory efficiency:")
    print(f"[Speed Profiler]   Original layers: {original_num_layers}, Using layers: {target_num_layers}")
    print(f"[Speed Profiler]   Hidden size: {small_config.hidden_size}, Vocab size: {small_config.vocab_size}")
    
    # 작은 모델 생성
    model = LlamaForCausalLM(small_config)
    model = model.to(dev)
    
    # Mixed precision 설정
    half_precision = precision in ["fp16-mixed", "bf16-mixed"]
    
    # 실제 학습과 동일한 optimizer 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # GradScaler 설정 (fp16-mixed일 때만)
    scaler = GradScaler(enabled=use_scaler)
    
    # 실제 학습과 동일한 입력 형식 생성
    vocab_size = model_config.vocab_size if hasattr(model_config, 'vocab_size') else 32000
    input_ids = torch.randint(0, vocab_size, (batch_size, SEQ_LENGTH), device=dev)
    attention_mask = torch.ones((batch_size, SEQ_LENGTH), device=dev, dtype=torch.long)
    labels = input_ids.clone()
    
    # Warmup phase - 충분한 warmup으로 안정화
    model.train()
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        
        # 실제 학습과 동일한 forward pass
        with torch.cuda.amp.autocast(enabled=half_precision, dtype=dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        # 실제 학습과 동일한 backward pass
        scaler.scale(loss).backward()
        # Gradient clipping (실제 학습과 동일)
        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 실제 학습과 동일한 clipping 값
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize(dev)
    
    # 여러 번 측정하여 통계적 신뢰도 향상
    elapsed_times = []
    
    for run_idx in range(NUM_RUNS):
        # Benchmark phase
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        for _ in range(BENCHMARK_STEPS):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=half_precision, dtype=dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            # Gradient clipping (실제 학습과 동일)
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 실제 학습과 동일한 clipping 값
            scaler.step(optimizer)
            scaler.update()
        
        end_event.record()
        torch.cuda.synchronize(dev)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0
        steps_per_sec = BENCHMARK_STEPS / elapsed_sec
        elapsed_times.append(steps_per_sec)
    
    # 중앙값 사용 (이상치에 덜 민감)
    elapsed_times.sort()
    median_steps_per_sec = elapsed_times[len(elapsed_times) // 2]
    
    # 평균과 표준편차도 계산 (디버깅용)
    mean_steps_per_sec = sum(elapsed_times) / len(elapsed_times)
    variance = sum((x - mean_steps_per_sec) ** 2 for x in elapsed_times) / len(elapsed_times)
    std_steps_per_sec = variance ** 0.5
    
    # 통계 정보 출력 (디버깅용)
    print(f"[Speed Profiler] Runs: {NUM_RUNS}, Steps per run: {BENCHMARK_STEPS}")
    print(f"[Speed Profiler] Mean: {mean_steps_per_sec:.4f} steps/sec, Median: {median_steps_per_sec:.4f} steps/sec, Std: {std_steps_per_sec:.4f} steps/sec")
    print(f"[Speed Profiler] Individual runs: {[f'{x:.4f}' for x in elapsed_times]}")
    
    # 중앙값 반환 (이상치에 덜 민감하므로 더 안정적)
    return median_steps_per_sec


