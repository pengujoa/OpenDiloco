"""
학습 속도를 측정하는 모듈
실제 학습 환경과 유사한 조건에서 학습 속도를 벤치마크합니다.
"""

import os

import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler


def measure_steps_per_second_with_model(
    model,
    batch_size,
    seq_length,
    precision,
    gradient_accumulation_steps=1,
    vocab_size=32000,
    pp_schedule=None,
    local_world_size: int = 1,
):
    """
    실제 모델을 사용하여 학습 속도를 측정합니다.
    FSDP, TP, PP 등 다양한 병렬화 전략의 모델을 지원합니다.

    Args:
        model: wrapped model (FSDP, TP, PP, or plain — already on GPU)
        batch_size: per-device micro batch size
        seq_length: sequence length
        precision: "fp16-mixed", "bf16-mixed", or "32-true"
        gradient_accumulation_steps: micro-steps per real step
        vocab_size: vocabulary size for random input generation

    Returns:
        steps_per_sec: real steps per second (median of NUM_RUNS)
    """
    WARMUP_STEPS = 5
    BENCHMARK_STEPS = 10
    NUM_RUNS = 3

    if precision == "bf16-mixed":
        dtype = torch.bfloat16
        use_scaler = False
    elif precision == "fp16-mixed":
        dtype = torch.float16
        use_scaler = True
    else:
        dtype = torch.float32
        use_scaler = False

    half_precision = precision in ["fp16-mixed", "bf16-mixed"]
    dev = next(model.parameters()).device

    temp_optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    temp_scaler = GradScaler(enabled=use_scaler)

    import torch.distributed as dist

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=dev)
    attention_mask = torch.ones((batch_size, seq_length), device=dev, dtype=torch.long)
    labels = input_ids.clone()

    # PP: all ranks must use identical microbatch tensors.
    if pp_schedule is not None and dist.is_initialized():
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        if lr == 0:
            pass
        else:
            input_ids = torch.empty(batch_size, seq_length, dtype=torch.long, device=dev)
            attention_mask = torch.empty(batch_size, seq_length, dtype=torch.long, device=dev)
            labels = torch.empty(batch_size, seq_length, dtype=torch.long, device=dev)
        dist.broadcast(input_ids, src=0)
        dist.broadcast(attention_mask, src=0)
        dist.broadcast(labels, src=0)

    has_no_sync = hasattr(model, "no_sync")
    model.train()

    def run_step():
        temp_optimizer.zero_grad()
        for micro_step in range(gradient_accumulation_steps):
            is_accumulating = micro_step < gradient_accumulation_steps - 1
            ctx = model.no_sync() if (is_accumulating and has_no_sync) else nullcontext()
            with ctx:
                with torch.cuda.amp.autocast(enabled=half_precision, dtype=dtype):
                    if pp_schedule is not None:
                        losses_list = []
                        pp_schedule.step(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            target=labels,
                            losses=losses_list,
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / gradient_accumulation_steps
                        temp_scaler.scale(loss).backward()
        if pp_schedule is not None:
            from model_parallel_utils import mp_clip_grad_norm_

            mp_clip_grad_norm_(model, 1.0)
            temp_optimizer.step()
        else:
            if use_scaler:
                from model_parallel_utils import mp_scaler_unscale_

                mp_scaler_unscale_(temp_scaler, temp_optimizer, model)
            from model_parallel_utils import mp_clip_grad_norm_

            mp_clip_grad_norm_(model, 1.0)
            temp_scaler.step(temp_optimizer)
            temp_scaler.update()

    # Warmup
    for _ in range(WARMUP_STEPS):
        run_step()
    torch.cuda.synchronize(dev)

    # Benchmark
    elapsed_times = []
    for run_idx in range(NUM_RUNS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(BENCHMARK_STEPS):
            run_step()
        end_event.record()
        torch.cuda.synchronize(dev)

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0
        elapsed_times.append(BENCHMARK_STEPS / elapsed_sec)

    elapsed_times.sort()
    median_steps_per_sec = elapsed_times[len(elapsed_times) // 2]

    mean_steps_per_sec = sum(elapsed_times) / len(elapsed_times)
    variance = sum((x - mean_steps_per_sec) ** 2 for x in elapsed_times) / len(elapsed_times)
    std_steps_per_sec = variance ** 0.5

    print(f"[Speed Profiler] Model benchmark (GA={gradient_accumulation_steps})")
    print(f"[Speed Profiler] Runs: {NUM_RUNS}, Steps/run: {BENCHMARK_STEPS}")
    print(f"[Speed Profiler] Mean: {mean_steps_per_sec:.4f}, Median: {median_steps_per_sec:.4f}, Std: {std_steps_per_sec:.4f} steps/sec")
    print(f"[Speed Profiler] Individual runs: {[f'{x:.4f}' for x in elapsed_times]}")

    # Clean up: break reference chains before deallocation to avoid
    # "Deallocating Tensor that still has live PyObject references" warnings
    model.zero_grad(set_to_none=True)
    temp_optimizer.state.clear()
    del temp_optimizer, temp_scaler, input_ids, attention_mask, labels
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return median_steps_per_sec
