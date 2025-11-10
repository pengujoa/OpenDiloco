"""
학습 속도를 측정하는 모듈
LLM attention block을 사용하여 실제 학습 속도를 벤치마크합니다.
"""

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding


def measure_steps_per_second(dev, batch_size, model_config, precision):
    """
    Measure training steps per second using actual LLM attention block.
    
    Args:
        dev: CUDA device
        batch_size: per-device training batch size
        model_config: LlamaConfig object
        precision: precision setting ("fp16-mixed", "bf16-mixed", "32-true")
    
    Returns:
        steps_per_sec: 측정된 초당 스텝 수
    """
    WARMUP_STEPS = 5      # Reduced from 20 to minimize overhead
    BENCHMARK_STEPS = 50  # Reduced from 500 to minimize overhead
    SEQ_LENGTH = 1024     # Sequence length for benchmark
    
    # Determine dtype based on precision setting
    if precision == "bf16-mixed":
        dtype = torch.bfloat16
    elif precision == "fp16-mixed":
        dtype = torch.float16
    else:  # "32-true"
        dtype = torch.float32
    
    # Create a single decoder layer (1 attention block) for benchmarking
    model = LlamaDecoderLayer(model_config, layer_idx=0).to(dev, dtype=dtype)
    
    # Create RoPE (Rotary Position Embeddings) for position encoding
    rotary_emb = LlamaRotaryEmbedding(config=model_config).to(dev, dtype=dtype)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create dummy inputs matching LLM format
    # hidden_states: [batch_size, seq_length, hidden_size]
    hidden_states = torch.randn(
        batch_size, SEQ_LENGTH, model_config.hidden_size, 
        device=dev, dtype=dtype
    )
    
    # Create position_ids for RoPE
    position_ids = torch.arange(SEQ_LENGTH, device=dev).unsqueeze(0).expand(batch_size, -1)
    
    # Generate position embeddings (cos, sin) for RoPE
    position_embeddings = rotary_emb(hidden_states, position_ids)
    
    # Warmup phase
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        outputs = model(hidden_states, position_embeddings=position_embeddings)
        hidden_output = outputs[0]  # LlamaDecoderLayer returns tuple
        # Simple loss: mean of outputs
        loss = hidden_output.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(dev)

    # Benchmark phase
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(BENCHMARK_STEPS):
        optimizer.zero_grad()
        outputs = model(hidden_states, position_embeddings=position_embeddings)
        hidden_output = outputs[0]
        loss = hidden_output.mean()
        loss.backward()
        optimizer.step()
    end.record()
    
    torch.cuda.synchronize(dev)
    
    elapsed_ms = start.elapsed_time(end)
    elapsed_sec = elapsed_ms / 1000
    steps_per_sec = BENCHMARK_STEPS / elapsed_sec
    
    return steps_per_sec


