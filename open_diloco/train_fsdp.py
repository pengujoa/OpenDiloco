"""

to test quickly do 
torchrun --nproc_per_node=2 \
        train_fsdp.py --per-device-train-batch-size 8 --total-batch-size 128 --lr 1e-2 --path-model ../tests/models/llama-2m-fresh \
        --no-torch-compile --log-activations-steps 5 --fake-data --max-steps 20
"""

from functools import partial
import os
import socket
import time
import math
import re
from contextlib import nullcontext
import datetime
from typing import Any, Literal, Dict
import sys

from pydantic import field_validator, Field
import torch
import torch.nn as nn
from typing import List, Union
from pydantic_config import parse_argv, BaseConfig

# TOML 파일 파싱 지원
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            tomllib = None
            print("Warning: TOML support not available. Install 'tomli' or 'toml' package for Python < 3.11")
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import destroy_process_group, init_process_group

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.device_mesh import init_device_mesh

from ckpt_utils import (
    CKPT_PREFIX,
    CkptConfig,
    check_checkpoint_path_access,
    delete_old_checkpoints,
    get_diloco_rank_dir_name,
    get_resume_info,
    load_checkpoint,
    save_checkpoint,
)
from hivemind_diloco import AllReduceStrategy, DiLoCoOptimizer
from open_diloco.utils import WandbLogger, DummyLogger

from hivemind.dht.dht import DHT
from hivemind.utils import get_dht_time
from hivemind.utils.networking import log_visible_maddrs
from hivemind.optim.optimizer import logger


try:
    from .memory_debugger import MemoryUsageTracker, bytes_to_gb  # type: ignore[attr-defined]
except ImportError:
    from memory_debugger import MemoryUsageTracker, bytes_to_gb
from open_diloco.utils import (
    FakeTokenizedDataset,
    get_compression_kwargs,
    get_sharding_strategy,
    register_metrics_hooks,
)
from batch_size_finder import find_max_batch_size_for_model
from speed_profiler import measure_steps_per_second

from peft import get_peft_model, LoraConfig, TaskType


TIMEOUT_NCCL_MINUTES = os.environ.get("TIMEOUT_NCCL_MINUTES", 120)
TARGET_LAYER_ACTIVATIONS = ["self_attn", "lm_head"]
TEST_VOCAB_SIZE = 1024


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # CAUSAL_LM is for language generation tasks
    inference_mode=False,           # Enable training mode
    r=8,                            # LoRA rank
    lora_alpha=32,                  # Scaling factor
    lora_dropout=0.1                # Dropout rate
)

# Function to compare the number of parameters before/after adapt lora
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # print(model)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def print_all_parameter_names(model, rank: int = 0, print_all: bool = False):
    """디버그용: 모델의 모든 파라미터 이름을 출력합니다."""
    if rank != 0 and not print_all:
        return
    
    print("\n" + "="*80)
    print("MODEL PARAMETER NAMES (DEBUG)")
    print("="*80)
    
    param_names = []
    layer_groups = {}
    
    for name, param in model.named_parameters():
        param_names.append(name)
        # 레이어 그룹화
        if 'layers.' in name:
            layer_idx = name.split('layers.')[1].split('.')[0]
            if layer_idx not in layer_groups:
                layer_groups[layer_idx] = []
            layer_groups[layer_idx].append(name)
        else:
            if 'root' not in layer_groups:
                layer_groups['root'] = []
            layer_groups['root'].append(name)
    
    print(f"\nTotal parameters: {len(param_names)}")
    
    # 레이어별로 그룹 출력 (숫자 순서로 정렬)
    print("\n--- By Layer Groups ---")
    
    # 레이어 키를 숫자와 문자열로 분리하여 정렬
    def sort_key(key):
        if key == 'root':
            return (-1, 'root')  # root는 가장 앞에
        try:
            return (int(key), '')  # 숫자로 변환 가능한 경우
        except ValueError:
            return (999999, key)  # 숫자가 아닌 경우 뒤로
    
    sorted_keys = sorted(layer_groups.keys(), key=sort_key)
    
    for layer_key in sorted_keys:
        if layer_key == 'root':
            print(f"\n[Root/Embedding/Head]: {len(layer_groups[layer_key])} params")
        else:
            print(f"\n[Layer {layer_key}]: {len(layer_groups[layer_key])} params")
        for name in sorted(layer_groups[layer_key]):  # 각 그룹 내에서도 정렬
            print(f"  {name}")
    
    # 선택적 레이어 업데이트를 위한 패턴 예시
    print("\n--- Example Selective Layer Patterns ---")
    print("First layer: ['model.layers.0']")
    print("First two layers: ['model.layers.0', 'model.layers.1']")
    print("Output head: ['lm_head']")
    print("Embedding + Output: ['model.embed_tokens', 'lm_head']")
    
    # 전체 파라미터 이름 목록
    print("\n--- All Parameter Names ---")
    for i, name in enumerate(param_names):
        print(f"{i+1:3d}. {name}")
    
    print("="*80 + "\n")
    
    return param_names


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


class HvConfig(BaseConfig):
    outer_lr: float = 0.7
    local_steps: int = 500
    initial_peers: list[str] | None = None
    host_maddrs: list[str] = ["/ip4/0.0.0.0/tcp/0"]
    announce_maddrs: list[str] | None = None
    matchmaking_time: float | None = None
    averaging_timeout: float | None = None
    hivemind_compression: Literal["fp16", "scaled-fp16", "uniform8bit", "quantile8bit", "blockwise8bit"] | None = None
    all_reduce_strategy: AllReduceStrategy = AllReduceStrategy.WAIT_FOR_ALL
    timeout_waiting_for_peers: float | None = None
    skip_load_from_peers: bool = False
    world_rank: int
    galaxy_size: int
    fail_rank_drop: bool = False  # fail if we lose a diloco worker
    # Selective layer update
    selective_layer_patterns: list[str] | None = None  # 업데이트할 레이어 패턴 목록 (예: ["model.layers.0", "model.layers.1", "lm_head"])

    
    @field_validator('initial_peers', mode='before')
    def _parse_str_to_str_list(cls, v: Union[str, List[str]]) -> List[str]:
        # 이미 리스트면 그대로 반환
        if isinstance(v, list):
            return v
        # 문자열인 경우 쉼표로 분할해 리스트로 변환
        return [item.strip() for item in v.strip('[]').split(',') if item.strip()]


class Config(BaseConfig):
    path_model: str = "PrimeIntellect/llama-150m-fresh"
    torch_compile: bool = True
    attn_implementation: str = "sdpa"
    # Data
    dataset_name_or_path: str = "allenai/c4"
    seq_length: int = 1024
    c4_tiny: bool = False
    num_workers: int = 4
    # Optimization
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    warmup_steps: int = 1000
    total_steps: int = 88_000
    sharding_strategy: str = "NO_SHARD"
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    # Checkpointing and logging
    project: str = "hivemind_debug"
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    log_activations_steps: int | None = None
    log_memory_breakdown_steps: int | None = None
    ckpt: CkptConfig = CkptConfig()
    # Hivemind
    hv: HvConfig | None = None  # if no hv config then hivemind is disabled
    fake_data: bool = False
    max_steps: int | None = None
    # Node-specific GPU configuration
    node_gpu_counts: list[int] = Field(..., description="List of GPU counts per node. Must be provided.") # List to store GPU counts per node
    # Lora
    lora: bool | None = False
    # Batch size finder
    find_max_batch_size: bool = False  # If True, estimate max batch size using Accelerate and exit

    @field_validator('node_gpu_counts', mode='before')
    def _parse_str_to_int_list(cls, v):
        if isinstance(v, list):
            return v
        items = v.strip('[]').split(',')
        return [int(item) for item in items if item.strip()]
    
    
def get_dataloader(tokenizer, world_size, rank, local_rank, config: Config) -> StatefulDataLoader:
    if config.fake_data:
        train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)
    else:
        # dataset NFS로 copy 해오면서 전체 디렉토리 복사 한게 아니어서 이렇게 설정
        # ds = load_dataset(config.dataset_name_or_path, "en", streaming=True, trust_remote_code=True)
        ds = load_dataset(config.dataset_name_or_path, "default", streaming=True, trust_remote_code=True)

        def tokenize_function(data):
            outputs = tokenizer(
                data["text"],
                truncation=True,
                max_length=config.seq_length,
                padding="max_length",
            )
            return outputs

        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])[
            "train"
        ]

        if config.hv is not None:
            print("MY LOCAL RANK: ", local_rank)
            print("MY RANK: ", sum(config.node_gpu_counts[:config.hv.world_rank]) + local_rank)
            train_dataset = split_dataset_by_node(
                tokenized_datasets,
                world_size=sum(config.node_gpu_counts),
                rank=sum(config.node_gpu_counts[:config.hv.world_rank]) + local_rank,
            )

        else:
            train_dataset = split_dataset_by_node(tokenized_datasets, world_size=world_size, rank=rank)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return StatefulDataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
    )


def get_model(config: Config) -> LlamaForCausalLM:
    # Load model
    config_model = LlamaConfig.from_pretrained(config.path_model, attn_implementation=config.attn_implementation, resume_download=True)
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=config.path_model, config=config_model, resume_download=True)
    if config.lora:
        print_trainable_parameters(model)
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
    return model


def read_speeds(dht: DHT, key: str) -> Dict[str, float]:
    res = dht.get(key, latest=True)
    root = unwrap(res) if res else None
    speeds: Dict[str, float] = {}
    if isinstance(root, dict):
        for k, v in root.items():
            p = unwrap(v)
            if isinstance(p, dict):
                if "steps_per_sec" in p and isinstance(p["steps_per_sec"], (int, float)):
                    speeds[k] = float(p["steps_per_sec"])
                elif "v" in p and isinstance(p["v"], (int, float)):
                    speeds[k] = float(p["v"])
                elif "step_time_s" in p and isinstance(p["step_time_s"], (int, float)) and p["step_time_s"] > 0:
                    speeds[k] = 1.0 / float(p["step_time_s"])
    return speeds


def unwrap(v):
    return getattr(v, "value", v)


def wait_for_all_nodes_ready(dht: DHT, galaxy_size: int, log_fn):
    """galaxy_size만큼의 노드가 준비될 때까지 대기"""
    log_fn("Waiting for all nodes to be ready before starting training...")
    
    RUN_ID = "OpenDiLoCo"
    ready_key = f"{RUN_ID}:ready"
    worker_id = f"{socket.gethostname()}-pid{os.getpid()}"
    
    # 현재 노드의 ready 상태를 DHT에 publish
    now = get_dht_time()
    ready_payload = {
        "ready": True,
        "ts": now,
        "host": socket.gethostname()
    }
    exp = now + 60.0  # 60초 TTL
    dht.store(key=ready_key, subkey=worker_id, value=ready_payload, expiration_time=exp)
    log_fn(f"Published ready state for {worker_id}")
    
    # galaxy_size만큼의 노드가 준비될 때까지 대기
    while True:
        ready_res = dht.get(ready_key, latest=True)
        ready_root = unwrap(ready_res) if ready_res else None
        ready_count = 0
        
        if isinstance(ready_root, dict):
            for k, v in ready_root.items():
                p = unwrap(v)
                if isinstance(p, dict) and p.get("ready") is True:
                    ready_count += 1
        
        log_fn(f"Ready nodes: {ready_count}/{galaxy_size}")
        
        if ready_count >= galaxy_size:
            log_fn(f"All {galaxy_size} nodes are ready!")
            break
        
        time.sleep(2.0)  # 2초마다 확인
    
    log_fn("All nodes are ready. Starting training loop.")


def train(config: Config):
    sharding_strategy = get_sharding_strategy(config.sharding_strategy)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    world_messenger_hv = config.hv is not None and local_rank == 0

    # batch_size is the total batch size for all GPUs
    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size

    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    if config.hv is not None:
        sharding_strategy = ShardingStrategy.NO_SHARD
        log("Hivemind is used, ShardingStrategy.NO_SHARD is used")

    resume_from_ckpt, resume_path = get_resume_info(config.ckpt)

    if rank == 0:
        logger_cls = WandbLogger if config.metric_logger_type == "wandb" else DummyLogger
        metric_logger = logger_cls(project=config.project, config=config.model_dump(), resume=resume_from_ckpt)

    if config.hv is not None:
        log("hivemind diloco enabled")

    if world_messenger_hv:
        # 원래 입력으로 받은 local_steps 값 저장
        original_local_steps = config.hv.local_steps
        print(f"[INFO] Original local_steps from input: {original_local_steps}")
        
        dht = DHT(
            start=True,
            initial_peers=config.hv.initial_peers,
            host_maddrs=config.hv.host_maddrs,
            announce_maddrs=config.hv.announce_maddrs,
        )
        log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

        # Measure speeds & compute inner (local) steps
        PUBLISH_INTERVAL = 10.0
        TTL = 30.0
        # 1) key
        RUN_ID = "OpenDiLoCo"
        key = f"{RUN_ID}:speed"
        # 2) subkey
        GPU = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(GPU)
        gpu_name = torch.cuda.get_device_name(GPU)
        worker_id = f"{socket.gethostname()}-pid{os.getpid()}-gpu{GPU}"
        # 3) Load model config for benchmarking
        model_config = LlamaConfig.from_pretrained(config.path_model, attn_implementation=config.attn_implementation, resume_download=True)
        # 4) measure speed using actual batch size, model config, and precision
        print(f"[{worker_id}] Starting speed profiling with precision={config.precision}...")
        profiling_start_time = time.time()
        steps_per_sec = measure_steps_per_second(GPU, config.per_device_train_batch_size, model_config, config.precision)
        profiling_elapsed_time = time.time() - profiling_start_time
        print(f"[{worker_id}] Speed profiling completed in {profiling_elapsed_time:.2f} seconds")
        # repeat
        while True:
            # 4) value            
            now = get_dht_time()        
            payload = {
                "steps_per_sec": float(steps_per_sec), 
                "ts": now, 
                "host": socket.gethostname(), 
                "gpu_id": GPU,
                "gpu_name": gpu_name
            }
            # 5) expiration time            
            exp = now + TTL
            # 6) store in DHT        
            ok = dht.store(key=key, subkey=worker_id, value=payload, expiration_time=exp)     
            # 7) print result 
            print(f"[publish] {worker_id} ({gpu_name}): {steps_per_sec:.2f} steps/sec ({'ok' if ok else 'fail'})")

            # 8) read all speeds
            speeds = read_speeds(dht, key)
            if not speeds:
                print(f"[error] no usable speeds at '{key}'. Is the publisher running?")
                return 2
            if len(speeds) < config.hv.galaxy_size:
                print(len(speeds), "speeds found, waiting for", config.hv.galaxy_size)
                # 다음 PUBLISH_INTERVAL초 단위 경계까지 대기
                time_in_cycle = now % PUBLISH_INTERVAL
                sleep_duration = PUBLISH_INTERVAL - time_in_cycle
                print(f"[sync] Waiting {sleep_duration:.2f}s until next {PUBLISH_INTERVAL}s boundary")
                time.sleep(sleep_duration)
            else:
                break
                
        # 9) compute inner (local) steps based on speed distribution
        # 전체 epoch의 작업량 = 노드 개수 * 원래 local_steps
        total_work = config.hv.galaxy_size * original_local_steps
        print(f"[INFO] Total work per epoch: {config.hv.galaxy_size} nodes × {original_local_steps} steps = {total_work} steps")
        
        # 모든 노드의 속도를 정렬된 리스트로 변환 (worker_id, speed)
        sorted_speeds = sorted(speeds.items(), key=lambda x: x[1], reverse=True)  # 속도 내림차순
        total_speed = sum(speeds.values())
        print(f"[INFO] Total speed across all nodes: {total_speed:.2f} steps/sec")
        
        # 각 노드의 local_steps 할당 계산 (비례 분배)
        allocations = {}
        allocated_sum = 0
        
        for i, (wid, speed) in enumerate(sorted_speeds):
            speed_ratio = speed / total_speed
            if i < len(sorted_speeds) - 1:
                # 마지막 노드가 아닌 경우 floor 사용
                allocated_steps = int(math.floor(total_work * speed_ratio))
                allocated_steps = max(1, allocated_steps)  # 최소 1 step 보장
            else:
                # 마지막 노드는 나머지를 모두 할당하여 total_work를 정확히 맞춤
                allocated_steps = total_work - allocated_sum
                allocated_steps = max(1, allocated_steps)  # 최소 1 step 보장
            
            allocations[wid] = allocated_steps
            allocated_sum += allocated_steps
        
        # 현재 노드의 local_steps 설정
        config.hv.local_steps = allocations[worker_id]
        
        # 검증: 모든 할당량의 합이 total_work와 일치하는지 확인
        print(f"[INFO] Local steps allocation verification:")
        print(f"[INFO]   - Total work: {total_work}")
        print(f"[INFO]   - Sum of allocations: {allocated_sum}")
        print(f"[INFO]   - Match: {allocated_sum == total_work}")
        
        # 현재 노드의 속도 비율
        speed_ratio = steps_per_sec / total_speed
        print(f"[INFO] Speed distribution-based local_steps allocation:")
        print(f"[INFO]   - Current node speed: {steps_per_sec:.2f} steps/sec ({speed_ratio*100:.2f}%)")
        print(f"[INFO]   - Allocated local_steps: {config.hv.local_steps}")
        print(f"[INFO]   - Actual contribution: {config.hv.local_steps / total_work * 100:.2f}% of total work")
        print(f"[DEBUG] batch_size={batch_size}, target_batch_size will be={batch_size * config.hv.local_steps}")
    
    # Broadcast local_steps to all ranks to ensure consistency
    if config.hv is not None:
        # local_rank 0에서 계산된 값을 모든 rank에 broadcast
        local_steps_tensor = torch.tensor([config.hv.local_steps], dtype=torch.int32, device='cuda')
        torch.distributed.broadcast(local_steps_tensor, src=0)
        config.hv.local_steps = int(local_steps_tensor.item())
        print(f"[DEBUG] rank={rank}, local_rank={local_rank}: synchronized local_steps={config.hv.local_steps}")
            

    if local_rank == 0:
        check_checkpoint_path_access(config.ckpt.path, rank, config.hv.world_rank if config.hv else None)

    # DataLoader preparation
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True, resume_download=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config)

    model = get_model(config)
    model = model.to(local_rank)

    # 디버그: 파라미터 이름 출력 (FSDP 래핑 전에)
    if rank == 0:
        print_all_parameter_names(model, rank=rank)

    half_precision = config.precision == "fp16-mixed" or config.precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if config.precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16-mixed")

    if sharding_strategy in [
        ShardingStrategy._HYBRID_SHARD_ZERO2,
        ShardingStrategy.HYBRID_SHARD,
    ]:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        nnodes = world_size // local_world_size
        device_mesh = init_device_mesh("cuda", (nnodes, local_world_size), mesh_dim_names=("global", "local"))
    else:
        device_mesh = None
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=half_precision_dtype) if half_precision else None,
        use_orig_params=config.torch_compile,
        device_mesh=device_mesh,
    )
    if config.torch_compile:
        model = torch.compile(model)

    # Setup optimizers
    inner_optimizer = partial(torch.optim.AdamW, lr=config.lr, weight_decay=0.1, betas=(0.9, 0.95))  # noqa: F821

    if config.hv is not None:
        outer_optimizer = partial(torch.optim.SGD, lr=config.hv.outer_lr, momentum=0.9, nesterov=True)

    def scheduler_fn(opt):
        return get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

    if config.hv is not None:
        if resume_from_ckpt:
            # We need to load with a fake optimizer to set the model parameters correctly before initializing the DiLoCoOptimizer
            # This is because the DiLoCoOptimizer makes a copy of the model parameters for the state averager which is hard to update later
            # We also need to do this on follower workers so that the world_messenger has friends to talk to when it does its two loads
            # Otherwise the world messenger will get lonely and hang
            # params = list(model.parameters())
            fake_optimizer = inner_optimizer(model.parameters())
            last_loss = load_checkpoint(
                checkpoint_path=os.path.join(resume_path, get_diloco_rank_dir_name(config.hv.world_rank)),
                model=model,
                optimizer=fake_optimizer,
                lora=config.lora,
                dataset=config.dataset_name_or_path,
            )
            del fake_optimizer

    if resume_from_ckpt:
        if config.hv is not None:
            ckpt_path = os.path.join(resume_path, get_diloco_rank_dir_name(config.hv.world_rank))
        else:
            ckpt_path = resume_path

    if world_messenger_hv:
        # Selective layer update를 위한 파라미터 이름 추출
        param_names = None
        if config.hv.selective_layer_patterns is not None:
            # 모델에서 파라미터 이름 추출 (trainable 파라미터만)
            if config.lora:
                # LoRA의 경우 trainable 파라미터만 추출
                param_names = [name for name, param in model.named_parameters() if param.requires_grad]
            else:
                # 모든 파라미터 이름 추출
                param_names = [name for name, _ in model.named_parameters()]
            log(f"Extracted {len(param_names)} parameter names for selective layer update")
            if config.hv.selective_layer_patterns:
                matched_count = sum(
                    1 for name in param_names 
                    if any(pattern in name or name.startswith(pattern) for pattern in config.hv.selective_layer_patterns)
                )
                log(f"Patterns {config.hv.selective_layer_patterns} match {matched_count}/{len(param_names)} parameters")
        
        diloco_args = dict(
            dht=dht,
            run_id="llama",
            batch_size=batch_size,
            num_inner_steps=config.hv.local_steps,
            outer_optimizer=outer_optimizer,
            inner_optimizer=inner_optimizer,
            scheduler=None,
            params=model.parameters(),
            delay_optimizer_step=False,
            delay_grad_averaging=False,
            verbose=True,
            all_reduce_strategy=config.hv.all_reduce_strategy,
            timeout_waiting_for_peers=config.hv.timeout_waiting_for_peers,
            lora=config.lora,
            selective_layer_patterns=config.hv.selective_layer_patterns,
            param_names=param_names,
        )

        diloco_args.update(get_compression_kwargs(config.hv.hivemind_compression))

        if config.hv.averaging_timeout is not None:
            diloco_args["averaging_timeout"] = config.hv.averaging_timeout

        if config.hv.matchmaking_time is not None:
            diloco_args["matchmaking_time"] = config.hv.matchmaking_time

        optimizer = DiLoCoOptimizer(**diloco_args)
        
        print(f"[DEBUG] DiLoCoOptimizer initialized:")
        print(f"[DEBUG]   num_inner_steps={optimizer.num_inner_steps}")
        print(f"[DEBUG]   batch_size_per_step={optimizer.batch_size_per_step}")
        print(f"[DEBUG]   target_batch_size={optimizer.tracker.target_batch_size}")
        print(f"[DEBUG]   tracker.batch_size={optimizer.tracker.batch_size}")

        scheduler = scheduler_fn(
            optimizer.inner_optimizer
        )  # scheduler(optimizer) should work but better to make it explicit here

        if resume_from_ckpt:
            last_loss = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer.inner_optimizer,
                scheduler=scheduler,
                outer_optimizer=optimizer.state_averager.optimizer,
                scaler=scaler,
                data_loader=train_dataloader,
                lora=config.lora,
                dataset=config.dataset_name_or_path,
            )
            # Resume 시 local_steps가 변경되었을 수 있으므로 optimizer 내부 상태를 업데이트
            optimizer.update_num_inner_steps(config.hv.local_steps)
            print(f"[DEBUG] After resume and update_num_inner_steps:")
            print(f"[DEBUG]   num_inner_steps={optimizer.num_inner_steps}")
            print(f"[DEBUG]   target_batch_size={optimizer.tracker.target_batch_size}")
            if config.lora:
                start_step = 0
            else:
                start_step = scheduler.last_epoch
        else:
            start_step = 0
        if config.lora:
            model = get_peft_model(model, lora_config)
            print_trainable_parameters(model)

    else:
        optimizer = inner_optimizer(model.parameters())
        scheduler = scheduler_fn(optimizer)
        if resume_from_ckpt:
            last_loss = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_loader=train_dataloader,
                lora=config.lora,
                dataset=config.dataset_name_or_path,
            )
            start_step = scheduler.last_epoch
        else:
            start_step = 0
        if config.lora:
            model = get_peft_model(model, lora_config)
            print_trainable_parameters(model)

    if resume_from_ckpt:
        log(f"Resumed from checkpoint at step {start_step} with loss {last_loss}")

    model.train()

    memory_tracker = MemoryUsageTracker(model=model, optimizer=optimizer)

    if world_messenger_hv and not config.hv.skip_load_from_peers:
        optimizer.load_state_from_peers()

    current_time = time.time()
    log(f"starting from step {start_step}")

    loss_batch = 0

    if world_messenger_hv:
        max_num_peers = 0

    log_activations = {}
    
    # 모든 노드 준비 체크 플래그
    all_nodes_ready = False

    for step, batch in enumerate(iterable=train_dataloader, start=start_step * gradient_accumulation_steps):
        real_step = (step + 1) // gradient_accumulation_steps
        is_accumulating = bool((step + 1) % gradient_accumulation_steps)
        should_profile_memory = (
            config.log_memory_breakdown_steps is not None
            and not is_accumulating
            and real_step > 0
            and real_step % config.log_memory_breakdown_steps == 0
        )
        memory_breakdown: dict[str, int] | None = None

        logging_activations_steps = (
            config.log_activations_steps is not None and real_step % config.log_activations_steps == 0
        )

        if logging_activations_steps:
            handles = register_metrics_hooks(
                model, TARGET_LAYER_ACTIVATIONS, log_activations, gradient_accumulation_steps
            )

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        with model.no_sync() if is_accumulating else nullcontext():
            activation_cm = memory_tracker.capture_activations() if should_profile_memory else nullcontext()
            with activation_cm:
                outputs = model(**batch)

            # 첫 번째 forward pass 후 모든 노드가 준비될 때까지 대기
            if world_messenger_hv and not all_nodes_ready:
                wait_for_all_nodes_ready(dht, config.hv.galaxy_size, log)
                all_nodes_ready = True

            loss = outputs.loss / gradient_accumulation_steps

            loss_batch += loss.detach()

            scaler.scale(loss).backward()

        if logging_activations_steps:
            for handle in handles:
                handle.remove()

        if not is_accumulating:
            if world_messenger_hv:
                scaler.unscale_(optimizer=optimizer.inner_optimizer)
            else:
                scaler.unscale_(optimizer=optimizer)

            model.clip_grad_norm_(1.0)  # gradient clipping

            if world_messenger_hv:
                if real_step % 10 == 0:  # 10 step마다 로깅
                    print(f"[DEBUG] Before optimizer.step at real_step={real_step}:")
                    print(f"[DEBUG]   samples_accumulated={optimizer.tracker.local_progress.samples_accumulated}")
                    print(f"[DEBUG]   target_batch_size={optimizer.tracker.target_batch_size}")
                    print(f"[DEBUG]   ready_to_update_epoch={optimizer.tracker.ready_to_update_epoch}")
                    print(f"[DEBUG]   local_step={optimizer.tracker.local_step}")
                
                optimizer.step(scaler=scaler)
                
                if real_step % 10 == 0:
                    print(f"[DEBUG] After optimizer.step:")
                    print(f"[DEBUG]   samples_accumulated={optimizer.tracker.local_progress.samples_accumulated}")
                    print(f"[DEBUG]   local_step={optimizer.tracker.local_step}")

                # todo(sami): refactor to use built in pytorch mechanism to handle scaler manually
                # should allow to just do scaler.step(optimizer)
            else:
                scaler.step(optimizer)

            scaler.update()

            if should_profile_memory:
                memory_breakdown = memory_tracker.snapshot()

            scheduler.step()
            optimizer.zero_grad()

            if config.hv is not None:
                if int(real_step) % config.hv.local_steps == 0:
                    for param in model.parameters():
                        torch.distributed.broadcast(param.data, src=0)

            if rank == 0:
                total_samples = real_step * config.total_batch_size
                effective_step = real_step

                if config.hv is not None:
                    # Note that this assumes that we have the right amount of worker since t0.
                    # Not robust to off/on ramping
                    effective_step = real_step * config.hv.galaxy_size
                    total_samples = real_step * config.total_batch_size * config.hv.galaxy_size

                metrics = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": effective_step,  # at each step the we have compute total_batch_size. Independent of the number of GPUs
                    "total_samples": total_samples,
                    "time_taken": time.time() - current_time,
                    "tokens_per_second": config.seq_length * config.total_batch_size / (time.time() - current_time),
                }

                if memory_breakdown is not None:
                    params_gb = bytes_to_gb(memory_breakdown["parameters_bytes"])
                    grads_gb = bytes_to_gb(memory_breakdown["gradients_bytes"])
                    optim_gb = bytes_to_gb(memory_breakdown["optimizer_bytes"])
                    acts_gb = bytes_to_gb(memory_breakdown["activations_bytes"])
                    cuda_alloc_gb = bytes_to_gb(memory_breakdown["cuda_allocated_bytes"])
                    cuda_reserved_gb = bytes_to_gb(memory_breakdown["cuda_reserved_bytes"])
                    cuda_max_alloc_gb = bytes_to_gb(memory_breakdown["cuda_max_allocated_bytes"])

                    metrics["memory/parameters_gb"] = params_gb
                    metrics["memory/gradients_gb"] = grads_gb
                    metrics["memory/optimizer_gb"] = optim_gb
                    metrics["memory/activations_gb"] = acts_gb
                    metrics["memory/cuda_allocated_gb"] = cuda_alloc_gb
                    metrics["memory/cuda_reserved_gb"] = cuda_reserved_gb
                    metrics["memory/cuda_max_allocated_gb"] = cuda_max_alloc_gb

                    top_modules = memory_tracker.activation_topk()
                    if top_modules:
                        top_summary = ", ".join(
                            f"{name}: {bytes_to_gb(value):.3f}GB" for name, value in top_modules
                        )
                        log(f"Activation memory top modules (step {real_step}): {top_summary}")

                    log(
                        "Memory breakdown (step {}): params {:.3f} GB | grads {:.3f} GB | optim {:.3f} GB | activations {:.3f} GB | allocated {:.3f} GB | reserved {:.3f} GB | max {:.3f} GB".format(
                            real_step,
                            params_gb,
                            grads_gb,
                            optim_gb,
                            acts_gb,
                            cuda_alloc_gb,
                            cuda_reserved_gb,
                            cuda_max_alloc_gb,
                        )
                    )

                if world_messenger_hv:
                    outer_lr = [group["lr"] for group in optimizer.state_averager.optimizer.param_groups][0]
                    num_peers = optimizer.tracker.global_progress.num_peers

                    max_num_peers = max(max_num_peers, num_peers)

                    if num_peers == 0:
                        num_peers = 1

                    metrics["outer_lr"] = outer_lr
                    metrics["num_peers"] = num_peers

                if logging_activations_steps:
                    metrics.update(log_activations)
                    log_activations = {}

                if world_messenger_hv and num_peers < max_num_peers:
                    log(message=f"Lost a diloco worker, num_peers: {num_peers}, galaxy_size: {config.hv.galaxy_size}")
                    if config.hv.fail_rank_drop:
                        raise ValueError(
                            f"Lost a diloco worker, num_peers: {num_peers}, galaxy_size: {config.hv.galaxy_size}"
                        )

                current_time = time.time()

                metric_logger.log(metrics)

                if config.hv is None:
                    log(
                        f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in optimizer.param_groups][0]}"
                    )

            # Save checkpoint every 'checkpoint_interval' steps
            if config.ckpt.interval is not None and real_step % config.ckpt.interval == 0:
                log(f"saving at step {real_step}, step {step+1}")
                ckpt_path = os.path.join(config.ckpt.path, f"{CKPT_PREFIX}_{int(real_step)}")

                if config.hv:
                    ckpt_path = os.path.join(ckpt_path, get_diloco_rank_dir_name(config.hv.world_rank))

                if world_messenger_hv:
                    assert isinstance(optimizer, DiLoCoOptimizer)
                    with optimizer.tracker.pause_updates():
                        save_checkpoint(
                            checkpoint_path=ckpt_path,
                            model=model,
                            optimizer=optimizer.inner_optimizer,
                            scheduler=scheduler,
                            outer_optimizer=optimizer.state_averager.optimizer,
                            loss=loss_batch.item(),
                            scaler=scaler,
                            data_loader=train_dataloader,
                            save_global_state=True,
                        )
                else:
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=loss_batch.item(),
                        scaler=scaler,
                        data_loader=train_dataloader,
                        save_global_state=rank == 0,
                    )

                if local_rank == 0:
                    # only the rank 0 deletes the checkpoints
                    if config.ckpt.topk is not None:
                        ckpt_deleted = delete_old_checkpoints(config.ckpt.path, config.ckpt.topk)
                        if ckpt_deleted:
                            log(f"Deleted old checkpoints: {ckpt_deleted}")

            loss_batch = 0

            if config.max_steps is not None and real_step >= config.max_steps:
                break

    log("Training completed.")
    if rank == 0:
        metric_logger.finish()


def expand_env_vars(value: Any) -> Any:
    """재귀적으로 딕셔너리와 리스트를 순회하며 문자열의 환경 변수를 대체합니다."""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            expanded = expand_env_vars(v)
            # 특정 숫자 필드가 문자열로 남아있는 경우 변환 시도
            if k == "world_rank" and isinstance(expanded, str) and expanded.strip().isdigit():
                result[k] = int(expanded)
            elif k in ["galaxy_size", "local_steps", "averaging_timeout", "interval", 
                      "max_steps", "warmup_steps", "per_device_train_batch_size", 
                      "total_batch_size"] and isinstance(expanded, str) and expanded.strip().isdigit():
                result[k] = int(expanded)
            else:
                result[k] = expanded
        return result
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    elif isinstance(value, str):
        # 환경 변수 대체: ${VAR} 또는 $VAR 형식
        # ${VAR} 형식 대체
        value = re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), value)
        # $VAR 형식 대체 (단어 경계 확인)
        value = re.sub(r'\$(\w+)', lambda m: os.environ.get(m.group(1), m.group(0)), value)
        return value
    else:
        return value


def load_config_from_toml(toml_path: str) -> dict:
    """TOML 파일에서 설정을 로드하고 환경 변수를 대체합니다."""
    if tomllib is None:
        raise ImportError("TOML support is not available. Please install 'tomli' or 'toml' package.")
    
    with open(toml_path, "rb") as f:
        config_dict = tomllib.load(f)
    
    # 환경 변수 대체
    config_dict = expand_env_vars(config_dict)
    
    # TOML 파일의 중첩 구조를 그대로 유지
    # pydantic은 중첩된 딕셔너리를 자동으로 처리할 수 있음
    return config_dict


def merge_config_args(config_file: str | None = None) -> dict:
    """TOML 파일과 명령줄 인자를 병합합니다."""
    # 명령줄 인자에서 --config 옵션 제거
    original_argv = sys.argv.copy()
    config_file_from_argv = None
    
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file_from_argv = sys.argv[config_idx + 1]
            # --config와 그 값 제거 (parse_argv에서 처리되지 않도록)
            sys.argv.pop(config_idx + 1)
            sys.argv.pop(config_idx)
    
    # 최종 config_file 결정
    final_config_file = config_file or config_file_from_argv
    
    config_dict = {}
    
    # TOML 파일 로드
    if final_config_file:
        if not os.path.exists(final_config_file):
            raise FileNotFoundError(f"Config file not found: {final_config_file}")
        toml_config = load_config_from_toml(final_config_file)
        config_dict = toml_config
    
    # 명령줄 인자 파싱
    try:
        argv_config = parse_argv()
        
        # parse_argv()가 반환한 'config' 키는 Config 클래스에 없으므로 제거
        if 'config' in argv_config:
            del argv_config['config']
        
        # 명령줄 인자가 TOML 설정을 덮어씀
        # 하지만 중첩 구조를 고려하여 병합해야 함
        def deep_merge(base: dict, override: dict) -> dict:
            """중첩된 딕셔너리를 재귀적으로 병합"""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        if config_dict:
            config_dict = deep_merge(config_dict, argv_config)
        else:
            config_dict = argv_config
    finally:
        # sys.argv 복원
        sys.argv = original_argv
    
    return config_dict


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "PRIME_INTELLECT_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    
    # TOML 파일과 명령줄 인자 병합 (--config 옵션이 있으면 자동으로 처리)
    config_dict = merge_config_args()
    config = Config(**config_dict)
    
    # Batch size 탐색 모드인 경우, 추정 후 학습 진행
    if config.find_max_batch_size:
        max_batch_size = find_max_batch_size_for_model(config)
        if max_batch_size is None or max_batch_size <= 0:
            print("Batch size 탐색에 실패했습니다. 종료합니다.")
            destroy_process_group()
            exit(1)
        
        # 추정된 batch size를 설정에 적용
        print(f"\n추정된 batch size ({max_batch_size})를 학습에 적용합니다.")
        config.per_device_train_batch_size = max_batch_size
    
    train(config)
    destroy_process_group()
