"""
Model Parallel (TP / PP) training script for DiLoCo.

Supports:
  - Tensor Parallelism (TP) via torch.distributed.tensor.parallel
  - Pipeline Parallelism (PP) via torch.distributed.pipelining
  - Mixed training with FSDP nodes (different nodes can use different strategies)

Usage (TP, 2 GPUs):
  torchrun --nproc_per_node=2 train_mp.py --config config_mp.toml

Usage (PP, 2 GPUs):
  torchrun --nproc_per_node=2 train_mp.py --config config_mp.toml --parallelism pp
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

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            tomllib = None
            print("Warning: TOML support not available.")

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import ScheduleGPipe
from torch.distributed.pipelining.microbatch import TensorChunkSpec

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)

from ckpt_utils import (
    CKPT_PREFIX,
    CkptConfig,
    check_checkpoint_path_access,
    delete_old_checkpoints,
    get_ckpt_base_path,
    get_parameter_tracking_log_dir,
    get_diloco_rank_dir_name,
    get_resume_info,
    load_checkpoint,
    save_checkpoint,
    should_save_final_checkpoint,
)
from hivemind_diloco import AllReduceStrategy, DiLoCoOptimizer, wait_for_all_nodes_validation_complete
from open_diloco.utils import WandbLogger, DummyLogger

from hivemind.dht.dht import DHT
from hivemind.utils import get_dht_time
from hivemind.utils.networking import log_visible_maddrs
from hivemind.optim.optimizer import logger

try:
    from .memory_debugger import MemoryUsageTracker, bytes_to_gb
except ImportError:
    from memory_debugger import MemoryUsageTracker, bytes_to_gb
from open_diloco.utils import (
    FakeTokenizedDataset,
    get_compression_kwargs,
    register_metrics_hooks,
)

try:
    from llama_pp_stages import causal_lm_shifted_loss
except ImportError:
    from open_diloco.llama_pp_stages import causal_lm_shifted_loss

from model_parallel_utils import (
    wrap_model_tp,
    wrap_model_pp,
    create_full_model_copy,
    gather_tp_params,
    scatter_tp_params,
    broadcast_tp_params,
    gather_pp_params,
    scatter_pp_params,
    mp_scaler_unscale_,
    mp_clip_grad_norm_,
    mp_no_sync,
)

from dataset_splitter import (
    split_dataset_by_worker_batch_size,
    publish_node_info,
    get_node_batch_sizes_from_dht,
    read_node_info,
)


TIMEOUT_NCCL_MINUTES = os.environ.get("TIMEOUT_NCCL_MINUTES", 120)
TARGET_LAYER_ACTIVATIONS = ["self_attn", "lm_head"]
TEST_VOCAB_SIZE = 1024


def ddp_setup():
    init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


# ── Config ────────────────────────────────────────────────────────────────

class HvConfig(BaseConfig):
    outer_lr: float = 0.7
    outer_optimizer_type: Literal["nesterov", "adam"] = "nesterov"
    outer_betas: tuple[float, float] = (0.9, 0.999)
    outer_eps: float = 1e-8
    local_steps: int = 500
    initial_peers: list[str] | None = None
    host_maddrs: list[str] = ["/ip4/0.0.0.0/tcp/0"]
    announce_maddrs: list[str] | None = None
    matchmaking_time: float | None = None
    averaging_timeout: float | None = None
    average_state_every: int = 1  # state averaging 주기 (1=매 epoch, 큰 값=사실상 비활성화)
    hivemind_compression: Literal["fp16", "scaled-fp16", "uniform8bit", "quantile8bit", "blockwise8bit", "sign1bit"] | None = None
    all_reduce_strategy: AllReduceStrategy = AllReduceStrategy.WAIT_FOR_ALL
    timeout_waiting_for_peers: float | None = None
    skip_load_from_peers: bool = False
    world_rank: int
    galaxy_size: int
    fail_rank_drop: bool = False
    selective_layer_patterns: list[str] | None = None
    gradient_magnitude_threshold: float | None = None
    gradient_magnitude_top_k_ratio: float | None = None
    gradient_magnitude_top_k_ratio_by_size: bool = False
    gradient_magnitude_selection_mode: str = "layer"
    gradient_importance_metric: Literal["magnitude", "taylor"] = "magnitude"
    residual_norm_threshold: float | None = None
    token_weighted_aggregation: bool = False
    token_weight_mode: Literal["linear", "sqrt"] = "linear"
    max_outer_optimization_steps: int | None = None
    enable_max_staleness: bool = False
    max_staleness: int = 100
    enable_warmup: bool = False
    warmup_epochs: int = 5
    enable_gradient_clipping: bool = False
    gradient_clip_norm: float = 1.0
    # Outer sign update: pseudo-gradient에 sign()을 적용하여 Lion처럼 uniform magnitude로 업데이트
    outer_sign_update: bool = False
    # outer_sign_aggregation: 노드 간 집계 방식 ("majority_vote" — sign만 통신 후 token-weighted 평균)
    outer_sign_aggregation: Literal["majority_vote"] = "majority_vote"
    # outer_sign_mode: "fixed_lr" (고정 outer_lr) or "adaptive_mean" (텐서별 mean(|pseudo_grad|)를 step size로 사용)
    outer_sign_mode: Literal["fixed_lr", "adaptive_mean"] = "fixed_lr"
    # Error Feedback: sign 양자화 잔차를 다음 outer step에 보정
    outer_sign_error_feedback: bool = False
    # Outer LR scheduling (fixed_lr 모드에서 주로 사용)
    outer_lr_scheduler_type: Literal["constant", "cosine", "linear"] = "constant"
    outer_warmup_steps: int = 0  # outer optimization step 기준 warmup
    # Outer optimizer: Nesterov SGD vs plain SGD (outer_optimizer_type != "adam" 일 때)
    outer_use_nesterov_sgd: bool = True
    outer_weight_decay: float = 0.0  # decoupled weight decay for outer optimizer
    # Stats 수집 최적화: true면 norm+abs_mean만 수집 (GPU sync 최소화)
    minimal_pseudo_grad_stats: bool = False
    use_throughput_adaptive_sizing: bool = True
    @field_validator('initial_peers', mode='before')
    def _parse_str_to_str_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, list):
            return v
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
    # Optimisation
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    warmup_steps: int = 1000
    total_steps: int = 88_000
    sharding_strategy: str = "NO_SHARD"
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    # Inner optimizer selection: "adamw" or "lion"
    inner_optimizer_type: Literal["adamw", "lion"] = "adamw"
    inner_weight_decay: float = 0.1
    inner_betas: tuple[float, float] = (0.9, 0.95)
    lion_lr_scale: float = 0.3
    lion_wd_scale: float = 3.0
    lion_betas: tuple[float, float] = (0.95, 0.98)  # 언어 모델링 권장값
    # Parallelism
    parallelism: Literal["tp", "pp"] = "tp"
    # Checkpointing / logging
    project: str = "hivemind_debug"
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    log_activations_steps: int | None = None
    log_memory_breakdown_steps: int | None = None
    ckpt: CkptConfig = CkptConfig()
    # Hivemind
    hv: HvConfig | None = None
    fake_data: bool = False
    max_steps: int | None = None
    # Lora
    lora: bool | None = False
    # Batch size finder
    find_max_batch_size: bool = False
    # Validation
    validation: bool = False
    initial_lr_adjust: bool = False
    adjust_local_steps: bool = False


# ── Data ──────────────────────────────────────────────────────────────────

def get_dataloader(tokenizer, world_size, rank, local_rank, config: Config, dht=None):
    if config.fake_data:
        train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)
    else:
        ds = load_dataset(config.dataset_name_or_path, "default", streaming=True, trust_remote_code=True)

        def tokenize_function(data):
            return tokenizer(data["text"], truncation=True, max_length=config.seq_length, padding="max_length")

        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])["train"]

        if config.hv is not None and dht is not None:
            RUN_ID = "OpenDiLoCo"
            node_batch_sizes, node_gpu_counts, node_per_device_batch_sizes = get_node_batch_sizes_from_dht(
                dht=dht, run_id=RUN_ID, galaxy_size=config.hv.galaxy_size, timeout=120.0,
            )
            global_rank = sum(node_gpu_counts[:config.hv.world_rank]) + local_rank
            print(f"MY LOCAL RANK: {local_rank}, MY WORLD RANK: {config.hv.world_rank}, MY GLOBAL RANK: {global_rank}")
            train_dataset = split_dataset_by_worker_batch_size(
                tokenized_datasets,
                node_batch_sizes=node_batch_sizes, node_gpu_counts=node_gpu_counts,
                node_per_device_batch_sizes=node_per_device_batch_sizes,
                world_rank=config.hv.world_rank, local_rank=local_rank,
            )
            if local_rank == 0:
                node_info_key = f"{RUN_ID}:node_info"
                node_info = read_node_info(dht, node_info_key)
                now = get_dht_time()
                for worker_id in node_info.keys():
                    dht.store(key=node_info_key, subkey=worker_id, value=None, expiration_time=now - 1)
        else:
            rank_for_split = (rank - local_rank) if getattr(config, "parallelism", "tp") == "pp" else rank
            train_dataset = split_dataset_by_node(
                tokenized_datasets, world_size=world_size, rank=rank_for_split
            )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return StatefulDataLoader(train_dataset, collate_fn=data_collator, batch_size=config.per_device_train_batch_size, num_workers=config.num_workers)


def get_model(config: Config) -> LlamaForCausalLM:
    config_model = LlamaConfig.from_pretrained(
        config.path_model,
        attn_implementation=config.attn_implementation,
        resume_download=True,
    )
    return LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.path_model,
        config=config_model,
        resume_download=True,
    )


# ── helpers (copied from train_fsdp.py for DHT speed/readiness) ──────────

def unwrap(v):
    return getattr(v, "value", v)


def read_speeds(dht, key):
    res = dht.get(key, latest=True)
    root = unwrap(res) if res else None
    speeds = {}
    if isinstance(root, dict):
        for k, v in root.items():
            p = unwrap(v)
            if isinstance(p, dict):
                if "steps_per_sec" in p:
                    speeds[k] = float(p["steps_per_sec"])
                elif "v" in p:
                    speeds[k] = float(p["v"])
                elif "step_time_s" in p and p["step_time_s"] > 0:
                    speeds[k] = 1.0 / float(p["step_time_s"])
    return speeds


def wait_for_all_nodes_ready(dht, galaxy_size, config_hv, log_fn):
    log_fn("Waiting for all nodes to be ready ...")
    RUN_ID = "OpenDiLoCo"
    ready_key = f"{RUN_ID}:ready"
    worker_id = f"{socket.gethostname()}-pid{os.getpid()}"
    now = get_dht_time()
    exp = now + config_hv.averaging_timeout
    dht.store(key=ready_key, subkey=worker_id, value={"ready": True, "ts": now, "host": socket.gethostname()}, expiration_time=exp)
    while True:
        ready_res = dht.get(ready_key, latest=True)
        root = unwrap(ready_res) if ready_res else None
        count = 0
        if isinstance(root, dict):
            for v in root.values():
                p = unwrap(v)
                if isinstance(p, dict) and p.get("ready"):
                    count += 1
        if count >= galaxy_size:
            break
        time.sleep(2.0)
    now = get_dht_time()
    dht.store(key=ready_key, subkey=worker_id, value={"ready": True, "ts": now}, expiration_time=now + 5.0)
    log_fn("All nodes ready.")


# ── Main training function ────────────────────────────────────────────────

def train(config: Config):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    world_messenger_hv = config.hv is not None and local_rank == 0

    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size
    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    resume_from_ckpt, resume_path = get_resume_info(config)

    if rank == 0:
        logger_cls = WandbLogger if config.metric_logger_type == "wandb" else DummyLogger
        metric_logger = logger_cls(project=config.project, config=config.model_dump(), resume=resume_from_ckpt)

    # ── DHT ───────────────────────────────────────────────────────────────
    dht = None
    if config.hv is not None:
        log("hivemind diloco enabled")
        dht = DHT(
            start=True,
            initial_peers=config.hv.initial_peers,
            host_maddrs=config.hv.host_maddrs,
            announce_maddrs=config.hv.announce_maddrs,
        )
        if world_messenger_hv:
            log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

    if config.hv is not None and not config.adjust_local_steps:
        local_steps_tensor = torch.tensor([config.hv.local_steps], dtype=torch.int32, device='cuda')
        torch.distributed.broadcast(local_steps_tensor, src=0)
        config.hv.local_steps = int(local_steps_tensor.item())

    if config.hv is not None and local_rank == 0:
        RUN_ID = "OpenDiLoCo"
        node_info_key = f"{RUN_ID}:node_info"
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        worker_id = f"{socket.gethostname()}-world_rank{config.hv.world_rank}"
        publish_node_info(dht=dht, key=node_info_key, worker_id=worker_id,
                          world_rank=config.hv.world_rank, gpu_count=local_world_size,
                          per_device_batch_size=config.per_device_train_batch_size, ttl=300.0)

    if local_rank == 0:
        check_checkpoint_path_access(get_ckpt_base_path(config), rank, config.hv.world_rank if config.hv else None)

    # ── DataLoader ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True, resume_download=True)
    tokenizer.pad_token = "</s>"
    train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config, dht=dht)

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(config)
    model = model.to(local_rank)

    half_precision = config.precision in ("fp16-mixed", "bf16-mixed")
    half_precision_dtype = torch.bfloat16 if config.precision == "bf16-mixed" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(config.precision == "fp16-mixed" and config.parallelism != "pp")
    )
    if config.parallelism == "pp" and config.precision == "fp16-mixed" and rank == 0:
        log(
            "PP: fp16-mixed runs pipeline backward without GradScaler; prefer bf16-mixed for stability."
        )

    # ── full_model for DiLoCo outer optimizer (CPU, rank 0 only) ──────────
    full_model = None
    pp_stage = None
    pp_stage_index = None
    pp_schedule = None
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    if config.parallelism == "tp":
        device_mesh = init_device_mesh("cuda", (local_world_size,), mesh_dim_names=("tp",))
        if world_messenger_hv:
            full_model = create_full_model_copy(model, device="cpu")
        wrap_model_tp(model, device_mesh)
        log(f"Model wrapped with TP (mesh size={local_world_size})")

    elif config.parallelism == "pp":
        if world_size != local_world_size:
            raise RuntimeError(
                "parallelism='pp' requires WORLD_SIZE == LOCAL_WORLD_SIZE (pipeline stages map 1:1 to local GPUs)."
            )
        pp_stage_index = local_rank
        num_stages = local_world_size
        if world_messenger_hv:
            full_model = create_full_model_copy(model, device="cpu")
        model, pp_stage = wrap_model_pp(
            model,
            num_stages,
            pp_stage_index,
            torch.device(f"cuda:{local_rank}"),
            microbatch_size=config.per_device_train_batch_size,
            seq_len=config.seq_length,
            activation_dtype=half_precision_dtype if half_precision else torch.float32,
        )
        pp_schedule = ScheduleGPipe(
            pp_stage,
            n_microbatches=1,
            loss_fn=lambda logits, labels: causal_lm_shifted_loss(logits, labels),
            kwargs_chunk_spec=TensorChunkSpec.from_dict({"input_ids": 0, "attention_mask": 0}),
        )
        log(f"Model wrapped with PP (stage {pp_stage_index}/{num_stages}, ScheduleGPipe)")
        device_mesh = None
    else:
        device_mesh = None

    # ── Speed profiling (adjust_local_steps) ────────────────────────────
    if config.hv is not None and config.adjust_local_steps:
        from speed_profiler import measure_steps_per_second_with_model

        original_local_steps = config.hv.local_steps
        if rank == 0:
            log(f"[Speed Profiling] Benchmarking TP/PP model (original local_steps={original_local_steps})")

        saved_param_data = [p.data.detach().cpu().clone() for p in model.parameters()]

        _inner_model = model.module if hasattr(model, "module") else model
        _vocab_size = getattr(getattr(_inner_model, "config", None), "vocab_size", 32000)

        profiling_start_time = time.time()
        steps_per_sec = measure_steps_per_second_with_model(
            model=model,
            batch_size=config.per_device_train_batch_size,
            seq_length=config.seq_length,
            precision=config.precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            vocab_size=_vocab_size,
            pp_schedule=pp_schedule,
            local_world_size=local_world_size,
        )
        profiling_elapsed_time = time.time() - profiling_start_time
        if rank == 0:
            log(f"[Speed Profiling] Completed in {profiling_elapsed_time:.2f}s: {steps_per_sec:.4f} steps/sec")

        with torch.no_grad():
            for param, saved in zip(model.parameters(), saved_param_data):
                param.data.copy_(saved.to(param.device))
        del saved_param_data
        torch.cuda.empty_cache()

        if world_messenger_hv:
            PUBLISH_INTERVAL = 10.0
            TTL = 30.0
            RUN_ID = "OpenDiLoCo"
            key = f"{RUN_ID}:speed"
            gpu_name = torch.cuda.get_device_name(local_rank)
            worker_id = f"{socket.gethostname()}-pid{os.getpid()}-gpu{local_rank}"

            while True:
                now = get_dht_time()
                payload = {
                    "steps_per_sec": float(steps_per_sec),
                    "ts": now,
                    "host": socket.gethostname(),
                    "gpu_id": local_rank,
                    "gpu_name": gpu_name,
                }
                exp = now + TTL
                ok = dht.store(key=key, subkey=worker_id, value=payload, expiration_time=exp)
                print(f"[publish] {worker_id} ({gpu_name}): {steps_per_sec:.2f} steps/sec ({'ok' if ok else 'fail'})")

                speeds = read_speeds(dht, key)
                if not speeds:
                    print(f"[error] no usable speeds at '{key}'. Is the publisher running?")
                    return
                if len(speeds) < config.hv.galaxy_size:
                    print(len(speeds), "speeds found, waiting for", config.hv.galaxy_size)
                    time_in_cycle = now % PUBLISH_INTERVAL
                    sleep_duration = PUBLISH_INTERVAL - time_in_cycle
                    print(f"[sync] Waiting {sleep_duration:.2f}s until next {PUBLISH_INTERVAL}s boundary")
                    time.sleep(sleep_duration)
                else:
                    break

            total_work = config.hv.galaxy_size * original_local_steps
            print(f"[INFO] Total work per epoch: {config.hv.galaxy_size} nodes × {original_local_steps} steps = {total_work} steps")

            sorted_speeds = sorted(speeds.items(), key=lambda x: x[1], reverse=True)
            total_speed = sum(speeds.values())
            print(f"[INFO] Total speed across all nodes: {total_speed:.2f} steps/sec")

            allocations = {}
            allocated_sum = 0
            for i, (wid, speed) in enumerate(sorted_speeds):
                speed_ratio = speed / total_speed
                if i < len(sorted_speeds) - 1:
                    allocated_steps = int(math.floor(total_work * speed_ratio))
                    allocated_steps = max(1, allocated_steps)
                else:
                    allocated_steps = total_work - allocated_sum
                    allocated_steps = max(1, allocated_steps)
                allocations[wid] = allocated_steps
                allocated_sum += allocated_steps

            config.hv.local_steps = allocations[worker_id]

            print(f"[INFO] Local steps allocation verification:")
            print(f"[INFO]   - Total work: {total_work}")
            print(f"[INFO]   - Sum of allocations: {allocated_sum}")
            print(f"[INFO]   - Match: {allocated_sum == total_work}")

            speed_ratio = steps_per_sec / total_speed
            print(f"[INFO] Speed distribution-based local_steps allocation:")
            print(f"[INFO]   - Current node speed: {steps_per_sec:.2f} steps/sec ({speed_ratio*100:.2f}%)")
            print(f"[INFO]   - Allocated local_steps: {config.hv.local_steps}")
            print(f"[INFO]   - Actual contribution: {config.hv.local_steps / total_work * 100:.2f}% of total work")
            print(f"[DEBUG] batch_size={batch_size}, target_batch_size will be={batch_size * config.hv.local_steps}")

            now = get_dht_time()
            print(f"[INFO] Cleaning up speed data from DHT...")
            for wid in speeds.keys():
                dht.store(key=key, subkey=wid, value=None, expiration_time=now - 1)
            print(f"[INFO] Deleted {len(speeds)} speed entries from DHT")

        local_steps_tensor = torch.tensor([config.hv.local_steps], dtype=torch.int32, device="cuda")
        torch.distributed.broadcast(local_steps_tensor, src=0)
        config.hv.local_steps = int(local_steps_tensor.item())
        print(f"[DEBUG] rank={rank}, local_rank={local_rank}: adjusted local_steps={config.hv.local_steps}")

    if config.torch_compile:
        model = torch.compile(model)

    # ── Lion auto-scaling ──────────────────────────────────────────────────
    if config.inner_optimizer_type == "lion":
        original_lr = config.lr
        original_wd = config.inner_weight_decay
        config.lr = config.lr * config.lion_lr_scale
        config.inner_weight_decay = config.inner_weight_decay * config.lion_wd_scale
        config.inner_betas = config.lion_betas
        if rank == 0:
            log(f"Lion auto-scaling applied:")
            log(f"  LR: {original_lr:.2e} * {config.lion_lr_scale} = {config.lr:.2e}")
            log(f"  Weight Decay: {original_wd:.2e} * {config.lion_wd_scale} = {config.inner_weight_decay:.2e}")
            log(f"  Betas: {config.inner_betas}")

    # ── Optimizers ────────────────────────────────────────────────────────
    if config.inner_optimizer_type == "lion":
        from lion_optimizer import Lion
        inner_optimizer_factory = partial(Lion, lr=config.lr, weight_decay=config.inner_weight_decay, betas=config.inner_betas)
        log(f"Using Lion inner optimizer (lr={config.lr}, wd={config.inner_weight_decay}, betas={config.inner_betas})")
    else:
        inner_optimizer_factory = partial(torch.optim.AdamW, lr=config.lr, weight_decay=config.inner_weight_decay, betas=config.inner_betas)
        log(f"Using AdamW inner optimizer (lr={config.lr}, wd={config.inner_weight_decay}, betas={config.inner_betas})")

    if config.hv is not None:
        if config.hv.outer_optimizer_type == "adam":
            outer_optimizer = partial(
                torch.optim.Adam, lr=config.hv.outer_lr, betas=config.hv.outer_betas, eps=config.hv.outer_eps, weight_decay=0
            )
            log(f"Using Adam outer optimizer (lr={config.hv.outer_lr}, betas={config.hv.outer_betas}, eps={config.hv.outer_eps})")
        elif config.hv.outer_use_nesterov_sgd:
            outer_optimizer = partial(
                torch.optim.SGD,
                lr=config.hv.outer_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=config.hv.outer_weight_decay,
            )
            log(
                f"Using Nesterov SGD outer optimizer (lr={config.hv.outer_lr}, momentum=0.9, wd={config.hv.outer_weight_decay})"
            )
        else:
            outer_optimizer = partial(
                torch.optim.SGD,
                lr=config.hv.outer_lr,
                momentum=0,
                nesterov=False,
                weight_decay=config.hv.outer_weight_decay,
            )
            log(f"Using plain SGD outer optimizer (lr={config.hv.outer_lr}, wd={config.hv.outer_weight_decay})")
        if config.hv.average_state_every > 1:
            log(
                f"State averaging: every {config.hv.average_state_every} epochs "
                f"({'DISABLED' if config.hv.average_state_every > 10000 else 'reduced'})"
            )
        else:
            log("State averaging: every epoch (default)")
        if config.hv.outer_sign_update:
            log(
                f"Outer sign update ENABLED [aggregation={config.hv.outer_sign_aggregation}, "
                f"mode={config.hv.outer_sign_mode}, "
                f"error_feedback={'ON' if config.hv.outer_sign_error_feedback else 'OFF'}]"
            )
            if config.hv.outer_sign_mode == "adaptive_mean":
                log(f"  adaptive_mean: step_size = mean(|pseudo_grad|) per tensor, outer_lr={config.hv.outer_lr} as global multiplier")
            else:
                log(f"  fixed_lr: step_size = outer_lr={config.hv.outer_lr}")
            if config.hv.hivemind_compression is None:
                config.hv.hivemind_compression = "sign1bit"
                log("  Auto-setting hivemind_compression='sign1bit' for 1-bit communication")
            elif config.hv.hivemind_compression == "sign1bit":
                log("  hivemind_compression='sign1bit': 1-bit packing enabled")
            elif config.hv.hivemind_compression is not None:
                log(f"  hivemind_compression='{config.hv.hivemind_compression}'")
            if config.hv.outer_sign_mode == "fixed_lr":
                if config.hv.outer_lr_scheduler_type != "constant" or config.hv.outer_warmup_steps > 0:
                    log(
                        f"  Outer LR scheduling: type={config.hv.outer_lr_scheduler_type}, "
                        f"warmup={config.hv.outer_warmup_steps}, max_steps={config.hv.max_outer_optimization_steps}"
                    )

    def scheduler_fn(opt):
        return get_cosine_schedule_with_warmup(opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_steps)

    if config.hv is not None:
        if resume_from_ckpt:
            fake_optimizer = inner_optimizer_factory(model.parameters())
            load_checkpoint(
                checkpoint_path=os.path.join(resume_path, get_diloco_rank_dir_name(config.hv.world_rank)),
                model=model, optimizer=fake_optimizer, dataset=config.dataset_name_or_path,
            )
            del fake_optimizer

    if resume_from_ckpt and config.hv is not None:
        ckpt_path = os.path.join(resume_path, get_diloco_rank_dir_name(config.hv.world_rank))
    elif resume_from_ckpt:
        ckpt_path = resume_path

    if world_messenger_hv:
        assert full_model is not None, "full_model must exist on world_messenger for DiLoCo"

        # Sync full_model with current model state before creating DiLoCoOptimizer
        if config.parallelism == "tp":
            gather_tp_params(model, full_model)
        elif config.parallelism == "pp":
            gather_pp_params(model, full_model, pp_stage_index, local_world_size)

        # Pre-create inner optimizer with TP/PP model parameters
        inner_opt = inner_optimizer_factory(model.parameters())

        # Extract param names from the full model (not TP/PP wrapped)
        param_names = None
        needs_param_names = (
            config.hv.selective_layer_patterns is not None or
            config.hv.gradient_magnitude_threshold is not None or
            config.hv.gradient_magnitude_top_k_ratio is not None
        )
        if needs_param_names:
            param_names = [name for name, _ in full_model.named_parameters()]
            log(f"Extracted {len(param_names)} parameter names from full_model for selective layer update")

        diloco_args = dict(
            dht=dht,
            run_id="llama",
            batch_size=batch_size,
            num_inner_steps=config.hv.local_steps,
            outer_optimizer=outer_optimizer,
            inner_optimizer=inner_opt,
            scheduler=None,
            params=full_model.parameters(),
            delay_optimizer_step=False,
            delay_grad_averaging=False,
            verbose=True,
            all_reduce_strategy=config.hv.all_reduce_strategy,
            timeout_waiting_for_peers=config.hv.timeout_waiting_for_peers,
            selective_layer_patterns=config.hv.selective_layer_patterns,
            gradient_magnitude_threshold=config.hv.gradient_magnitude_threshold,
            gradient_magnitude_top_k_ratio=config.hv.gradient_magnitude_top_k_ratio,
            gradient_magnitude_top_k_ratio_by_size=config.hv.gradient_magnitude_top_k_ratio_by_size,
            gradient_magnitude_selection_mode=config.hv.gradient_magnitude_selection_mode,
            gradient_importance_metric=config.hv.gradient_importance_metric,
            param_names=param_names,
            token_weighted_aggregation=config.hv.token_weighted_aggregation,
            token_weight_mode=config.hv.token_weight_mode,
            residual_norm_threshold=config.hv.residual_norm_threshold,
            enable_max_staleness=config.hv.enable_max_staleness,
            max_staleness=config.hv.max_staleness,
            enable_warmup=config.hv.enable_warmup,
            warmup_epochs=config.hv.warmup_epochs,
            enable_gradient_clipping=config.hv.enable_gradient_clipping,
            gradient_clip_norm=config.hv.gradient_clip_norm,
            galaxy_size=config.hv.galaxy_size,
            outer_sign_update=config.hv.outer_sign_update,
            outer_sign_aggregation=config.hv.outer_sign_aggregation,
            outer_sign_mode=config.hv.outer_sign_mode,
            outer_sign_error_feedback=config.hv.outer_sign_error_feedback,
            outer_lr_scheduler_type=config.hv.outer_lr_scheduler_type,
            outer_warmup_steps=config.hv.outer_warmup_steps,
            max_outer_optimization_steps=config.hv.max_outer_optimization_steps,
            minimal_pseudo_grad_stats=config.hv.minimal_pseudo_grad_stats,
        )
        diloco_args.update(get_compression_kwargs(config.hv.hivemind_compression))
        if "averager_opts" not in diloco_args:
            diloco_args["averager_opts"] = {}
        diloco_args["averager_opts"]["use_throughput_adaptive_sizing"] = config.hv.use_throughput_adaptive_sizing
        if config.hv.averaging_timeout is not None:
            diloco_args["averaging_timeout"] = config.hv.averaging_timeout
        diloco_args["average_state_every"] = config.hv.average_state_every
        if config.hv.matchmaking_time is not None:
            diloco_args["matchmaking_time"] = config.hv.matchmaking_time

        diloco_args["log_dir"] = get_parameter_tracking_log_dir(config)
        optimizer = DiLoCoOptimizer(**diloco_args)
        optimizer._world_rank = config.hv.world_rank

        # ── Register gather/scatter hooks ─────────────────────────────────
        if config.parallelism == "tp":
            optimizer.pre_outer_step_hook = lambda: gather_tp_params(model, full_model)
            optimizer.post_outer_step_hook = lambda: scatter_tp_params(full_model, model)
        elif config.parallelism == "pp":
            optimizer.pre_outer_step_hook = lambda: gather_pp_params(model, full_model, pp_stage_index, local_world_size)
            optimizer.post_outer_step_hook = lambda: scatter_pp_params(full_model, model, pp_stage_index, local_world_size)

        scheduler = scheduler_fn(optimizer.inner_optimizer)

        if resume_from_ckpt:
            load_checkpoint(
                checkpoint_path=ckpt_path, model=model,
                optimizer=optimizer.inner_optimizer, scheduler=scheduler,
                outer_optimizer=optimizer.state_averager.optimizer,
                scaler=scaler, data_loader=train_dataloader,
                dataset=config.dataset_name_or_path,
            )
            optimizer.update_num_inner_steps(config.hv.local_steps)
            start_step = scheduler.last_epoch
        else:
            start_step = 0

    else:
        optimizer = inner_optimizer_factory(model.parameters())
        scheduler = scheduler_fn(optimizer)
        if resume_from_ckpt:
            load_checkpoint(
                checkpoint_path=ckpt_path, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, data_loader=train_dataloader,
                dataset=config.dataset_name_or_path,
            )
            start_step = scheduler.last_epoch
        else:
            start_step = 0

    if config.hv is not None:
        torch.distributed.barrier()
        if local_rank == 0:
            log("All ranks synchronized after optimizer construction.")

    if resume_from_ckpt:
        log(f"Resumed from checkpoint at step {start_step}")

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
    all_nodes_ready = False

    # ── Training loop ─────────────────────────────────────────────────────
    real_step = start_step  # used after loop for final checkpoint if loop body never runs
    for step, batch in enumerate(iterable=train_dataloader, start=start_step * gradient_accumulation_steps):
        real_step = (step + 1) // gradient_accumulation_steps
        is_accumulating = bool((step + 1) % gradient_accumulation_steps)

        logging_activations_steps = (
            config.log_activations_steps is not None and real_step % config.log_activations_steps == 0
        )

        if logging_activations_steps:
            handles = register_metrics_hooks(model, TARGET_LAYER_ACTIVATIONS, log_activations, gradient_accumulation_steps)

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        with mp_no_sync(model) if is_accumulating else nullcontext():
            if config.parallelism == "pp":
                assert pp_schedule is not None
                losses_list: list = []
                with torch.autocast(device_type="cuda", dtype=half_precision_dtype) if half_precision else nullcontext():
                    pp_schedule.step(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        target=batch["labels"],
                        losses=losses_list,
                    )
                loss = torch.zeros((), device=f"cuda:{local_rank}", dtype=torch.float32)
                if local_rank == local_world_size - 1 and losses_list:
                    loss = (losses_list[0] / gradient_accumulation_steps).detach()
                torch.distributed.broadcast(loss, src=local_world_size - 1)
                loss_batch += loss.detach()

                if (world_messenger_hv and not is_accumulating
                        and config.hv is not None and config.hv.token_weighted_aggregation):
                    if 'attention_mask' in batch:
                        batch_token_count = batch['attention_mask'].sum().item()
                    elif 'input_ids' in batch:
                        batch_token_count = batch['input_ids'].numel()
                    else:
                        batch_token_count = config.seq_length * config.per_device_train_batch_size
                    if hasattr(optimizer, 'diloco_grad_averager'):
                        optimizer.diloco_grad_averager.accumulate_tokens(batch_token_count)
            else:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss_batch += loss.detach()

                if (world_messenger_hv and not is_accumulating
                        and config.hv is not None and config.hv.token_weighted_aggregation):
                    if 'attention_mask' in batch:
                        batch_token_count = batch['attention_mask'].sum().item()
                    elif 'input_ids' in batch:
                        batch_token_count = batch['input_ids'].numel()
                    else:
                        batch_token_count = config.seq_length * config.per_device_train_batch_size
                    if hasattr(optimizer, 'diloco_grad_averager'):
                        optimizer.diloco_grad_averager.accumulate_tokens(batch_token_count)

                scaler.scale(loss).backward()

        # All ranks must sync after DHT readiness (see train_mp_jr PP deadlock note).
        if config.hv is not None and not all_nodes_ready:
            if local_rank == 0:
                wait_for_all_nodes_ready(dht, config.hv.galaxy_size, config.hv, log)
            torch.distributed.barrier()
            all_nodes_ready = True

        if logging_activations_steps:
            for handle in handles:
                handle.remove()

        if not is_accumulating:
            if config.parallelism == "pp":
                mp_clip_grad_norm_(model, 1.0)
                if world_messenger_hv:
                    optimizer.step(scaler=None, current_loss=float(loss.detach().item()))
                else:
                    optimizer.step()
            else:
                if world_messenger_hv:
                    mp_scaler_unscale_(scaler, optimizer.inner_optimizer, model)
                else:
                    mp_scaler_unscale_(scaler, optimizer, model)

                mp_clip_grad_norm_(model, 1.0)

                if world_messenger_hv:
                    optimizer.step(scaler=scaler, current_loss=float(loss.detach().item()))
                else:
                    scaler.step(optimizer)

                scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if config.hv is not None and config.parallelism in ("pp", "tp"):
                torch.distributed.barrier()

            # Sync parameters to all TP/PP ranks after outer step
            if config.hv is not None and int(real_step) % config.hv.local_steps == 0:
                if config.parallelism == "tp":
                    broadcast_tp_params(model, full_model, src_rank=0)
                elif config.parallelism == "pp":
                    for param in model.parameters():
                        torch.distributed.broadcast(param.data, src=0)

            if rank == 0:
                total_samples = real_step * config.total_batch_size
                effective_step = real_step
                if config.hv is not None:
                    effective_step = real_step * config.hv.galaxy_size
                    total_samples = real_step * config.total_batch_size * config.hv.galaxy_size

                metrics = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": effective_step,
                    "total_samples": total_samples,
                    "time_taken": time.time() - current_time,
                    "tokens_per_second": config.seq_length * config.total_batch_size / (time.time() - current_time),
                }

                if world_messenger_hv:
                    outer_lr = [group["lr"] for group in optimizer.state_averager.optimizer.param_groups][0]
                    num_peers = optimizer.tracker.global_progress.num_peers
                    max_num_peers = max(max_num_peers, num_peers)
                    if num_peers == 0:
                        num_peers = 1
                    metrics["outer_lr"] = outer_lr
                    metrics["num_peers"] = num_peers
                    metrics["outer_optimization_steps"] = optimizer.tracker.local_progress.epoch

                if logging_activations_steps:
                    metrics.update(log_activations)
                    log_activations = {}

                if world_messenger_hv and num_peers < max_num_peers:
                    log(f"Lost a diloco worker, num_peers: {num_peers}, galaxy_size: {config.hv.galaxy_size}")
                    if config.hv.fail_rank_drop:
                        raise ValueError(f"Lost a diloco worker")

                current_time = time.time()
                metric_logger.log(metrics)

                if config.hv is None:
                    log(f"step: {real_step}, loss: {loss_batch.item()}")

            # Checkpoint
            if config.ckpt.interval is not None and real_step % config.ckpt.interval == 0:
                log(f"saving at step {real_step}")
                ckpt_save_path = os.path.join(get_ckpt_base_path(config), f"{CKPT_PREFIX}_{int(real_step)}")
                if config.hv:
                    ckpt_save_path = os.path.join(ckpt_save_path, get_diloco_rank_dir_name(config.hv.world_rank))
                if world_messenger_hv:
                    assert isinstance(optimizer, DiLoCoOptimizer)
                    with optimizer.tracker.pause_updates():
                        save_checkpoint(
                            checkpoint_path=ckpt_save_path, model=model,
                            optimizer=optimizer.inner_optimizer, scheduler=scheduler,
                            outer_optimizer=optimizer.state_averager.optimizer,
                            loss=loss_batch.item(), scaler=scaler,
                            data_loader=train_dataloader, save_global_state=True,
                        )
                else:
                    save_checkpoint(
                        checkpoint_path=ckpt_save_path, model=model,
                        optimizer=optimizer, scheduler=scheduler,
                        loss=loss_batch.item(), scaler=scaler,
                        data_loader=train_dataloader, save_global_state=rank == 0,
                    )
                if local_rank == 0 and config.ckpt.topk is not None:
                    deleted = delete_old_checkpoints(get_ckpt_base_path(config), config.ckpt.topk)
                    if deleted:
                        log(f"Deleted old checkpoints: {deleted}")

            loss_batch = 0

            # All ranks must agree to break: only world_messenger has tracker; without sync, rank0
            # exits alone and dcp.save deadlocks waiting for other local ranks.
            if config.hv is not None and config.hv.max_outer_optimization_steps is not None:
                stop_outer = torch.zeros(1, dtype=torch.int32, device=f"cuda:{local_rank}")
                if world_messenger_hv and optimizer.tracker.local_progress.epoch >= config.hv.max_outer_optimization_steps:
                    stop_outer.fill_(1)
                torch.distributed.all_reduce(stop_outer, op=torch.distributed.ReduceOp.MAX)
                if stop_outer.item() != 0:
                    if rank == 0:
                        log(f"Reached max outer optimization steps ({config.hv.max_outer_optimization_steps}). Stopping.")
                    break

            if config.max_steps is not None and real_step >= config.max_steps:
                break

    # Final checkpoint if last step was not already saved at interval
    if should_save_final_checkpoint(real_step, config.ckpt.interval):
        log(f"saving final checkpoint at step {real_step}")
        ckpt_save_path = os.path.join(get_ckpt_base_path(config), f"{CKPT_PREFIX}_{int(real_step)}")
        if config.hv:
            ckpt_save_path = os.path.join(ckpt_save_path, get_diloco_rank_dir_name(config.hv.world_rank))
        if world_messenger_hv:
            assert isinstance(optimizer, DiLoCoOptimizer)
            with optimizer.tracker.pause_updates():
                save_checkpoint(
                    checkpoint_path=ckpt_save_path,
                    model=model,
                    optimizer=optimizer.inner_optimizer,
                    scheduler=scheduler,
                    outer_optimizer=optimizer.state_averager.optimizer,
                    loss=0.0,
                    scaler=scaler,
                    data_loader=train_dataloader,
                    save_global_state=True,
                )
        else:
            save_checkpoint(
                checkpoint_path=ckpt_save_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=0.0,
                scaler=scaler,
                data_loader=train_dataloader,
                save_global_state=rank == 0,
            )
        if local_rank == 0 and config.ckpt.topk is not None:
            deleted = delete_old_checkpoints(get_ckpt_base_path(config), config.ckpt.topk)
            if deleted:
                log(f"Deleted old checkpoints: {deleted}")

    log("Training completed.")
    if rank == 0:
        metric_logger.finish()


# ── TOML / CLI config parsing (reused from train_fsdp.py) ────────────────

def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            expanded = expand_env_vars(v)
            if k in ("world_rank", "galaxy_size", "local_steps", "averaging_timeout",
                      "interval", "max_steps", "warmup_steps", "per_device_train_batch_size",
                      "total_batch_size") and isinstance(expanded, str) and expanded.strip().isdigit():
                result[k] = int(expanded)
            else:
                result[k] = expanded
        return result
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    elif isinstance(value, str):
        value = re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), value)
        value = re.sub(r'\$(\w+)', lambda m: os.environ.get(m.group(1), m.group(0)), value)
        return value
    return value


def load_config_from_toml(toml_path: str) -> dict:
    if tomllib is None:
        raise ImportError("TOML support not available")
    with open(toml_path, "rb") as f:
        config_dict = tomllib.load(f)
    return expand_env_vars(config_dict)


def merge_config_args(config_file=None):
    original_argv = sys.argv.copy()
    config_file_from_argv = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_file_from_argv = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
    final = config_file or config_file_from_argv
    config_dict = {}
    if final:
        if not os.path.exists(final):
            raise FileNotFoundError(f"Config file not found: {final}")
        config_dict = load_config_from_toml(final)
    try:
        argv_config = parse_argv()
        if 'config' in argv_config:
            del argv_config['config']

        def deep_merge(base, override):
            result = base.copy()
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = deep_merge(result[k], v)
                else:
                    result[k] = v
            return result

        config_dict = deep_merge(config_dict, argv_config) if config_dict else argv_config
    finally:
        sys.argv = original_argv
    return config_dict


if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = "PRIME_INTELLECT_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    config_dict = merge_config_args()
    config = Config(**config_dict)
    train(config)
    destroy_process_group()
