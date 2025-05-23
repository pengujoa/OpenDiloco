"""

to test quickly do 
torchrun --nproc_per_node=2 \
        train_fsdp.py --per-device-train-batch-size 8 --total-batch-size 128 --lr 1e-2 --path-model ../tests/models/llama-2m-fresh \
        --no-torch-compile --log-activations-steps 5 --fake-data --max-steps 20
"""

from functools import partial
import os
import time
from contextlib import nullcontext
import datetime
from typing import Any, Literal

from pydantic import model_validator, validator, Field
import torch
from pydantic_config import parse_argv, BaseConfig
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
from hivemind.utils.networking import log_visible_maddrs
from hivemind.optim.optimizer import logger


from open_diloco.utils import (
    FakeTokenizedDataset,
    get_compression_kwargs,
    get_sharding_strategy,
    register_metrics_hooks,
)

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

    @model_validator(mode="before")
    def cast_str_to_list(cls, values: dict[str, Any]) -> dict[str, Any]:
        """This allow to only pass a string and it will still be cast as a list"""
        for arg_name in ["initial_peers", "host_maddrs", "announce_maddrs"]:
            if arg_name in values.keys() and isinstance(values[arg_name], str):
                values[arg_name] = [values[arg_name]]
        return values


class Config(BaseConfig):
    path_model: str = "PrimeIntellect/llama-150m-fresh"
    torch_compile: bool = True
    attn_implementation: str = "sdpa"
    # Data
    # dataset_name_or_path: str = "allenai/c4"
    dataset_name_or_path: str = "qiaojin/PubMedQA"
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
    ckpt: CkptConfig = CkptConfig()
    # Hivemind
    hv: HvConfig | None = None  # if no hv config then hivemind is disabled
    fake_data: bool = False
    max_steps: int | None = None
    # Node-specific GPU configuration
    node_gpu_counts: list[int] = Field(..., description="List of GPU counts per node. Must be provided.") # List to store GPU counts per node
    # Lora
    lora: bool | None = False

    @validator("node_gpu_counts",  pre=True)
    def validate_node_gpu_counts(cls, value, values):
        if isinstance(value, str):
            try:
                value = [int(item) for item in value.split(",")]  
            except ValueError:
                raise ValueError("All elements in node_gpu_counts must be integers.")
            
        hv = values.get('hv')

        if len(value) > 1 and hv is None:
            raise ValueError("hv configuration must be provided for multi-node training.")

        if hv and len(value) != hv.galaxy_size:
            raise ValueError("Number of nodes must be equal to hv.galaxy_size.")

        if not all(isinstance(count, int) and count > 0 for count in value):
            raise ValueError("All node GPU counts must be positive integers.")

        return value
    

def get_dataloader(tokenizer, world_size, rank, local_rank, config: Config) -> StatefulDataLoader:
    if config.fake_data:
        train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)
    else:
        # ds = load_dataset(config.dataset_name_or_path, "en", streaming=True)
        dataset_dict = load_dataset(config.dataset_name_or_path, "pqa_artificial", streaming=True)
        print(dataset_dict.keys())
        ds = dataset_dict["train"]

        # def tokenize_function(data):
        #     outputs = tokenizer(
        #         data["text"],
        #         truncation=True,
        #         max_length=config.seq_length,
        #         padding="max_length",
        #     )
        #     return outputs
        label_mapping = {"yes": 0, "no": 1, "maybe": 2}

        def tokenize_function(data):
            inputs = [f"Question: {q} Context: {c}" for q, c in zip(data["question"], data["context"])]
            outputs = tokenizer(
                inputs,
                truncation=True,
                max_length=config.seq_length,
                padding="max_length",
            )
            # outputs["labels"] = tokenizer( 
            #     data["final_decision"],  # `final_decision`은 레이블 필드
            #     truncation=True,
            #     max_length=config.seq_length,
            #     padding="max_length",
            # )["input_ids"]
            outputs["labels"] = [label_mapping[ans] for ans in data["final_decision"]]  # 변환된 값 사용

            return outputs


        # tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])[
        #     "train"
        # ]
        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["question", "context", "long_answer", "final_decision"])

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
    config_model = LlamaConfig.from_pretrained(config.path_model, attn_implementation=config.attn_implementation)
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=config.path_model, config=config_model)
    if config.lora:
        print_trainable_parameters(model)
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
    return model


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
        dht = DHT(
            start=True,
            initial_peers=config.hv.initial_peers,
            host_maddrs=config.hv.host_maddrs,
            announce_maddrs=config.hv.announce_maddrs,
        )
        log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

    if local_rank == 0:
        check_checkpoint_path_access(config.ckpt.path, rank, config.hv.world_rank if config.hv else None)

    # DataLoader preparation
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config)

    model = get_model(config)
    # print("BASE MODEL")
    # for name, param in base_model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    # print("PEFT MODEL")
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    
    # keys1 = set(name for name, _ in base_model.named_parameters())
    # keys2 = set(name for name, _ in model.named_parameters())

    # only_in_model1 = keys1 - keys2
    # only_in_model2 = keys2 - keys1

    # print("🔹 Keys only in model1:", len(keys1))
    # for key in only_in_model1:
    #     print(key)
    
    # print("\n🔸 Keys only in model2:", len(keys2))
    # for key in only_in_model2:
    #     print(key)
    
    # print("\n✅ Comparison complete!")

    model = model.to(local_rank)

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
            # if lora:
            #     params = [p for p in params if p.requires_grad]
            # fake_optimizer = inner_optimizer(params)
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
        )

        diloco_args.update(get_compression_kwargs(config.hv.hivemind_compression))

        if config.hv.averaging_timeout is not None:
            diloco_args["averaging_timeout"] = config.hv.averaging_timeout

        if config.hv.matchmaking_time is not None:
            diloco_args["matchmaking_time"] = config.hv.matchmaking_time

        optimizer = DiLoCoOptimizer(**diloco_args)

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
            if config.lora:
                start_step = 0
            else:
                start_step = scheduler.last_epoch
        else:
            start_step = 0
        # if config.lora:
        #     model = get_peft_model(model, lora_config)
        #     print_trainable_parameters(model)

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
        # if config.lora:
        #     model = get_peft_model(model, lora_config)
        #     print_trainable_parameters(model)

    if resume_from_ckpt:
        log(f"Resumed from checkpoint at step {start_step} with loss {last_loss}")

    model.train()

    if world_messenger_hv and not config.hv.skip_load_from_peers:
        optimizer.load_state_from_peers()

    current_time = time.time()
    log(f"starting from step {start_step}")

    loss_batch = 0

    if world_messenger_hv:
        max_num_peers = 0

    log_activations = {}

    real_step = start_step
    for epoch in range(100):        
        train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config)
        log(f"Epoch {epoch + 1}/{100} started.")
        for step, batch in enumerate(iterable=train_dataloader, start=start_step * gradient_accumulation_steps):
            real_step = real_step + (step + 1) // gradient_accumulation_steps
            is_accumulating = bool((step + 1) % gradient_accumulation_steps)

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
                outputs = model(**batch)
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
                    optimizer.step(scaler=scaler)

                    # todo(sami): refactor to use built in pytorch mechanism to handle scaler manually
                    # should allow to just do scaler.step(optimizer)
                else:
                    scaler.step(optimizer)

                scaler.update()

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


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "PRIME_INTELLECT_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    config = Config(**parse_argv())
    train(config)
    destroy_process_group()
