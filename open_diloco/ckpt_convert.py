import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)

from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)

from open_diloco.utils import (
    FakeTokenizedDataset,
    get_compression_kwargs,
    get_sharding_strategy,
    register_metrics_hooks,
)
from torch.distributed.device_mesh import init_device_mesh
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed import destroy_process_group, init_process_group
import datetime
from functools import partial
import fsspec

from peft import get_peft_model, LoraConfig, TaskType
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # CAUSAL_LM is for language generation tasks
    inference_mode=False,           # Enable training mode
    r=8,                            # LoRA rank
    lora_alpha=32,                  # Scaling factor
    lora_dropout=0.1                # Dropout rate
)

lora_ckpt_test = False

TIMEOUT_NCCL_MINUTES = os.environ.get("TIMEOUT_NCCL_MINUTES", 120)
GLOBAL_STATE_FILE = "global_state_dict.pt"
CKPT_PREFIX = "model_step"

torch.set_float32_matmul_precision("high")

init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES))
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

sharding_strategy = ShardingStrategy.NO_SHARD

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

# train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank)"
ds = load_dataset("allenai/c4", "en", streaming=True)
seq_length = 1024

def tokenize_function(data):
    outputs = tokenizer(
        data["text"],
        truncation=True,
        max_length=seq_length,
        padding="max_length",
    )
    return outputs

tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])[
    "train"
]

train_dataset = split_dataset_by_node(tokenized_datasets, world_size=world_size, rank=rank)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

per_device_train_batch_size = 8
num_workers = 4

data_loader = StatefulDataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=per_device_train_batch_size,
        num_workers=num_workers,
    )

#  model = get_model(config)
config_model = LlamaConfig.from_pretrained("PrimeIntellect/llama-1b-fresh", attn_implementation="sdpa")
model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path="PrimeIntellect/llama-1b-fresh", config=config_model)

# lora
if lora_ckpt_test:
    model = get_peft_model(model, lora_config)

model = model.to(local_rank)

precision = "bf16-mixed"
half_precision = precision == "fp16-mixed" or precision == "bf16-mixed"
half_precision_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16-mixed")

if sharding_strategy in [
    ShardingStrategy._HYBRID_SHARD_ZERO2,
    ShardingStrategy.HYBRID_SHARD,
]:
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    nnodes = world_size // local_world_size
    device_mesh = init_device_mesh("cuda", (nnodes, local_world_size), mesh_dim_names=("global", "local"))
else:
    device_mesh = None

torch_compile = True

model = FSDP(
    model,
    sharding_strategy=sharding_strategy,
    mixed_precision=MixedPrecision(param_dtype=half_precision_dtype) if half_precision else None,
    use_orig_params=torch_compile,
    device_mesh=device_mesh,
)

if torch_compile:
    model = torch.compile(model)

lr = 4e-4
inner_optimizer = partial(torch.optim.AdamW, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))  # noqa: F821

warmup_steps = 1000
total_steps = 24_000
def scheduler_fn(opt):
    return get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

resume_from_ckpt = True

#path
if lora_ckpt_test:
    resume_path = "~/cy/OpenDiloco/open_diloco/model_step_24000_cnvt/diloco_rank_0"
else:
    resume_path = "~/cy/OpenDiloco/open_diloco/model_step_24000/diloco_rank_0"

checkpoint_path = resume_path

optimizer = inner_optimizer(model.parameters())
scheduler = scheduler_fn(optimizer)

# 1. Load distributed states
fs_storage_reader = dcp.FsspecReader(checkpoint_path)

model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)

# cy
if lora_ckpt_test:
    model_state_dict_cnvt = {}
    optimizer_state_dict_cnvt = {"state":{}, "param_groups":[]}
    for key, value in model_state_dict.items():
        if "lora" in key:
            pass
        else:
            model_state_dict_cnvt[key] = value
    
    for key, value in optimizer_state_dict["state"].items():
        if "lora" in key:
            pass
        else:
            optimizer_state_dict_cnvt["state"][key] = value

    dcp_state_dict = {
        "model": model_state_dict_cnvt,
        # "optimizer": optimizer_state_dict_cnvt,
    }
else:
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }

# print(dcp_state_dict["optimizer"]["state"].keys())
# print(dcp_state_dict["optimizer"]["param_groups"])

dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)

set_state_dict(
    model,
    optimizer,
    model_state_dict=model_state_dict,
    optim_state_dict=optimizer_state_dict,
)

if lora_ckpt_test:
    pass
else:
    if data_loader is not None:
        with fsspec.open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "rb") as f:
            rank_state_dict = torch.load(f)
        data_loader.load_state_dict(rank_state_dict["data_loader"])

# with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "rb") as f:
#     global_state_dict = torch.load(f)

# # 2. Load global states
# outer_optimizer = None
# if scheduler is not None:
#     scheduler.load_state_dict(global_state_dict["scheduler"])
#     optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
# if outer_optimizer is not None:
#     outer_optimizer.load_state_dict(global_state_dict["outer_optimizer"])
# if scaler is not None:
#     scaler.load_state_dict(global_state_dict["scaler"])


# ##################
# # save
if lora_ckpt_test:
    checkpoint_save_path = "~/cy/OpenDiloco/open_diloco/model_step_24000_cnvt_dummy/diloco_rank_0"
else:
    checkpoint_save_path = "~/cy/OpenDiloco/open_diloco/model_step_24000_cnvt/diloco_rank_0"

fs_storage_writer = dcp.FsspecWriter(checkpoint_save_path, sync_files=False)


# cy
# print(len(dcp_state_dict["model"].keys()))
# for key in dcp_state_dict["model"].keys():
#     print(dcp_state_dict["model"][key])

model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
model_state_dict_cnvt = {}

for key, value in model_state_dict.items():
    new_key = "base_model.model." + key
    if "q_proj" in new_key:
        new_key = new_key.replace("q_proj", "q_proj.base_layer")
    if "v_proj" in new_key:
        new_key = new_key.replace("v_proj", "v_proj.base_layer")
    model_state_dict_cnvt[new_key] = value    

dcp_state_dict = {
    "model": model_state_dict_cnvt,
    "optimizer": optimizer_state_dict,
}

print(dcp_state_dict["model"].keys())

dcp.save(dcp_state_dict, storage_writer=fs_storage_writer)

if data_loader is not None:
    rank_state_dict = {}
    rank_state_dict["data_loader"] = data_loader.state_dict()
    with fsspec.open(os.path.join(checkpoint_save_path, f"__{rank}_0.pt"), "wb") as f:
        torch.save(rank_state_dict, f)

# ##################

# if not save_global_state:
#     return

# # 2. Save global states
# global_state_dict = {"scheduler": scheduler.state_dict(), "loss": loss if loss is not None else 0}
# if outer_optimizer is not None:
#     global_state_dict["outer_optimizer"] = outer_optimizer.state_dict()
# if scaler is not None:
#     global_state_dict["scaler"] = scaler.state_dict()

# with fsspec.open(os.path.join(checkpoint_save_path, GLOBAL_STATE_FILE), "wb") as f:
#     torch.save(global_state_dict, f)