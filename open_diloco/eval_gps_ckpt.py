"""
GPS 체크포인트를 로드하여 validation loss / perplexity를 측정하는 스크립트.

Usage:
    python eval_gps_ckpt.py --config config_halos.toml --ckpt_dir /data2tb/gps_ckpts
    python eval_gps_ckpt.py --config config_halos.toml --ckpt_dir /data2tb/gps_ckpts --ckpt step_000032.pt
    python eval_gps_ckpt.py --config config_halos.toml --ckpt_dir /data2tb/gps_ckpts --max_batches 50
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_validation_dataloader(tokenizer, config: dict, batch_size: int):
    VALIDATION_SEED = 42
    VALIDATION_SAMPLE_SIZE = 1000
    seq_length = config.get("seq_length", 1024)

    dataset_name = config.get("dataset_name_or_path", "")
    if dataset_name.startswith("$"):
        dataset_name = os.environ.get(dataset_name[1:], dataset_name)

    ds = load_dataset(dataset_name, "default", streaming=True, trust_remote_code=True)

    def tokenize_fn(data):
        return tokenizer(data["text"], truncation=True, max_length=seq_length, padding="max_length")

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text", "timestamp", "url"])

    if "validation" in tokenized:
        val_ds = tokenized["validation"]
    elif "val" in tokenized:
        val_ds = tokenized["val"]
    else:
        print("No validation split found, using train split subset.")
        val_ds = tokenized["train"]

    val_ds = val_ds.shuffle(seed=VALIDATION_SEED, buffer_size=10000).take(VALIDATION_SAMPLE_SIZE)
    print(f"Validation dataset: {VALIDATION_SAMPLE_SIZE} samples (seed={VALIDATION_SEED})")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(val_ds, collate_fn=collator, batch_size=batch_size)


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches: int | None = None, half_precision: bool = True):
    model.eval()
    dtype = torch.float16 if half_precision else torch.float32
    total_loss = 0.0
    total_steps = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=half_precision):
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_steps += 1
        if (i + 1) % 20 == 0:
            avg = total_loss / total_steps
            print(f"  batch {i+1}: running avg loss={avg:.4f}, ppl={math.exp(avg):.2f}")

    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_steps


def load_gps_ckpt_into_model(model, ckpt_path: str):
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    params = state["params"]
    step_count = state["step_count"]

    for p, saved_p in zip(model.parameters(), params):
        p.data.copy_(saved_p)

    return step_count


def main():
    pa = argparse.ArgumentParser(description="Evaluate GPS checkpoints")
    pa.add_argument("--config", required=True, help="Path to config_halos.toml")
    pa.add_argument("--ckpt_dir", required=True, help="GPS checkpoint directory")
    pa.add_argument("--ckpt", default=None, help="Specific checkpoint file (e.g. step_000032.pt). If not set, evaluates all.")
    pa.add_argument("--max_batches", type=int, default=None, help="Limit validation batches per checkpoint")
    pa.add_argument("--batch_size", type=int, default=4, help="Validation batch size")
    pa.add_argument("--device", default="cuda:0", help="Device to use")
    pa.add_argument("--output", default=None, help="Save results to JSON file")
    pa.add_argument("--base_model", action="store_true", help="Also evaluate the base model (step 0) for comparison")
    args = pa.parse_args()

    cfg = load_config(args.config)
    model_name = cfg.get("path_model")
    if not model_name:
        raise ValueError("path_model not found in config")

    ckpt_dir = Path(args.ckpt_dir)
    if args.ckpt:
        ckpt_files = [ckpt_dir / args.ckpt]
    else:
        ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    print(f"Model: {model_name}")
    print(f"Checkpoints: {len(ckpt_files)} file(s) in {ckpt_dir}")
    print(f"Device: {args.device}")
    print()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"
    dataloader = get_validation_dataloader(tokenizer, cfg, args.batch_size)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(args.device)

    results = []

    if args.base_model:
        print(f"\n{'='*60}")
        print(f"Evaluating BASE MODEL (step 0)")
        print(f"{'='*60}")
        t0 = time.time()
        loss, ppl, n_batches = evaluate(model, dataloader, args.device, args.max_batches)
        elapsed = time.time() - t0
        print(f"  loss={loss:.4f}  ppl={ppl:.2f}  batches={n_batches}  time={elapsed:.1f}s")
        results.append({"step": 0, "loss": loss, "perplexity": ppl, "batches": n_batches, "file": "base_model"})

    for ckpt_path in ckpt_files:
        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {ckpt_path.name}")
        step_count = load_gps_ckpt_into_model(model, str(ckpt_path))
        print(f"GPS step={step_count}")
        print(f"{'='*60}")

        t0 = time.time()
        loss, ppl, n_batches = evaluate(model, dataloader, args.device, args.max_batches)
        elapsed = time.time() - t0

        print(f"  loss={loss:.4f}  ppl={ppl:.2f}  batches={n_batches}  time={elapsed:.1f}s")
        results.append({"step": step_count, "loss": loss, "perplexity": ppl, "batches": n_batches, "file": ckpt_path.name})

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Step':>8}  {'Loss':>10}  {'PPL':>10}  {'File'}")
    print(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*20}")
    for r in results:
        print(f"{r['step']:>8}  {r['loss']:>10.4f}  {r['perplexity']:>10.2f}  {r['file']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
