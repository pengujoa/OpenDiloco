"""
GPS 체크포인트를 로드하여 validation loss / perplexity를 측정하는 스크립트.

Usage:
    python eval_gps_ckpt.py --config config_halos.toml --ckpt_dir /data2tb/gps_ckpts
    python eval_gps_ckpt.py --config config_halos.toml --ckpt_dir /data2tb/gps_ckpts --ckpt step_000032.pt
    python eval_gps_ckpt.py --config config_halos.toml --ckpt_dir /data2tb/gps_ckpts --max_batches 50
    # GPU 1장에 워커 여러 개 (기본: --device 한 곳에만 붙음). 여러 GPU 쓰려면 --devices
    python eval_gps_ckpt.py --config config.toml --ckpt_dir /data/ckpts --parallel 10 --device cuda:0
    python eval_gps_ckpt.py --config config.toml --ckpt_dir /data/ckpts --parallel 4 --devices cuda:0,cuda:1
    python eval_gps_ckpt.py --config config.toml --ckpt_dir /data/ckpts --gps-log gps.log --output eval.csv
"""

import argparse
import csv
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from gps_log_parse import event_time_for_step, parse_gps_log

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


def _gps_context(gps_log_path: str | None) -> tuple[datetime | None, dict[int, datetime], dict[int, datetime]] | None:
    """--gps-log 없으면 None. 있으면 (t0_ready, ckpt_save_ts, step_last_ts)."""
    if not gps_log_path:
        return None
    t0_ready, _, ckpt_save_ts, step_last_ts = parse_gps_log(gps_log_path)
    return (t0_ready, ckpt_save_ts, step_last_ts)


def _apply_gps_fields_to_row(
    r: dict,
    ctx: tuple[datetime | None, dict[int, datetime], dict[int, datetime]] | None,
) -> None:
    """행 dict 에 gps_event_time_iso, seconds_since_gps_ready 설정."""
    if ctx is None:
        r["gps_event_time_iso"] = ""
        r["seconds_since_gps_ready"] = ""
        return
    t0_ready, ckpt_save_ts, step_last_ts = ctx
    step = int(r["step"])
    if step == 0 and t0_ready is not None:
        dt = t0_ready
    else:
        dt = event_time_for_step(step, ckpt_save_ts, step_last_ts)
    if dt:
        r["gps_event_time_iso"] = dt.isoformat()
        if t0_ready is not None:
            r["seconds_since_gps_ready"] = round((dt - t0_ready).total_seconds(), 3)
        else:
            r["seconds_since_gps_ready"] = ""
    else:
        r["gps_event_time_iso"] = ""
        r["seconds_since_gps_ready"] = ""


def _csv_fieldnames(include_device: bool) -> list[str]:
    keys = [
        "gps_event_time_iso",
        "seconds_since_gps_ready",
        "step",
        "loss",
        "perplexity",
        "batches",
        "file",
        "eval_wall_time_iso",
    ]
    if include_device:
        keys.append("device")
    return keys


def _csv_open_sink(path: str, keys: list[str]) -> tuple[object, csv.DictWriter]:
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    f.flush()
    return f, w


def _csv_append_row(f, w: csv.DictWriter, keys: list[str], row: dict) -> None:
    w.writerow({k: row.get(k, "") for k in keys})
    f.flush()


def _rewrite_csv_sorted_by_step(path: str, fieldnames: list[str]) -> None:
    """디스크의 CSV 를 읽어 step(동률이면 file) 순으로 정렬 후 덮어씀."""
    if not os.path.isfile(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        rows = list(reader)
    if not rows:
        return
    rows.sort(key=lambda r: (int((r.get("step") or "0").strip() or "0"), r.get("file", "")))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_devices(devices_arg: str | None) -> list[str]:
    if devices_arg:
        return [d.strip() for d in devices_arg.split(",") if d.strip()]
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    return [f"cuda:{i}" for i in range(n)]


def _eval_job(job: dict) -> dict:
    """spawn 서브프로세스에서 체크포인트(또는 베이스) 하나만 평가."""
    config_path = job["config_path"]
    device = job["device"]
    job_index = job.get("job_index")
    job_total = job.get("job_total")
    max_batches = job.get("max_batches")
    batch_size = job["batch_size"]
    half_precision = job.get("half_precision", True)
    is_base = job.get("is_base", False)
    ckpt_path = job.get("ckpt_path")

    if is_base:
        prog = f"[{job_index}/{job_total}] " if job_index and job_total else ""
        print(f"[{device}] {prog}작업 시작: base_model (검증 데이터/모델 로딩 중…)", flush=True)
    else:
        name = Path(str(ckpt_path)).name
        prog = f"[{job_index}/{job_total}] " if job_index and job_total else ""
        print(f"[{device}] {prog}작업 시작: {name} (검증 데이터/모델 로딩 중…)", flush=True)

    cfg = load_config(config_path)
    model_name = cfg.get("path_model")
    if not model_name:
        raise ValueError("path_model not found in config")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"
    dataloader = get_validation_dataloader(tokenizer, cfg, batch_size)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    if is_base:
        print(f"[{device}] base_model — 평가(validation) 실행 중…", flush=True)
    else:
        print(f"[{device}] {Path(ckpt_path).name} — 가중치 로드 후 평가 실행 중…", flush=True)

    t0 = time.time()
    if is_base:
        loss, ppl, n_batches = evaluate(model, dataloader, device, max_batches, half_precision)
        elapsed = time.time() - t0
        print(f"[{device}] base_model  loss={loss:.4f}  ppl={ppl:.2f}  batches={n_batches}  {elapsed:.1f}s", flush=True)
        return {
            "step": 0,
            "loss": loss,
            "perplexity": ppl,
            "batches": n_batches,
            "file": "base_model",
            "device": device,
            "eval_wall_time_iso": _now_iso(),
        }

    ckpt_path = str(ckpt_path)
    step_count = load_gps_ckpt_into_model(model, ckpt_path)
    loss, ppl, n_batches = evaluate(model, dataloader, device, max_batches, half_precision)
    elapsed = time.time() - t0
    name = Path(ckpt_path).name
    print(f"[{device}] {name}  step={step_count}  loss={loss:.4f}  ppl={ppl:.2f}  batches={n_batches}  {elapsed:.1f}s", flush=True)
    return {
        "step": step_count,
        "loss": loss,
        "perplexity": ppl,
        "batches": n_batches,
        "file": name,
        "device": device,
        "eval_wall_time_iso": _now_iso(),
    }


def main():
    pa = argparse.ArgumentParser(description="Evaluate GPS checkpoints")
    pa.add_argument("--config", required=True, help="Path to config_halos.toml")
    pa.add_argument("--ckpt_dir", required=True, help="GPS checkpoint directory")
    pa.add_argument("--ckpt", default=None, help="Specific checkpoint file (e.g. step_000032.pt). If not set, evaluates all.")
    pa.add_argument("--max_batches", type=int, default=None, help="Limit validation batches per checkpoint")
    pa.add_argument("--batch_size", type=int, default=4, help="Validation batch size")
    pa.add_argument(
        "--device",
        default="cuda:0",
        help="기본 GPU. 순차 모드와 병렬 모드( --devices 없을 때) 모두에서 사용.",
    )
    pa.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="동시 워커 수. 1이면 순차. 2 이상이면 프로세스마다 모델을 따로 올림(같은 GPU에 여러 개 가능).",
    )
    pa.add_argument(
        "--devices",
        default=None,
        help="쉼표 구분 GPU 목록. 지정 시 작업을 라운드로빈 배분. 미지정 시 전부 --device 한 장만 사용.",
    )
    pa.add_argument(
        "--output",
        default=None,
        help="CSV 경로. eval 하나 끝날 때마다 행을 쓰고 flush (중단돼도 지금까지 결과 유지).",
    )
    pa.add_argument(
        "--gps-log",
        default=None,
        help="GPS 서버 로그 경로. 주면 CSV에 gps_event_time_iso, seconds_since_gps_ready(Ready 대비 초) 기록.",
    )
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

    gps_ctx = _gps_context(args.gps_log)
    csv_keys = _csv_fieldnames(args.parallel > 1)
    csv_file = None
    csv_writer = None
    if args.output:
        csv_file, csv_writer = _csv_open_sink(args.output, csv_keys)
        print(f"CSV: 각 eval 직후 append+flush → {args.output}", flush=True)

    def _emit_row(r: dict) -> None:
        _apply_gps_fields_to_row(r, gps_ctx)
        if csv_writer is not None and csv_file is not None:
            _csv_append_row(csv_file, csv_writer, csv_keys, r)

    try:
        if args.parallel > 1:
            devs = _parse_devices(args.devices) if args.devices else [args.device]
            print(f"Parallel workers: {args.parallel}  devices (round-robin): {devs}")
            if len(devs) == 1:
                print(f"  (모든 워커가 동일 GPU {devs[0]} 공유 — VRAM 부족 시 --parallel 또는 --batch_size 줄이기)")
            print()

            jobs: list[dict] = []
            dev_i = 0
            job_total = (1 if args.base_model else 0) + len(ckpt_files)
            ji = 1
            if args.base_model:
                jobs.append(
                    {
                        "config_path": str(Path(args.config).resolve()),
                        "is_base": True,
                        "device": devs[dev_i % len(devs)],
                        "max_batches": args.max_batches,
                        "batch_size": args.batch_size,
                        "job_index": ji,
                        "job_total": job_total,
                    }
                )
                dev_i += 1
                ji += 1
            for ckpt_path in ckpt_files:
                jobs.append(
                    {
                        "config_path": str(Path(args.config).resolve()),
                        "is_base": False,
                        "ckpt_path": str(ckpt_path.resolve()),
                        "device": devs[dev_i % len(devs)],
                        "max_batches": args.max_batches,
                        "batch_size": args.batch_size,
                        "job_index": ji,
                        "job_total": job_total,
                    }
                )
                dev_i += 1
                ji += 1

            max_workers = min(args.parallel, len(jobs))
            ctx = multiprocessing.get_context("spawn")
            results = []
            print(f"Starting {len(jobs)} job(s) with max_workers={max_workers} (spawn)...", flush=True)
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futures = {ex.submit(_eval_job, j): j for j in jobs}
                for fut in as_completed(futures):
                    row = fut.result()
                    results.append(row)
                    _emit_row(row)
            results.sort(key=lambda r: (r["step"], r["file"]))
        else:
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
                print(f"[1/{1 + len(ckpt_files)}] Evaluating BASE MODEL (step 0)", flush=True)
                print(f"{'='*60}")
                t0 = time.time()
                loss, ppl, n_batches = evaluate(model, dataloader, args.device, args.max_batches)
                elapsed = time.time() - t0
                print(f"  loss={loss:.4f}  ppl={ppl:.2f}  batches={n_batches}  time={elapsed:.1f}s")
                row = {
                    "step": 0,
                    "loss": loss,
                    "perplexity": ppl,
                    "batches": n_batches,
                    "file": "base_model",
                    "eval_wall_time_iso": _now_iso(),
                }
                results.append(row)
                _emit_row(row)

            n_ckpt = len(ckpt_files)
            offset = 2 if args.base_model else 1
            for i, ckpt_path in enumerate(ckpt_files):
                k = i + offset
                total_jobs = n_ckpt + (1 if args.base_model else 0)
                print(f"\n{'='*60}", flush=True)
                print(f"[{k}/{total_jobs}] 체크포인트 처리 중: {ckpt_path.name}", flush=True)
                print(f"  경로: {ckpt_path}", flush=True)
                step_count = load_gps_ckpt_into_model(model, str(ckpt_path))
                print(f"GPS step={step_count}")
                print(f"{'='*60}")

                t0 = time.time()
                loss, ppl, n_batches = evaluate(model, dataloader, args.device, args.max_batches)
                elapsed = time.time() - t0

                print(f"  loss={loss:.4f}  ppl={ppl:.2f}  batches={n_batches}  time={elapsed:.1f}s")
                row = {
                    "step": step_count,
                    "loss": loss,
                    "perplexity": ppl,
                    "batches": n_batches,
                    "file": ckpt_path.name,
                    "eval_wall_time_iso": _now_iso(),
                }
                results.append(row)
                _emit_row(row)

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Step':>8}  {'Loss':>10}  {'PPL':>10}  {'File'}")
        print(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*20}")
        for r in results:
            print(f"{r['step']:>8}  {r['loss']:>10.4f}  {r['perplexity']:>10.2f}  {r['file']}")

    finally:
        if csv_file is not None:
            csv_file.close()
        if args.output and os.path.isfile(args.output):
            _rewrite_csv_sorted_by_step(args.output, csv_keys)
            print(f"\nCSV 저장 완료 (step 순 정렬): {args.output}", flush=True)


if __name__ == "__main__":
    main()
