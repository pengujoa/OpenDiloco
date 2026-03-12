"""
WandB 메트릭 수집 및 실험 결과 요약

1. Raw data 수집:
   - System 메트릭: GPU/비-GPU CSV 저장
   - Training 메트릭: charts (validation_perplexity, validation_loss, tokens_per_second 등) CSV 저장
   - Logs: output.log (콘솔 stdout/stderr, 웹 UI 10k 제한 없이 전체)
2. 실험 요약: 실험별·GPU별 평균 utilization, 최대 memory allocated 정리
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import wandb

# 설정
ENTITY = "cyshin-korea-university"
PROJECTS = [
    "motivation_emnlp_4nodes_1345-0",
    "motivation_emnlp_4nodes_1345-1",
    "motivation_emnlp_4nodes_1345-2",
    "motivation_emnlp_4nodes_1345-3",
    # "emnlp_finding_ema_alpha-0",
    # "emnlp_finding_ema_alpha-1",
    # "emnlp_finding_ema_alpha-2",
    # "emnlp_finding_ema_alpha-3",
    # 추가 프로젝트 예시:
    # "other_project_name",
]
SYSTEM_METRICS_SAMPLES = 100_000  # raw data에 가깝게 (WandB 서버 상한 있을 수 있음)
WANDB_LOGS_DIR = "wandb_logs"  # raw data·summary 상위 디렉터리

# 요약용 컬럼명
UTIL_COL = "gpu"
MEMORY_COL = "memoryAllocatedBytes"


# =============================================================================
# 1. Raw Data 수집 함수
# =============================================================================


def fetch_raw_system_metrics_from_wandb(
    entity: str,
    projects: List[str],
    samples: int = SYSTEM_METRICS_SAMPLES,
    save_dir: str = ".",
    skip_if_exists: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    WandB API에서 각 Project의 Run별 System 메트릭을 가져와 GPU/비-GPU CSV로 저장합니다.
    skip_if_exists=True이면 이미 system_metrics_gpu_*.csv가 있는 Run은 다운로드 생략하고 로컬 CSV에서 로드.
    """
    api = wandb.Api()
    time_cols = ["_runtime", "_timestamp", "_step"]
    gpu_data: Dict[str, pd.DataFrame] = {}

    def drop_empty_and_dupes(sub_df, metric_cols):
        if not metric_cols:
            return sub_df
        has_data = sub_df[metric_cols].notna().any(axis=1)
        return sub_df.loc[has_data].drop_duplicates()

    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}")
        except Exception:
            continue

        for run in runs:
            if not run.name.startswith("[v]"):
                continue
            safe_run_name = run.name.replace("/", "_")
            dir_name = os.path.join(save_dir, f"{entity}_{project}_{safe_run_name}")
            gpu_path = os.path.join(dir_name, f"system_metrics_gpu_{safe_run_name}.csv")
            run_key = f"{project}_{safe_run_name}"

            if skip_if_exists and os.path.isfile(gpu_path):
                try:
                    gpu_data[run_key] = pd.read_csv(gpu_path)
                except Exception:
                    pass
                continue

            os.makedirs(dir_name, exist_ok=True)
            try:
                df = run.history(stream="system", samples=samples)
            except Exception:
                try:
                    df = run.history(stream="events", samples=samples)
                except Exception:
                    try:
                        df = run.history(stream="systemMetrics", samples=samples)
                    except Exception:
                        continue

            if df is None or df.empty:
                continue

            tc = [c for c in time_cols if c in df.columns]
            gpu_cols = [c for c in df.columns if "gpu" in c.lower()]
            non_gpu_cols = [c for c in df.columns if "gpu" not in c.lower() and c != "_wandb"]

            if gpu_cols:
                gpu_df = df[tc + gpu_cols].copy()
                gpu_df = drop_empty_and_dupes(gpu_df, gpu_cols)
                gpu_df.to_csv(gpu_path, index=False)
                gpu_data[run_key] = gpu_df

            if non_gpu_cols:
                metric_cols = [c for c in non_gpu_cols if c not in tc]
                non_gpu_df = df[non_gpu_cols].copy()
                non_gpu_df = drop_empty_and_dupes(non_gpu_df, metric_cols)
                non_gpu_path = os.path.join(dir_name, f"system_metrics_non_gpu_{safe_run_name}.csv")
                non_gpu_df.to_csv(non_gpu_path, index=False)

    return gpu_data


def fetch_raw_training_metrics_from_wandb(
    entity: str,
    projects: List[str],
    save_dir: str = ".",
    skip_if_exists: bool = False,
) -> None:
    """
    WandB API에서 각 Run의 Training 메트릭(charts)을 가져와 CSV로 저장합니다.
    skip_if_exists=True이면 이미 training_metrics_*.csv가 있는 Run은 다운로드 생략.
    """
    api = wandb.Api()

    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}")
        except Exception:
            continue

        for run in runs:
            if not run.name.startswith("[v]"):
                continue
            safe_run_name = run.name.replace("/", "_")
            dir_name = os.path.join(save_dir, f"{entity}_{project}_{safe_run_name}")
            out_path = os.path.join(dir_name, f"training_metrics_{safe_run_name}.csv")
            if skip_if_exists and os.path.isfile(out_path):
                continue
            os.makedirs(dir_name, exist_ok=True)

            df = None
            try:
                history_data = [row for row in run.scan_history()]
                if history_data:
                    df = pd.DataFrame(history_data)
            except Exception:
                pass

            if df is None or df.empty:
                try:
                    df = run.history(stream="default", samples=100_000)
                except Exception:
                    continue

            if df is None or df.empty:
                continue
            df.to_csv(out_path, index=False)


def fetch_raw_logs_from_wandb(
    entity: str,
    projects: List[str],
    save_dir: str = ".",
    skip_if_exists: bool = False,
) -> None:
    """
    WandB API에서 각 Run의 output.log(콘솔 stdout/stderr)를 다운로드합니다.
    skip_if_exists=True이면 이미 output.log가 있는 Run은 다운로드 생략.
    """
    api = wandb.Api()

    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}")
        except Exception:
            continue

        for run in runs:
            if not run.name.startswith("[v]"):
                continue
            safe_run_name = run.name.replace("/", "_")
            dir_name = os.path.join(save_dir, f"{entity}_{project}_{safe_run_name}")
            log_path = os.path.join(dir_name, "output.log")
            if skip_if_exists and os.path.isfile(log_path):
                continue
            os.makedirs(dir_name, exist_ok=True)

            try:
                f = run.file("output.log")
                f.download(root=dir_name, replace=True)
            except Exception:
                pass


# output.log 파싱용 정규식
_VALIDATION_PATTERN = re.compile(
    r"validation_loss=([\d.]+)\s*,\s*validation_perplexity=([\d.]+)"
)
_TIMESTAMP_PATTERN = re.compile(r"(\w{3}\s+\d+\s+\d+:\d+:\d+\.\d+)")
_GPU_LOCAL_TRAINING_PATTERN = re.compile(
    r"\[TIMING\] GPU local training time:\s*([\d.]+)\s*sec"
)
_GRADIENT_MAGNITUDE_PATTERN = re.compile(
    r"Gradient magnitude selection:\s*([\d.]+)\s*sec.*Sync wait \(GPU idle\):\s*([\d.]+)\s*sec"
)
_TOKEN_WEIGHT_SYNC_PATTERN = re.compile(r"Token weight sync:\s*([\d.]+)\s*sec")
_SYNC_WAIT_GPU_IDLE_PATTERN = re.compile(r"Sync wait \(GPU idle\):\s*([\d.]+)\s*sec")
_VALIDATION_SYNC_WAIT_PATTERN = re.compile(r"\[TIMING\] Validation sync wait:\s*([\d.]+)\s*sec")
_ALLREDUCE_PATTERN = re.compile(r"All-reduce networking time:\s*([\d.]+)\s*sec")
_PHASE_TRANSITION_EPOCH_PATTERN = re.compile(r"Phase transition completed at epoch\s*(\d+)")
_STATE_AVERAGER_PATTERN = re.compile(r"Time taken for state_averager_step:\s*([\d.]+)\s*sec")
_VALIDATION_TIME_PATTERN = re.compile(r"Validation completed:.*time:\s*([\d.]+)\s*seconds")


def _parse_timestamp_to_seconds(ts_str: str, default_year: int = 2026) -> float:
    """타임스탬프 문자열(예: Feb 18 19:25:32.606, 2026-02-22 12:50:29)을 epoch 초로 변환"""
    ts_str = ts_str.strip()
    frac = "0"
    if "." in ts_str:
        ts_str, frac = ts_str.rsplit(".", 1)
        frac = frac.ljust(6, "0")[:6]
    try:
        # Feb 18 19:25:32 형식
        dt = datetime.strptime(ts_str, "%b %d %H:%M:%S")
        return dt.replace(year=default_year).timestamp() + int(frac) / 1_000_000
    except Exception:
        pass
    try:
        # 2026-02-22 12:50:29 형식
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp() + int(frac) / 1_000_000
    except Exception:
        return 0.0


def parse_log_file(log_path: str, pattern: str = None) -> List[str]:
    """
    output.log 파일을 읽어 파싱합니다.

    Args:
        log_path: output.log 파일 경로
        pattern: 정규식 패턴. 지정 시 매칭되는 줄만 반환 (예: r"\[TIMING\]", r"\[INFO\]")

    Returns:
        로그 줄 리스트
    """
    if not os.path.isfile(log_path):
        return []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if pattern:
        lines = [ln for ln in lines if re.search(pattern, ln)]
    return lines


def extract_e2e_training_time_from_log(log_path: str) -> Optional[float]:
    """
    output.log에서 e2e training time(초) 추출.
    'Starting training loop' ~ 'Training completed' 구간의 시간 차이.
    """
    lines = parse_log_file(log_path, pattern=r"Starting training loop|Training completed")
    start_ts, end_ts = None, None
    for ln in lines:
        ts_m = _TIMESTAMP_PATTERN.search(ln)
        if not ts_m:
            continue
        ts_str = ts_m.group(1)
        ts_sec = _parse_timestamp_to_seconds(ts_str)
        if "Starting training loop" in ln:
            start_ts = ts_sec
        elif "Training completed" in ln:
            end_ts = ts_sec
    if start_ts and end_ts and end_ts > start_ts:
        return end_ts - start_ts
    return None


def extract_training_start_timestamp_from_log(log_path: str) -> Optional[float]:
    """output.log에서 'Starting training loop' 줄의 타임스탬프(epoch 초) 추출."""
    lines = parse_log_file(log_path, pattern=r"Starting training loop")
    for ln in lines:
        ts_m = _TIMESTAMP_PATTERN.search(ln)
        if ts_m:
            return _parse_timestamp_to_seconds(ts_m.group(1))
    return None


def extract_phase_transition_timestamp_from_log(log_path: str) -> Optional[float]:
    """output.log에서 'Phase transition completed at epoch' 줄의 타임스탬프(epoch 초) 추출."""
    lines = parse_log_file(log_path, pattern=r"Phase transition completed at epoch")
    for ln in lines:
        ts_m = _TIMESTAMP_PATTERN.search(ln)
        if ts_m:
            return _parse_timestamp_to_seconds(ts_m.group(1))
    return None


def extract_gpu_local_training_times_from_log(log_path: str) -> pd.DataFrame:
    """
    output.log에서 [TIMING] GPU local training time 전부 추출.
    Returns: DataFrame with columns: timestamp, gpu_local_training_time_sec
    """
    lines = parse_log_file(log_path, pattern=r"\[TIMING\] GPU local training time:")
    rows = []
    for ln in lines:
        m = _GPU_LOCAL_TRAINING_PATTERN.search(ln)
        if not m:
            continue
        sec = float(m.group(1))
        ts = ""
        ts_m = _TIMESTAMP_PATTERN.search(ln)
        if ts_m:
            ts = ts_m.group(1)
        rows.append({"timestamp": ts, "gpu_local_training_time_sec": sec})
    return pd.DataFrame(rows)


def extract_gradient_magnitude_sync_wait_from_log(log_path: str) -> pd.DataFrame:
    """
    output.log에서 Gradient magnitude selection, Sync wait (GPU idle) 추출.
    """
    lines = parse_log_file(log_path, pattern=r"Gradient magnitude selection:")
    rows = []
    for ln in lines:
        m = _GRADIENT_MAGNITUDE_PATTERN.search(ln)
        if not m:
            continue
        grad_sec, sync_sec = float(m.group(1)), float(m.group(2))
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({
            "timestamp": ts,
            "gradient_magnitude_selection_sec": grad_sec,
            "sync_wait_gpu_idle_sec": sync_sec,
        })
    return pd.DataFrame(rows)


def extract_allreduce_networking_times_from_log(log_path: str) -> pd.DataFrame:
    """output.log에서 All-reduce networking time 추출."""
    lines = parse_log_file(log_path, pattern=r"All-reduce networking time:")
    rows = []
    for ln in lines:
        m = _ALLREDUCE_PATTERN.search(ln)
        if not m:
            continue
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({"timestamp": ts, "allreduce_networking_time_sec": float(m.group(1))})
    return pd.DataFrame(rows)


def extract_phase_transition_epoch_from_log(log_path: str) -> Optional[int]:
    """output.log에서 Phase transition completed at epoch N 추출. 없으면 None."""
    lines = parse_log_file(log_path, pattern=r"Phase transition completed at epoch")
    for ln in lines:
        m = _PHASE_TRANSITION_EPOCH_PATTERN.search(ln)
        if m:
            return int(m.group(1))
    return None


def extract_state_averager_step_times_from_log(log_path: str) -> pd.DataFrame:
    """output.log에서 state_averager_step time 추출."""
    lines = parse_log_file(log_path, pattern=r"Time taken for state_averager_step:")
    rows = []
    for ln in lines:
        m = _STATE_AVERAGER_PATTERN.search(ln)
        if not m:
            continue
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({"timestamp": ts, "state_averager_step_sec": float(m.group(1))})
    return pd.DataFrame(rows)


def extract_validation_times_from_log(log_path: str) -> pd.DataFrame:
    """output.log에서 Validation completed time 추출."""
    lines = parse_log_file(log_path, pattern=r"Validation completed:.*time:")
    rows = []
    for ln in lines:
        m = _VALIDATION_TIME_PATTERN.search(ln)
        if not m:
            continue
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({"timestamp": ts, "validation_time_sec": float(m.group(1))})
    return pd.DataFrame(rows)


def extract_token_weight_sync_times_from_log(log_path: str) -> pd.DataFrame:
    """output.log에서 Token weight sync 전부 추출 (standalone + gradient line)."""
    lines = parse_log_file(log_path, pattern=r"Token weight sync:")
    rows = []
    for ln in lines:
        m = _TOKEN_WEIGHT_SYNC_PATTERN.search(ln)
        if not m:
            continue
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({"timestamp": ts, "token_weight_sync_sec": float(m.group(1))})
    return pd.DataFrame(rows)


def extract_sync_wait_gpu_idle_all_from_log(log_path: str) -> pd.DataFrame:
    """output.log에서 Sync wait (GPU idle) 전부 추출 (standalone + gradient line)."""
    lines = parse_log_file(log_path, pattern=r"Sync wait \(GPU idle\):")
    rows = []
    for ln in lines:
        m = _SYNC_WAIT_GPU_IDLE_PATTERN.search(ln)
        if not m:
            continue
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({"timestamp": ts, "sync_wait_gpu_idle_sec": float(m.group(1))})
    return pd.DataFrame(rows)


def extract_validation_intervals_from_log(log_path: str) -> List[Tuple[float, float]]:
    """
    output.log에서 validation 구간의 (시작 epoch, 종료 epoch) 리스트 추출.
    시작: "Starting validation loop (max_batches=None)..."
    종료: "[TIMING] Validation sync wait: ..." (해당 줄 시각까지가 validation 시간)
    Returns:
        [(start_epoch_sec, end_epoch_sec), ...] — 로그 순서대로 쌍 구성
    """
    start_lines = parse_log_file(log_path, pattern=r"Starting validation loop \(max_batches=None\)")
    end_lines = parse_log_file(log_path, pattern=r"\[TIMING\] Validation sync wait:")
    starts: List[float] = []
    ends: List[float] = []
    for ln in start_lines:
        m = _TIMESTAMP_PATTERN.search(ln)
        if m:
            starts.append(_parse_timestamp_to_seconds(m.group(1)))
    for ln in end_lines:
        m = _TIMESTAMP_PATTERN.search(ln)
        if m:
            ends.append(_parse_timestamp_to_seconds(m.group(1)))
    if not starts or not ends or len(starts) != len(ends):
        return []
    return list(zip(starts, ends))


def extract_validation_sync_wait_times_from_log(log_path: str) -> pd.DataFrame:
    """output.log에서 Validation sync wait 추출."""
    lines = parse_log_file(log_path, pattern=r"\[TIMING\] Validation sync wait:")
    rows = []
    for ln in lines:
        m = _VALIDATION_SYNC_WAIT_PATTERN.search(ln)
        if not m:
            continue
        ts = _TIMESTAMP_PATTERN.search(ln).group(1) if _TIMESTAMP_PATTERN.search(ln) else ""
        rows.append({"timestamp": ts, "validation_sync_wait_sec": float(m.group(1))})
    return pd.DataFrame(rows)


def extract_time_to_ppl_below_threshold(
    validation_df: pd.DataFrame,
    validation_times: List[float],
    threshold: float = 30.0,
    validation_sync_wait_times: Optional[List[float]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    2번째 epoch 시작(=1번째 validation 완료 시점)부터
    validation perplexity가 처음으로 threshold 미만이 되는 시점까지의 시간과 epoch 수.

    Returns:
        (time_incl_val, time_excl_val, epoch_count) — 조건 미충족 시 (None, None, None)
        - time_incl_val: validation 시간 포함 wall clock (초)
        - time_excl_val: validation 시간 + validation_sync_wait(idle) 제외 (초)
        - epoch_count: 2번째 epoch부터 threshold 도달까지 수행한 epoch 수
    """
    if validation_df is None or validation_df.empty or len(validation_df) < 2:
        return None, None, None

    start_ts_str = validation_df.iloc[0].get("timestamp", "")
    if not start_ts_str:
        return None, None, None
    start_ts = _parse_timestamp_to_seconds(start_ts_str)
    if start_ts == 0.0:
        return None, None, None

    sync_wait = validation_sync_wait_times or []

    for idx in range(1, len(validation_df)):
        row = validation_df.iloc[idx]
        if row["validation_perplexity"] < threshold:
            ts_str = row.get("timestamp", "")
            if not ts_str:
                return None, None, None
            end_ts = _parse_timestamp_to_seconds(ts_str)
            if end_ts <= start_ts:
                return None, None, None

            time_incl_val = end_ts - start_ts

            cumul_val_time = 0.0
            for vi in range(1, min(idx + 1, len(validation_times))):
                cumul_val_time += validation_times[vi]
            for vi in range(1, min(idx + 1, len(sync_wait))):
                cumul_val_time += sync_wait[vi]
            time_excl_val = time_incl_val - cumul_val_time

            epoch_count = idx
            return time_incl_val, time_excl_val, epoch_count

    return None, None, None


def extract_validation_from_log(log_path: str) -> pd.DataFrame:
    """
    output.log에서 Validation results 줄을 파싱해 validation_loss, validation_perplexity를 추출합니다.

    Returns:
        DataFrame with columns: timestamp, validation_loss, validation_perplexity
    """
    lines = parse_log_file(log_path, pattern=r"Validation results:")
    rows = []
    for ln in lines:
        m = _VALIDATION_PATTERN.search(ln)
        if not m:
            continue
        loss, ppl = float(m.group(1)), float(m.group(2))
        ts = ""
        ts_m = _TIMESTAMP_PATTERN.search(ln)
        if ts_m:
            ts = ts_m.group(1)
        rows.append({"timestamp": ts, "validation_loss": loss, "validation_perplexity": ppl})
    return pd.DataFrame(rows)


def _load_validation_results_from_csv(path: str) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    실험 폴더에 output.log 없이 이미 저장된 CSV만으로 validation_data·log_extra 한 개분 복원.
    최소한 validation_results.csv 또는 allreduce_networking_time.csv 등 핵심 CSV 하나 이상 있어야 유효.
    Returns:
        (validation_df, log_extra_dict) — 복원 불가면 (None, None)
    """
    val_path = os.path.join(path, "validation_results.csv")
    val_df = pd.read_csv(val_path) if os.path.isfile(val_path) else None
    if val_df is not None and val_df.empty:
        val_df = None

    def _read_sec_col(csv_name: str, col: str) -> List[float]:
        p = os.path.join(path, csv_name)
        if not os.path.isfile(p):
            return []
        try:
            df = pd.read_csv(p)
            if col not in df.columns:
                return []
            return pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        except Exception:
            return []

    gpu_times = _read_sec_col("gpu_local_training_time.csv", "gpu_local_training_time_sec")
    token_weight_times = _read_sec_col("token_weight_sync_time.csv", "token_weight_sync_sec")
    sync_wait_times = _read_sec_col("sync_wait_gpu_idle.csv", "sync_wait_gpu_idle_sec")
    validation_sync_wait_times = _read_sec_col("validation_sync_wait_time.csv", "validation_sync_wait_sec")
    allreduce_times = _read_sec_col("allreduce_networking_time.csv", "allreduce_networking_time_sec")
    allreduce_before_pt = _read_sec_col("allreduce_networking_time_before_phase_transition.csv", "allreduce_networking_time_sec")
    allreduce_after_pt = _read_sec_col("allreduce_networking_time_after_phase_transition.csv", "allreduce_networking_time_sec")
    state_times = _read_sec_col("state_averager_step_time.csv", "state_averager_step_sec")
    val_times = _read_sec_col("validation_time.csv", "validation_time_sec")

    if val_df is None and not allreduce_times and not gpu_times:
        return None, None

    val_df_for_ppl = val_df
    if val_df_for_ppl is None:
        val_df_for_ppl = pd.DataFrame(columns=["timestamp", "validation_loss", "validation_perplexity"])
    ppl50_incl, ppl50_excl, ppl50_epochs = extract_time_to_ppl_below_threshold(
        val_df_for_ppl, val_times, threshold=50.0, validation_sync_wait_times=validation_sync_wait_times or None
    )
    ppl40_incl, ppl40_excl, ppl40_epochs = extract_time_to_ppl_below_threshold(
        val_df_for_ppl, val_times, threshold=40.0, validation_sync_wait_times=validation_sync_wait_times or None
    )
    ppl30_incl, ppl30_excl, ppl30_epochs = extract_time_to_ppl_below_threshold(
        val_df_for_ppl, val_times, threshold=30.0, validation_sync_wait_times=validation_sync_wait_times or None
    )

    extra = {
        "e2e_sec": None,
        "gpu_local_times": gpu_times,
        "sync_wait_gpu_idle_times": sync_wait_times,
        "token_weight_sync_times": token_weight_times,
        "validation_sync_wait_times": validation_sync_wait_times,
        "ppl50_incl_val_sec": ppl50_incl,
        "ppl50_excl_val_sec": ppl50_excl,
        "ppl50_epochs": ppl50_epochs,
        "ppl40_incl_val_sec": ppl40_incl,
        "ppl40_excl_val_sec": ppl40_excl,
        "ppl40_epochs": ppl40_epochs,
        "ppl30_incl_val_sec": ppl30_incl,
        "ppl30_excl_val_sec": ppl30_excl,
        "ppl30_epochs": ppl30_epochs,
        "allreduce_times": allreduce_times,
        "allreduce_times_before_pt": allreduce_before_pt,
        "allreduce_times_after_pt": allreduce_after_pt,
        "phase_transition_timestamp_sec": None,
        "training_start_timestamp_sec": None,
        "state_averager_times": state_times,
        "validation_times": val_times,
    }
    return val_df, extra


def extract_validation_results_from_logs(
    base_dir: str = ".",
    from_csv_only: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    """
    wandb_logs 내 각 실험 폴더에서 validation·timing 데이터 수집.
    - from_csv_only=True: output.log를 읽지 않고 기존 CSV만 사용.
    - output.log가 있으면(그리고 from_csv_only=False): 로그 파싱 후 CSV 저장.
    - output.log가 없으면: 이미 있는 CSV만 읽어서 반영.
    Returns:
        (validation_data, log_extra)
    """
    validation_result: Dict[str, pd.DataFrame] = {}
    log_extra: Dict[str, dict] = {}

    for dir_name in os.listdir(base_dir):
        path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(path):
            continue
        parts = dir_name.split("_", 1)
        exp_key = parts[1] if len(parts) >= 2 else dir_name
        log_path = os.path.join(path, "output.log")

        if from_csv_only or not os.path.isfile(log_path):
            # CSV만 사용 (로그 미사용 또는 로그 없음)
            val_df, extra = _load_validation_results_from_csv(path)
            if val_df is not None:
                validation_result[exp_key] = val_df
            if extra is not None:
                log_extra[exp_key] = extra
            continue

        # output.log 있음 → 로그 파싱 후 CSV 저장
        # validation
        df = extract_validation_from_log(log_path)
        if not df.empty:
            validation_result[exp_key] = df
            out_path = os.path.join(path, "validation_results.csv")
            df.to_csv(out_path, index=False)

        # e2e training time
        e2e_sec = extract_e2e_training_time_from_log(log_path)

        # GPU local training time
        gpu_df = extract_gpu_local_training_times_from_log(log_path)
        gpu_times = gpu_df["gpu_local_training_time_sec"].tolist() if not gpu_df.empty else []
        if not gpu_df.empty:
            gpu_df.to_csv(os.path.join(path, "gpu_local_training_time.csv"), index=False)

        # Token weight sync (모든 발생)
        token_weight_df = extract_token_weight_sync_times_from_log(log_path)
        token_weight_times = token_weight_df["token_weight_sync_sec"].tolist() if not token_weight_df.empty else []
        if not token_weight_df.empty:
            token_weight_df.to_csv(os.path.join(path, "token_weight_sync_time.csv"), index=False)

        # Sync wait (GPU idle) (모든 발생)
        sync_wait_all_df = extract_sync_wait_gpu_idle_all_from_log(log_path)
        sync_wait_times = sync_wait_all_df["sync_wait_gpu_idle_sec"].tolist() if not sync_wait_all_df.empty else []
        if not sync_wait_all_df.empty:
            sync_wait_all_df.to_csv(os.path.join(path, "sync_wait_gpu_idle.csv"), index=False)

        # Validation sync wait
        val_sync_df = extract_validation_sync_wait_times_from_log(log_path)
        validation_sync_wait_times = val_sync_df["validation_sync_wait_sec"].tolist() if not val_sync_df.empty else []
        if not val_sync_df.empty:
            val_sync_df.to_csv(os.path.join(path, "validation_sync_wait_time.csv"), index=False)

        # All-reduce networking time
        allreduce_df = extract_allreduce_networking_times_from_log(log_path)
        allreduce_times = allreduce_df["allreduce_networking_time_sec"].tolist() if not allreduce_df.empty else []
        if not allreduce_df.empty:
            allreduce_df.to_csv(os.path.join(path, "allreduce_networking_time.csv"), index=False)

        # outer opt change 실험: phase transition 전/후 all-reduce 시간 구분 + GPU 구간 분할용 타임스탬프
        allreduce_times_before_pt: List[float] = []
        allreduce_times_after_pt: List[float] = []
        phase_transition_timestamp_sec: Optional[float] = None
        training_start_timestamp_sec: Optional[float] = None
        if "outer opt change" in exp_key and allreduce_times:
            pt_epoch = extract_phase_transition_epoch_from_log(log_path)
            if pt_epoch is not None:
                # phase transition completed at epoch K → 0..K 구간 전, K+1.. 후
                split_idx = pt_epoch + 1
                allreduce_times_before_pt = allreduce_times[:split_idx]
                allreduce_times_after_pt = allreduce_times[split_idx:]
                if allreduce_times_before_pt and not allreduce_df.empty:
                    before_df = allreduce_df.iloc[:split_idx]
                    before_df.to_csv(os.path.join(path, "allreduce_networking_time_before_phase_transition.csv"), index=False)
                if allreduce_times_after_pt and not allreduce_df.empty and split_idx < len(allreduce_df):
                    after_df = allreduce_df.iloc[split_idx:]
                    after_df.to_csv(os.path.join(path, "allreduce_networking_time_after_phase_transition.csv"), index=False)
                phase_transition_timestamp_sec = extract_phase_transition_timestamp_from_log(log_path)
                training_start_timestamp_sec = extract_training_start_timestamp_from_log(log_path)

        # state_averager_step time
        state_df = extract_state_averager_step_times_from_log(log_path)
        state_times = state_df["state_averager_step_sec"].tolist() if not state_df.empty else []
        if not state_df.empty:
            state_df.to_csv(os.path.join(path, "state_averager_step_time.csv"), index=False)

        # validation time
        val_time_df = extract_validation_times_from_log(log_path)
        val_times = val_time_df["validation_time_sec"].tolist() if not val_time_df.empty else []
        if not val_time_df.empty:
            val_time_df.to_csv(os.path.join(path, "validation_time.csv"), index=False)

        # Time to perplexity < threshold (50, 40, 30) — 2번째 epoch 시작 기준
        val_df = validation_result.get(exp_key)
        if val_df is None:
            val_df = extract_validation_from_log(log_path)
        ppl50_incl, ppl50_excl, ppl50_epochs = extract_time_to_ppl_below_threshold(
            val_df, val_times, threshold=50.0, validation_sync_wait_times=validation_sync_wait_times
        )
        ppl40_incl, ppl40_excl, ppl40_epochs = extract_time_to_ppl_below_threshold(
            val_df, val_times, threshold=40.0, validation_sync_wait_times=validation_sync_wait_times
        )
        ppl30_incl, ppl30_excl, ppl30_epochs = extract_time_to_ppl_below_threshold(
            val_df, val_times, threshold=30.0, validation_sync_wait_times=validation_sync_wait_times
        )

        validation_intervals_epoch_sec = extract_validation_intervals_from_log(log_path)

        log_extra[exp_key] = {
            "e2e_sec": e2e_sec,
            "gpu_local_times": gpu_times,
            "sync_wait_gpu_idle_times": sync_wait_times,
            "token_weight_sync_times": token_weight_times,
            "validation_sync_wait_times": validation_sync_wait_times,
            "ppl50_incl_val_sec": ppl50_incl,
            "ppl50_excl_val_sec": ppl50_excl,
            "ppl50_epochs": ppl50_epochs,
            "ppl40_incl_val_sec": ppl40_incl,
            "ppl40_excl_val_sec": ppl40_excl,
            "ppl40_epochs": ppl40_epochs,
            "ppl30_incl_val_sec": ppl30_incl,
            "ppl30_excl_val_sec": ppl30_excl,
            "ppl30_epochs": ppl30_epochs,
            "allreduce_times": allreduce_times,
            "allreduce_times_before_pt": allreduce_times_before_pt,
            "allreduce_times_after_pt": allreduce_times_after_pt,
            "phase_transition_timestamp_sec": phase_transition_timestamp_sec,
            "training_start_timestamp_sec": training_start_timestamp_sec,
            "validation_intervals_epoch_sec": validation_intervals_epoch_sec,
            "state_averager_times": state_times,
            "validation_times": val_times,
        }

    return validation_result, log_extra


def fetch_raw_gpu_metrics_from_directory(base_dir: str = ".") -> Dict[str, pd.DataFrame]:
    """
    로컬에서 이미 저장된 system_metrics_gpu_*.csv를 로드합니다.
    키: project_runname (디렉터리명 entity_project_runname에서 entity_ 제거)
    """
    result: Dict[str, pd.DataFrame] = {}
    for dir_name in os.listdir(base_dir):
        path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(path):
            continue
        for f in os.listdir(path):
            if f.startswith("system_metrics_gpu_") and f.endswith(".csv"):
                # entity_project_runname → project_runname (fetch 시 사용한 키와 동일)
                parts = dir_name.split("_", 1)
                exp_key = parts[1] if len(parts) >= 2 else dir_name
                result[exp_key] = pd.read_csv(os.path.join(path, f))
                break
    return result


# =============================================================================
# 2. 실험 결과 요약 함수
# =============================================================================


def _extract_gpu_ids(df: pd.DataFrame) -> list:
    pattern = re.compile(r"system\.gpu\.(\d+)\.")
    ids = set()
    for c in df.columns:
        m = pattern.match(c)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _gpu_util_before_after_phase_transition(
    gpu_df: pd.DataFrame,
    phase_transition_timestamp_sec: float,
    training_start_timestamp_sec: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    GPU 메트릭을 phase transition 시점 기준으로 나누어 전/후 평균 utilization(% ) 반환.
    Returns:
        (avg_util_before_pct, avg_util_after_pct) — 구간에 데이터 없으면 None
    """
    util_cols = [c for c in gpu_df.columns if c.endswith(f".{UTIL_COL}") and "system.gpu." in c]
    if not util_cols:
        return None, None

    time_col = None
    phase_bound_sec = phase_transition_timestamp_sec
    if "_timestamp" in gpu_df.columns:
        time_col = "_timestamp"
        ser = pd.to_numeric(gpu_df[time_col], errors="coerce").dropna()
        if ser.max() > 1e12:
            phase_bound_sec = phase_transition_timestamp_sec * 1000.0
    elif "_runtime" in gpu_df.columns and training_start_timestamp_sec is not None:
        time_col = "_runtime"
        phase_bound_sec = phase_transition_timestamp_sec - training_start_timestamp_sec
    else:
        return None, None

    gpu_df = gpu_df.copy()
    gpu_df["_time_sec"] = pd.to_numeric(gpu_df[time_col], errors="coerce")
    before_df = gpu_df[gpu_df["_time_sec"] < phase_bound_sec].dropna(subset=["_time_sec"])
    after_df = gpu_df[gpu_df["_time_sec"] >= phase_bound_sec].dropna(subset=["_time_sec"])

    def _mean_util(sub_df: pd.DataFrame) -> Optional[float]:
        if sub_df.empty:
            return None
        vals = sub_df[util_cols].apply(pd.to_numeric, errors="coerce").values.flatten()
        vals = vals[~pd.isna(vals)]
        return round(float(sum(vals) / len(vals)), 2) if len(vals) > 0 else None

    return _mean_util(before_df), _mean_util(after_df)


def _gpu_util_incl_excl_validation(
    gpu_df: pd.DataFrame,
    validation_intervals: List[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    GPU 메트릭 전체 평균(validation 포함)과 validation 구간 제외 평균 반환.
    validation_intervals: [(start_epoch_sec, end_epoch_sec), ...]
    Returns:
        (avg_gpu_util_incl_val_pct, avg_gpu_util_excl_val_pct)
    """
    util_cols = [c for c in gpu_df.columns if c.endswith(f".{UTIL_COL}") and "system.gpu." in c]
    if not util_cols:
        return None, None
    if "_timestamp" not in gpu_df.columns:
        return None, None

    gpu_df = gpu_df.copy()
    ser = pd.to_numeric(gpu_df["_timestamp"], errors="coerce")
    if ser.max() > 1e12:
        # ms 단위로 저장된 경우
        gpu_df["_time_sec"] = ser / 1000.0
    else:
        gpu_df["_time_sec"] = ser
    gpu_df = gpu_df.dropna(subset=["_time_sec"])

    def _mean_util(sub_df: pd.DataFrame) -> Optional[float]:
        if sub_df.empty:
            return None
        vals = sub_df[util_cols].apply(pd.to_numeric, errors="coerce").values.flatten()
        vals = vals[~pd.isna(vals)]
        return round(float(sum(vals) / len(vals)), 2) if len(vals) > 0 else None

    avg_incl = _mean_util(gpu_df)

    if not validation_intervals:
        return avg_incl, avg_incl

    def in_any_interval(t: float) -> bool:
        for start, end in validation_intervals:
            if start <= t <= end:
                return True
        return False

    mask = gpu_df["_time_sec"].apply(lambda t: not in_any_interval(t))
    excl_df = gpu_df.loc[mask]
    avg_excl = _mean_util(excl_df)
    return avg_incl, avg_excl


def summarize_experiments(
    gpu_data: Dict[str, pd.DataFrame],
    log_extra: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    실험별·GPU별 평균 utilization, 최대 memory allocated 요약.
    log_extra가 있으면 validation 구간을 제외한 avg_gpu_util_excl_val_pct도 계산.
    """
    rows = []
    for exp_name, df in gpu_data.items():
        validation_intervals = []
        if log_extra and exp_name in log_extra:
            validation_intervals = log_extra[exp_name].get("validation_intervals_epoch_sec") or []

        time_sec = None
        mask_excl_val = None
        if "_timestamp" in df.columns and validation_intervals:
            ser = pd.to_numeric(df["_timestamp"], errors="coerce")
            time_sec = ser / 1000.0 if ser.max() > 1e12 else ser

            def in_any_interval(t: float) -> bool:
                if pd.isna(t):
                    return False
                for start, end in validation_intervals:
                    if start <= t <= end:
                        return True
                return False

            mask_excl_val = time_sec.notna() & ~time_sec.apply(in_any_interval)

        for gpu_id in _extract_gpu_ids(df):
            gpu_util_col = f"system.gpu.{gpu_id}.{UTIL_COL}"
            gpu_mem_col = f"system.gpu.{gpu_id}.{MEMORY_COL}"
            if gpu_util_col not in df.columns or gpu_mem_col not in df.columns:
                continue

            util_series = pd.to_numeric(df[gpu_util_col], errors="coerce").dropna()
            mem_series = pd.to_numeric(df[gpu_mem_col], errors="coerce").dropna()

            avg_util = util_series.mean() if len(util_series) > 0 else None
            max_mem_bytes = mem_series.max() if len(mem_series) > 0 else None
            max_mem_gb = (max_mem_bytes / (1024**3)) if max_mem_bytes else None

            avg_gpu_util_incl_val_pct = round(avg_util, 2) if avg_util is not None else None
            avg_gpu_util_excl_val_pct: Optional[float] = None
            if mask_excl_val is not None and gpu_util_col in df.columns:
                excl_series = df.loc[mask_excl_val, gpu_util_col]
                excl_series = pd.to_numeric(excl_series, errors="coerce").dropna()
                if len(excl_series) > 0:
                    avg_gpu_util_excl_val_pct = round(float(excl_series.mean()), 2)
                else:
                    avg_gpu_util_excl_val_pct = avg_gpu_util_incl_val_pct
            else:
                avg_gpu_util_excl_val_pct = avg_gpu_util_incl_val_pct

            rows.append({
                "experiment": exp_name,
                "gpu_id": gpu_id,
                "avg_utilization_pct": avg_gpu_util_incl_val_pct,
                "avg_gpu_util_incl_val_pct": avg_gpu_util_incl_val_pct,
                "avg_gpu_util_excl_val_pct": avg_gpu_util_excl_val_pct,
                "max_gpu_memory_allocated_bytes": int(max_mem_bytes) if max_mem_bytes else None,
                "max_gpu_memory_allocated_gb": round(max_mem_gb, 2) if max_mem_gb else None,
            })
    return pd.DataFrame(rows)


def _avg_exclude_first(times: list) -> Optional[float]:
    """첫 번째 제외, 나머지 평균"""
    if len(times) > 1:
        return round(sum(times[1:]) / (len(times) - 1), 4)
    return None


def summarize_training(
    validation_data: Dict[str, pd.DataFrame],
    log_extra: Dict[str, dict],
    gpu_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    실험별 최종 validation_loss, validation_perplexity, e2e_training_time, 각종 avg time 요약
    avg_*: 첫 번째 제외한 나머지 평균
    gpu_data가 있고 outer opt change + phase transition 타임스탬프가 있으면 GPU utilization을 전/후 구간으로 구분해 기록.
    """
    rows = []
    for exp_name in set(validation_data.keys()) | set(log_extra.keys()):
        df = validation_data.get(exp_name)
        extra = log_extra.get(exp_name, {})

        final_loss = None
        final_ppl = None
        if df is not None and not df.empty:
            last = df.iloc[-1]
            final_loss = round(last["validation_loss"], 4)
            final_ppl = round(last["validation_perplexity"], 4)

        e2e_sec = extra.get("e2e_sec")
        gpu_times = extra.get("gpu_local_times", [])
        sync_wait_times = extra.get("sync_wait_gpu_idle_times", [])
        token_weight_times = extra.get("token_weight_sync_times", [])
        validation_sync_wait_times = extra.get("validation_sync_wait_times", [])
        allreduce_times = extra.get("allreduce_times", [])
        allreduce_before_pt = extra.get("allreduce_times_before_pt", [])
        allreduce_after_pt = extra.get("allreduce_times_after_pt", [])
        state_times = extra.get("state_averager_times", [])
        val_times = extra.get("validation_times", [])

        avg_gpu_util_before_pt: Optional[float] = None
        avg_gpu_util_after_pt: Optional[float] = None
        if gpu_data and exp_name in gpu_data and "outer opt change" in exp_name:
            phase_ts = extra.get("phase_transition_timestamp_sec")
            start_ts = extra.get("training_start_timestamp_sec")
            if phase_ts is not None:
                avg_gpu_util_before_pt, avg_gpu_util_after_pt = _gpu_util_before_after_phase_transition(
                    gpu_data[exp_name], phase_ts, start_ts
                )

        def _round_opt(v):
            return round(v, 2) if v is not None else None

        rows.append({
            "experiment": exp_name,
            "final_validation_loss": final_loss,
            "final_validation_perplexity": final_ppl,
            "e2e_training_time_sec": _round_opt(e2e_sec),
            "avg_gpu_local_training_time_sec": _avg_exclude_first(gpu_times),
            "avg_sync_wait_gpu_idle_sec": _avg_exclude_first(sync_wait_times),
            "avg_token_weight_sync_sec": _avg_exclude_first(token_weight_times),
            "avg_validation_sync_wait_sec": _avg_exclude_first(validation_sync_wait_times),
            "ppl50_incl_val_sec": _round_opt(extra.get("ppl50_incl_val_sec")),
            "ppl50_excl_val_sec": _round_opt(extra.get("ppl50_excl_val_sec")),
            "ppl50_epochs": extra.get("ppl50_epochs"),
            "ppl40_incl_val_sec": _round_opt(extra.get("ppl40_incl_val_sec")),
            "ppl40_excl_val_sec": _round_opt(extra.get("ppl40_excl_val_sec")),
            "ppl40_epochs": extra.get("ppl40_epochs"),
            "ppl30_incl_val_sec": _round_opt(extra.get("ppl30_incl_val_sec")),
            "ppl30_excl_val_sec": _round_opt(extra.get("ppl30_excl_val_sec")),
            "ppl30_epochs": extra.get("ppl30_epochs"),
            "avg_allreduce_networking_time_sec": _avg_exclude_first(allreduce_times),
            "avg_allreduce_before_phase_transition_sec": _avg_exclude_first(allreduce_before_pt) if allreduce_before_pt else None,
            "avg_allreduce_after_phase_transition_sec": _avg_exclude_first(allreduce_after_pt) if allreduce_after_pt else None,
            "avg_gpu_util_before_phase_transition_pct": avg_gpu_util_before_pt,
            "avg_gpu_util_after_phase_transition_pct": avg_gpu_util_after_pt,
            "avg_state_averager_step_sec": _avg_exclude_first(state_times),
            "avg_validation_time_sec": _avg_exclude_first(val_times),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. 실행
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WandB System 메트릭 수집 및 실험 요약")
    parser.add_argument(
        "--mode",
        choices=["fetch", "summary", "all"],
        default="all",
        help="fetch: WandB에서 raw 수집·저장만 / summary: 로컬 CSV에서 요약만 / all: 둘 다",
    )
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--projects", nargs="+", default=PROJECTS, help="처리할 프로젝트 목록 (기본: 설정의 PROJECTS)")
    parser.add_argument("--base-dir", default=".", help="wandb_logs의 상위 디렉터리")
    parser.add_argument("--output", default="experiment_gpu_summary.csv", help="GPU 요약 CSV 파일명")
    parser.add_argument("--training-output", default="experiment_training_summary.csv", help="Training 요약 CSV 파일명")
    parser.add_argument(
        "--from-csv-only",
        action="store_true",
        help="output.log를 읽지 않고, 각 실험 폴더의 기존 CSV만 사용해 요약 (로그 재파싱 없음)",
    )
    parser.add_argument(
        "--skip-fetch-if-exists",
        action="store_true",
        default=True,
        help="이미 받아 둔 파일(시스템 메트릭 CSV, output.log 등)이 있으면 해당 Run은 다시 다운로드하지 않음 (기본: True)",
    )
    parser.add_argument(
        "--no-skip-fetch-if-exists",
        action="store_false",
        dest="skip_fetch_if_exists",
        help="--skip-fetch-if-exists 해제. 모든 Run을 WandB에서 다시 다운로드",
    )
    args = parser.parse_args()

    # wandb_logs: raw data·summary가 저장되는 상위 디렉터리
    logs_dir = os.path.join(args.base_dir, WANDB_LOGS_DIR)
    os.makedirs(logs_dir, exist_ok=True)

    gpu_data: Dict[str, pd.DataFrame] = {}

    if args.mode in ("fetch", "all"):
        if args.skip_fetch_if_exists:
            print(f"WandB raw 수집 (이미 있는 파일은 스킵): {args.entity} / projects={args.projects}")
        else:
            print(f"WandB에서 raw data 수집: {args.entity} / projects={args.projects}")
        gpu_data = fetch_raw_system_metrics_from_wandb(
            args.entity, args.projects, SYSTEM_METRICS_SAMPLES, logs_dir,
            skip_if_exists=args.skip_fetch_if_exists,
        )
        print(f" -> System 메트릭: {len(gpu_data)}개 Run 처리 완료")
        print(" -> Training 메트릭(charts) 수집 중...")
        fetch_raw_training_metrics_from_wandb(
            args.entity, args.projects, logs_dir,
            skip_if_exists=args.skip_fetch_if_exists,
        )
        print(" -> Logs(output.log) 다운로드 중...")
        fetch_raw_logs_from_wandb(
            args.entity, args.projects, logs_dir,
            skip_if_exists=args.skip_fetch_if_exists,
        )
        print(f" -> 저장 완료: {logs_dir}")

    if args.mode == "summary" or (args.mode == "all" and not gpu_data):
        print(f"로컬에서 raw data 로드: {logs_dir}")
        gpu_data = fetch_raw_gpu_metrics_from_directory(logs_dir)

    # validation·timing 수집 (로그 파싱 또는 기존 CSV만 사용)
    if args.from_csv_only:
        print(" -> 기존 CSV만 사용 (output.log 미사용)...")
    else:
        print(" -> output.log 파싱 및 CSV 로드 중 (validation, e2e time, GPU local training time)...")
    validation_data, log_extra = extract_validation_results_from_logs(logs_dir, from_csv_only=args.from_csv_only)

    if not gpu_data:
        print("수집된 데이터가 없습니다.")
        exit(1)

    print(f"실험 수: {len(gpu_data)}")
    summary = summarize_experiments(gpu_data, log_extra=log_extra)
    summary_path = os.path.join(logs_dir, args.output)
    summary.to_csv(summary_path, index=False)
    print(f"GPU 요약 저장: {summary_path}")
    print(summary.to_string(index=False))

    if validation_data or log_extra:
        training_summary = summarize_training(validation_data, log_extra, gpu_data=gpu_data)
        training_path = os.path.join(logs_dir, args.training_output)
        training_summary.to_csv(training_path, index=False)
        print(f"Training 요약 저장: {training_path}")
        print(training_summary.to_string(index=False))
