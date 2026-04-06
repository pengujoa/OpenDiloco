import fsspec
from pydantic_config import BaseConfig
import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
import os
import re
from torchdata.stateful_dataloader import StatefulDataLoader
from fsspec.generic import GenericFileSystem
from hivemind.optim.optimizer import logger


GLOBAL_STATE_FILE = "global_state_dict.pt"
CKPT_PREFIX = "model_step"


def _float_to_tag(x: float) -> str:
    """Filesystem-safe: 0.5 -> 0p5, 1e-4 -> 1em4"""
    if x == 0:
        return "0"
    s = f"{x:.10g}"
    if "e" in s.lower():
        return s.lower().replace("e", "em").replace(".", "p").replace("-", "m")
    return s.replace(".", "p").replace("-", "m")


def _outer_compression_tag(hv) -> str:
    """
    Sign update + sign1bit -> outer_1bitsign (후보1 예시).
    끄면 hivemind_compression 기반 outer_fp16, outer_fp32 등.
    """
    if hv is None:
        return "outer_fp32"
    sign_on = getattr(hv, "outer_sign_update", False)
    comp = getattr(hv, "hivemind_compression", None)
    if sign_on and comp == "sign1bit":
        return "outer_1bitsign"
    if comp is None:
        return "outer_fp32"
    # fp16, scaled-fp16, uniform8bit, ...
    m = {
        "fp16": "outer_fp16",
        "scaled-fp16": "outer_scaled_fp16",
        "uniform8bit": "outer_uniform8bit",
        "quantile8bit": "outer_quantile8bit",
        "blockwise8bit": "outer_blockwise8bit",
        "sign1bit": "outer_1bitsign",
    }
    return m.get(comp, "outer_" + comp.replace("-", "_"))


def _tw_tag(hv) -> str:
    if hv is None or not getattr(hv, "token_weighted_aggregation", False):
        return "tw_off"
    mode = getattr(hv, "token_weight_mode", "linear")
    return f"tw_{mode}"


def get_ckpt_experiment_dir_name(config) -> str:
    """
    하나의 설정을 나타내는 디렉터리명 한 덩어리 (후보1 형식).

    예: inner_lion_outer_1bitsign_adaptive_mean_outerlr0p5_majority_vote_constant_warm0_tw_sqrt_maxouter500
    """
    inner = getattr(config, "inner_optimizer_type", "adamw")
    hv = getattr(config, "hv", None)

    parts = [f"inner_{inner}", _outer_compression_tag(hv)]

    if hv is not None:
        # adaptive_mean은 하나의 토큰으로 (outer_sign_mode)
        outer_sign_mode = getattr(hv, "outer_sign_mode", "fixed_lr")
        parts.append(outer_sign_mode)

        outer_lr = getattr(hv, "outer_lr", 0.7)
        parts.append(f"outerlr{_float_to_tag(outer_lr)}")

        agg = getattr(hv, "outer_sign_aggregation", "majority_vote")
        parts.append(agg)

        sched = getattr(hv, "outer_lr_scheduler_type", "constant")
        parts.append(sched)

        warm = getattr(hv, "outer_warmup_steps", 0)
        parts.append(f"warm{warm}")

        parts.append(_tw_tag(hv))

        max_outer = getattr(hv, "max_outer_optimization_steps", None)
        if max_outer is not None:
            parts.append(f"maxouter{int(max_outer)}")
        else:
            parts.append("maxouter_none")
    else:
        parts.append("fixed_lr")
        parts.append(f"outerlr{_float_to_tag(getattr(config, 'lr', 4e-4))}")
        parts.append("majority_vote")
        parts.append("constant")
        parts.append("warm0")
        parts.append("tw_off")
        parts.append("maxouter_none")

    name = "_".join(parts)
    # 파일시스템 안전: 나머지 특수문자 제거
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return name


def get_ckpt_base_path(config) -> str:
    """config.ckpt.path 아래 설정 ID 한 덩어리 디렉터리 = 실험 베이스. 그 안에 model_step_* / diloco_rank_* 유지."""
    return os.path.join(config.ckpt.path, get_ckpt_experiment_dir_name(config))


def get_parameter_tracking_log_dir(config) -> str:
    """
    parameter_tracking_logs를 체크포인트와 동일한 실험 ID 아래에 두어 CSV/JSON이 실험별로 구분되게 함.
    hivemind_diloco DiLoCoGradAverager.log_dir 으로 전달.
    """
    return os.path.join(get_ckpt_base_path(config), "parameter_tracking_logs")


def should_save_final_checkpoint(real_step: int, interval: int | None) -> bool:
    """
    학습 루프 종료 직후, 마지막 스텝이 interval 저장으로 이미 저장되지 않았을 때
    한 번 더 저장할지 여부. interval이 None이면 루프 중 저장이 없었으므로
    real_step > 0이면 최종 저장 권장.
    """
    if real_step <= 0:
        return False
    if interval is None:
        return True
    return real_step % interval != 0


class CkptConfig(BaseConfig):
    resume: str | bool | None = None  # if resume is a boolean, it means we should resume from the last checkpoint
    interval: int | None = None
    path: str = "outputs"
    topk: int | None = None  # how many checkpoints to keep


def get_resume_info(config) -> tuple[bool, str | None]:
    """
    config 전체를 받아 실험 베이스(config.ckpt.path/설정ID/) 아래에서
    최신 model_step_* 디렉터리를 찾는다.
    resume이 문자열이면 기존처럼 그 경로를 그대로 반환.
    """
    ckpt_config = config.ckpt
    if ckpt_config.resume is None:
        return False, None
    if isinstance(ckpt_config.resume, str):
        return True, ckpt_config.resume

    # resume is True: 실험 베이스 아래에서 최신 model_step_* 검색
    base = get_ckpt_base_path(config)
    fs = GenericFileSystem()
    try:
        # base가 없으면 path만 나열 시도 (구버전 평탄 구조)
        if not fs.exists(base):
            ckpt_files = [f for f in fs.ls(ckpt_config.path, detail=False) if filter_ckpt_files(f)]
        else:
            ckpt_files = []
            for f in fs.ls(base, detail=False):
                if filter_ckpt_files(f):
                    ckpt_files.append(f)
            if not ckpt_files:
                # 평탄 구조 fallback
                ckpt_files = [f for f in fs.ls(ckpt_config.path, detail=False) if filter_ckpt_files(f)]
    except FileNotFoundError:
        logger.info(f"Checkpoint path {ckpt_config.path} not found, starting from scratch")
        return False, None

    if len(ckpt_files) == 0:
        logger.info(f"No checkpoints found under {base} (or {ckpt_config.path}), starting from scratch")
        return False, None

    def _step_key(path: str) -> int:
        try:
            return int(path.rstrip("/").split("_")[-1])
        except ValueError:
            return 0

    latest_ckpt = max(ckpt_files, key=_step_key)
    return True, latest_ckpt


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    outer_optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    loss: float | None = None,
    data_loader: StatefulDataLoader | None = None,
    save_global_state: bool = True,
):
    """Save the model and optimizer state to a checkpoint folderx

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to save
        optimizer: the optimizer to save
        scheduler: the scheduler to save
        outer_optimizer: the outer optimizer to save
        loss: the loss to save
        data_loader: the data loader to save
        save_global_state: whether to save the global state
    """
    rank = int(os.environ["RANK"])
    os.makedirs(checkpoint_path, exist_ok=True)
    abs_ckpt = os.path.abspath(checkpoint_path)

    def _log_save_done(note: str = "") -> None:
        try:
            nfiles = sum(len(files) for _, _, files in os.walk(abs_ckpt))
            logger.info(
                f"Checkpoint save finished{note}: {abs_ckpt} ({nfiles} files under this directory)"
            )
        except OSError:
            logger.info(f"Checkpoint save finished{note}: {abs_ckpt}")

    # 1. Save distributed states
    if hasattr(dcp, "FsspecWriter"):
        fs_storage_writer = dcp.FsspecWriter(checkpoint_path, sync_files=False)
        # for some reason sync_files = True try to call stream.fileno which is not supported with gcp ffspec storage.
    else:
        # PyTorch 2.4 uses FileSystemWriter/FileSystemReader
        fs_storage_writer = dcp.FileSystemWriter(checkpoint_path, sync_files=False)

    try:
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        dcp_state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }
        dcp.save(dcp_state_dict, storage_writer=fs_storage_writer)
    except Exception as e:
        logger.warning(f"DCP get_state_dict failed ({e}), falling back to plain state_dict")
        dcp_state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        dcp.save(dcp_state_dict, storage_writer=fs_storage_writer)
    if data_loader is not None:
        rank_state_dict = {}
        rank_state_dict["data_loader"] = data_loader.state_dict()
        with fsspec.open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "wb") as f:
            torch.save(rank_state_dict, f)

    if not save_global_state:
        _log_save_done(" (DCP/dataloader only; global_state_dict.pt on rank 0 / messenger ranks)")
        return

    # 2. Save global states
    global_state_dict = {"scheduler": scheduler.state_dict(), "loss": loss if loss is not None else 0}
    if outer_optimizer is not None:
        global_state_dict["outer_optimizer"] = outer_optimizer.state_dict()
    if scaler is not None:
        global_state_dict["scaler"] = scaler.state_dict()

    with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "wb") as f:
        torch.save(global_state_dict, f)

    _log_save_done()


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    data_loader: StatefulDataLoader | None = None,
    lora: bool | None = False,
    dataset: str | None = "allenai/c4"
) -> float:
    """Load the model and optimizer state from a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
        outer_optimizer: the outer optimizer to load
        data_loader: the data loader to load

    Returns:
        loss: the loss from the checkpoint
    """
    rank = int(os.environ["RANK"])
    # 1. Load distributed states
    if hasattr(dcp, "FsspecReader"):
        fs_storage_reader = dcp.FsspecReader(checkpoint_path)
    else:
        fs_storage_reader = dcp.FileSystemReader(checkpoint_path)

    try:
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    except Exception as e:
        logger.warning(f"DCP get_state_dict failed ({e}), falling back to plain state_dict")
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

    if lora:
        model_state_dict_cnvt = {}
        optimizer_state_dict_cnvt = {"state":{}, "param_groups":[]}
        for key, value in model_state_dict.items():
            if "lora" in key:
                pass
            else:
                model_state_dict_cnvt[key] = value

        dcp_state_dict = {
            "model": model_state_dict_cnvt,
        }
    else:
        dcp_state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

    dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)
    try:
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict,
        )
    except Exception as e:
        logger.warning(f"DCP set_state_dict failed ({e}), falling back to load_state_dict")
        model.load_state_dict(dcp_state_dict.get("model", model_state_dict), strict=False)
        if "optimizer" in dcp_state_dict:
            optimizer.load_state_dict(dcp_state_dict["optimizer"])
    if lora & (dataset!="allenai/c4"):
        pass
    else:
        if data_loader is not None:
            with fsspec.open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "rb") as f:
                rank_state_dict = torch.load(f)
            data_loader.load_state_dict(rank_state_dict["data_loader"])

    if lora:
        return 0
    else:
        # 2. Load global states
        with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "rb") as f:
            global_state_dict = torch.load(f)
        if scheduler is not None:
            scheduler.load_state_dict(global_state_dict["scheduler"])
            optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
        if outer_optimizer is not None:
            outer_optimizer.load_state_dict(global_state_dict["outer_optimizer"])
        if scaler is not None:
            scaler.load_state_dict(global_state_dict["scaler"])
        return global_state_dict["loss"]


def filter_ckpt_files(f):
    if CKPT_PREFIX not in f:
        return False
    else:
        try:
            int(f.split("_")[-1])
            return True
        except ValueError:
            return False


def delete_old_checkpoints(checkpoint_path: str, topk: int) -> list[str]:
    fs = GenericFileSystem()
    ckpt_files = [f for f in fs.ls(checkpoint_path, detail=False) if filter_ckpt_files(f)]
    ckpt_files.sort(key=lambda x: int(x.split("_")[-1]))

    ckpt_deleted = []
    for ckpt_file in ckpt_files[:-topk]:
        fs.rm(ckpt_file, recursive=True)
        ckpt_deleted.append(ckpt_file)
    return ckpt_deleted


def check_checkpoint_path_access(checkpoint_path: str, rank: int, world_rank_hv: int | None = None):
    if world_rank_hv is not None:
        dummy_file_path = os.path.join(
            checkpoint_path, get_diloco_rank_dir_name(world_rank_hv), f"dummy_file_{rank}.txt"
        )
    else:
        dummy_file_path = os.path.join(checkpoint_path, f"dummy_file_{rank}.txt")

    with fsspec.open(dummy_file_path, "w") as f:
        f.write("This is a dummy file for testing access.")
    gfs = GenericFileSystem()
    gfs.rm(dummy_file_path)


def get_diloco_rank_dir_name(world_rank_diloco: int) -> str:
    return f"diloco_rank_{world_rank_diloco}"
