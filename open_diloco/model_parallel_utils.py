"""
Utilities for Model Parallel (Tensor Parallel / Pipeline Parallel) training
with DiLoCo.

Provides:
- TP/PP model wrapping for LlamaForCausalLM
- Full model copy creation on CPU (for DiLoCo outer optimizer)
- Parameter gather (TP/PP shards -> full model) and scatter (full model -> shards)
- FSDP API replacements (clip_grad_norm_, no_sync)
"""

import copy
import logging
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor Parallel helpers
# ---------------------------------------------------------------------------

def _build_llama_tp_plan() -> Dict[str, object]:
    """
    Build a tensor-parallel plan for LlamaForCausalLM.

    ColwiseParallel: shard output dim  (weight cols)
    RowwiseParallel: shard input dim   (weight rows)

    Llama attention:  q/k/v_proj  -> ColwiseParallel
                      o_proj      -> RowwiseParallel
    Llama MLP:        gate/up_proj -> ColwiseParallel
                      down_proj    -> RowwiseParallel
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from torch.distributed._tensor import Replicate

    plan: Dict[str, object] = {}
    plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())

    return plan


def _build_llama_layer_tp_plan() -> Dict[str, object]:
    """Per-decoder-layer TP plan."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    return {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }


def wrap_model_tp(
    model: nn.Module,
    device_mesh: "torch.distributed.device_mesh.DeviceMesh",
    tp_mesh_dim: str = "tp",
) -> nn.Module:
    """
    Apply Tensor Parallelism to a LlamaForCausalLM model using
    ``torch.distributed.tensor.parallel.parallelize_module``.

    Returns the same model object (mutated in-place) with DTensor parameters.
    """
    from torch.distributed.tensor.parallel import parallelize_module

    top_plan = _build_llama_tp_plan()
    layer_plan = _build_llama_layer_tp_plan()

    parallelize_module(model, device_mesh[tp_mesh_dim], top_plan)

    for layer in model.model.layers:
        parallelize_module(layer, device_mesh[tp_mesh_dim], layer_plan)

    logger.info("Model wrapped with Tensor Parallelism")
    return model


# ---------------------------------------------------------------------------
# Pipeline Parallel helpers
# ---------------------------------------------------------------------------

def wrap_model_pp(
    model: nn.Module,
    num_stages: int,
    stage_index: int,
    device: torch.device,
) -> Tuple[nn.Module, object]:
    """
    Split a LlamaForCausalLM into pipeline stages.

    Returns (stage_module, PipelineStage) where stage_module contains only
    the layers assigned to ``stage_index``.

    Requires PyTorch >= 2.4 for ``torch.distributed.pipelining``.
    """
    try:
        from torch.distributed.pipelining import PipelineStage
    except ModuleNotFoundError:
        raise RuntimeError(
            "Pipeline Parallelism requires PyTorch >= 2.4. "
            f"Current version: {torch.__version__}. "
            "Please upgrade PyTorch or use --parallelism tp instead."
        )

    num_layers = len(model.model.layers)
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages

    boundaries: List[int] = []
    start = 0
    for i in range(num_stages):
        end = start + layers_per_stage + (1 if i < remainder else 0)
        boundaries.append((start, end))
        start = end

    layer_start, layer_end = boundaries[stage_index]

    keep_indices = set(range(layer_start, layer_end))
    drop_indices = sorted(set(range(num_layers)) - keep_indices, reverse=True)
    for idx in drop_indices:
        del model.model.layers[idx]

    if stage_index != 0:
        model.model.embed_tokens = None
    if stage_index != num_stages - 1:
        model.lm_head = None
        model.model.norm = None

    model = model.to(device)

    stage = PipelineStage(model, stage_index, num_stages, device)
    logger.info(
        f"Pipeline stage {stage_index}/{num_stages}: "
        f"layers [{layer_start}, {layer_end})"
    )
    return model, stage


# ---------------------------------------------------------------------------
# Full model copy on CPU (for DiLoCo outer optimizer)
# ---------------------------------------------------------------------------

def create_full_model_copy(model: nn.Module, device: str = "cpu") -> nn.Module:
    """
    Create an independent full copy of *model* on *device*.

    The copy shares no storage with the original so that updates to one do
    not affect the other.
    """
    full_model = copy.deepcopy(model)
    full_model = full_model.to(device)
    full_model.requires_grad_(False)
    logger.info(f"Created full model copy on {device}")
    return full_model


# ---------------------------------------------------------------------------
# Parameter gather / scatter  (TP)
# ---------------------------------------------------------------------------

def _is_dtensor(t: torch.Tensor) -> bool:
    return hasattr(t, "full_tensor")


def gather_tp_params(
    tp_model: nn.Module,
    full_model: nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Gather sharded TP parameters into the full model on CPU.

    For DTensor parameters, ``full_tensor()`` materialises the full view;
    for plain tensors the value is copied directly.
    """
    tp_params = dict(tp_model.named_parameters())
    for name, full_param in full_model.named_parameters():
        tp_param = tp_params.get(name)
        if tp_param is None:
            continue
        if _is_dtensor(tp_param):
            gathered = tp_param.full_tensor().detach()
        else:
            gathered = tp_param.detach()
        full_param.data.copy_(gathered.to(full_param.device), non_blocking=True)


def scatter_tp_params(
    full_model: nn.Module,
    tp_model: nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Distribute full-model parameters back into TP shards.

    For DTensor parameters we write only the local shard; for plain tensors
    we copy the whole value.
    """
    full_params = dict(full_model.named_parameters())
    for name, tp_param in tp_model.named_parameters():
        full_param = full_params.get(name)
        if full_param is None:
            continue

        src = full_param.data
        if _is_dtensor(tp_param):
            local_tensor = tp_param._local_tensor
            placement = tp_param.placements[0]
            mesh = tp_param.device_mesh
            local_rank = mesh.get_local_rank()
            tp_size = mesh.size()

            chunk_dim = _get_shard_dim(placement)
            if chunk_dim is not None:
                chunks = src.to(local_tensor.device).chunk(tp_size, dim=chunk_dim)
                local_tensor.copy_(chunks[local_rank], non_blocking=True)
            else:
                local_tensor.copy_(src.to(local_tensor.device), non_blocking=True)
        else:
            tp_param.data.copy_(src.to(tp_param.device), non_blocking=True)


def _get_shard_dim(placement) -> Optional[int]:
    """Extract the shard dimension from a DTensor placement."""
    if hasattr(placement, "dim"):
        return placement.dim
    return None


def broadcast_tp_params(
    tp_model: nn.Module,
    full_model: Optional[nn.Module] = None,
    src_rank: int = 0,
) -> None:
    """
    Broadcast updated parameters from *src_rank* to all TP ranks.

    Rank *src_rank* reads the full parameter from *full_model*, broadcasts
    it, then each rank extracts its own shard.  For non-DTensor parameters
    a simple broadcast is used.
    """
    full_params = dict(full_model.named_parameters()) if full_model is not None else {}
    rank = dist.get_rank()

    for name, tp_param in tp_model.named_parameters():
        if _is_dtensor(tp_param):
            local_tensor = tp_param._local_tensor
            placement = tp_param.placements[0]
            mesh = tp_param.device_mesh
            local_rank_in_mesh = mesh.get_local_rank()
            tp_size = mesh.size()

            full_data = torch.empty(
                tp_param.shape, device=local_tensor.device, dtype=local_tensor.dtype
            )
            if rank == src_rank:
                fp = full_params.get(name)
                if fp is not None:
                    full_data.copy_(fp.data.to(full_data.device))

            dist.broadcast(full_data, src=src_rank)

            chunk_dim = _get_shard_dim(placement)
            if chunk_dim is not None:
                chunks = full_data.chunk(tp_size, dim=chunk_dim)
                local_tensor.copy_(chunks[local_rank_in_mesh])
            else:
                local_tensor.copy_(full_data)
        else:
            if rank == src_rank:
                fp = full_params.get(name)
                if fp is not None:
                    tp_param.data.copy_(fp.data.to(tp_param.device))
            dist.broadcast(tp_param.data, src=src_rank)


# ---------------------------------------------------------------------------
# Parameter gather / scatter  (PP)
# ---------------------------------------------------------------------------

def gather_pp_params(
    pp_model: nn.Module,
    full_model: nn.Module,
    stage_index: int,
    num_stages: int,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Gather PP stage parameters into *full_model* on rank 0.

    Each rank sends its local stage parameters; rank 0 places them into the
    correct positions in full_model.
    """
    rank = dist.get_rank(process_group)

    pp_state = {}
    for name, param in pp_model.named_parameters():
        pp_state[name] = param.detach().cpu()

    gathered_states: Optional[List] = None
    if rank == 0:
        gathered_states = [None] * num_stages

    dist.gather_object(
        (stage_index, pp_state),
        object_gather_list=gathered_states,
        dst=0,
        group=process_group,
    )

    if rank == 0 and gathered_states is not None:
        for stage_idx, state in gathered_states:
            for name, value in state.items():
                full_param = dict(full_model.named_parameters()).get(name)
                if full_param is not None:
                    full_param.data.copy_(value, non_blocking=True)


def scatter_pp_params(
    full_model: nn.Module,
    pp_model: nn.Module,
    stage_index: int,
    num_stages: int,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Scatter full-model parameters back to each PP stage.

    Rank 0 sends each stage its relevant parameters; other ranks receive.
    """
    rank = dist.get_rank(process_group)

    pp_param_names = [name for name, _ in pp_model.named_parameters()]

    if rank == 0:
        scatter_list = [None] * num_stages
        full_params = dict(full_model.named_parameters())
        scatter_list[stage_index] = {
            name: full_params[name].detach().cpu()
            for name in pp_param_names
            if name in full_params
        }
        for other_stage in range(num_stages):
            if other_stage == stage_index:
                continue
            scatter_list[other_stage] = {}
    else:
        scatter_list = None

    received = [None]
    dist.scatter_object_list(received, scatter_list, src=0, group=process_group)

    if received[0]:
        for name, value in received[0].items():
            param = dict(pp_model.named_parameters()).get(name)
            if param is not None:
                param.data.copy_(value.to(param.device), non_blocking=True)


# ---------------------------------------------------------------------------
# FSDP API replacements
# ---------------------------------------------------------------------------

def _collect_local_grads(model: nn.Module) -> List[torch.Tensor]:
    """
    Collect gradient tensors from all parameters, extracting
    ``_local_tensor`` from DTensors so that low-level CUDA ops
    (GradScaler, clip_grad_norm) can work without DTensor dispatch.
    """
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        if _is_dtensor(p.grad):
            grads.append(p.grad._local_tensor)
        else:
            grads.append(p.grad)
    return grads


def mp_scaler_unscale_(
    scaler: "torch.cuda.amp.GradScaler",
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> None:
    """
    GradScaler.unscale_() replacement that works with DTensor gradients.

    Directly calls the underlying CUDA unscale op on local grad tensors,
    then registers the result with the scaler so that ``scaler.step()``
    and ``scaler.update()`` work correctly afterwards.
    """
    from torch.amp.grad_scaler import OptState

    opt_id = id(optimizer)
    if opt_id not in scaler._per_optimizer_states:
        scaler._per_optimizer_states[opt_id] = {
            "stage": OptState.READY,
            "found_inf_per_device": {},
        }

    opt_state = scaler._per_optimizer_states[opt_id]
    if opt_state["stage"] is OptState.UNSCALED:
        return

    grads = _collect_local_grads(model)

    if not grads:
        opt_state["stage"] = OptState.UNSCALED
        return

    device = grads[0].device
    inv_scale = scaler._scale.double().reciprocal().float()
    found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=device)

    torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)

    opt_state["found_inf_per_device"] = {device: found_inf}
    opt_state["stage"] = OptState.UNSCALED


def mp_clip_grad_norm_(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Gradient clipping for TP/PP models.
    Operates on local grad tensors to avoid DTensor dispatch issues.
    """
    grads = _collect_local_grads(model)
    if not grads:
        return torch.tensor(0.0)

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float("inf"):
        total_norm = max(g.detach().abs().max() for g in grads)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
            norm_type,
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)

    return total_norm


@contextmanager
def mp_no_sync(model: nn.Module):
    """
    Context manager replacing FSDP/DDP ``model.no_sync()``.

    For DDP-wrapped models, delegates to the real ``no_sync()``.
    For plain models or TP models, gradient accumulation works automatically
    so this is a no-op.
    """
    if hasattr(model, "no_sync"):
        with model.no_sync():
            yield
    else:
        yield
