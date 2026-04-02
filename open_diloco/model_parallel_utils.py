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


def _build_llama_layer_tp_plan(model: nn.Module, tp_world_size: int) -> Dict[str, object]:
    """Per-decoder-layer TP plan with model-structure/head-divisibility safeguards."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    plan: Dict[str, object] = {}
    sample_layer = model.model.layers[0]

    # MLP projection names differ by architecture.
    # Llama-family: gate_proj + up_proj + down_proj
    # Phi3-family: gate_up_proj + down_proj
    if hasattr(sample_layer, "mlp"):
        mlp = sample_layer.mlp
        if hasattr(mlp, "gate_proj"):
            plan["mlp.gate_proj"] = ColwiseParallel()
        if hasattr(mlp, "up_proj"):
            plan["mlp.up_proj"] = ColwiseParallel()
        if hasattr(mlp, "gate_up_proj"):
            plan["mlp.gate_up_proj"] = ColwiseParallel()
        if hasattr(mlp, "down_proj"):
            plan["mlp.down_proj"] = RowwiseParallel()

    cfg = getattr(model, "config", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    num_key_value_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)

    q_heads_ok = (
        isinstance(num_attention_heads, int)
        and num_attention_heads > 0
        and num_attention_heads % tp_world_size == 0
    )
    kv_heads_ok = (
        isinstance(num_key_value_heads, int)
        and num_key_value_heads > 0
        and num_key_value_heads % tp_world_size == 0
    )

    # Keep the whole attention TP plan consistent.
    # Partial sharding can produce projection matmul dim mismatch.
    if q_heads_ok and kv_heads_ok:
        if hasattr(sample_layer, "self_attn"):
            attn = sample_layer.self_attn
            # Llama/Gemma style split q,k,v projections
            if all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
                plan["self_attn.q_proj"] = ColwiseParallel()
                plan["self_attn.k_proj"] = ColwiseParallel()
                plan["self_attn.v_proj"] = ColwiseParallel()
                plan["self_attn.o_proj"] = RowwiseParallel()
            # Phi3 style fused qkv projection
            elif all(hasattr(attn, name) for name in ("qkv_proj", "o_proj")):
                plan["self_attn.qkv_proj"] = ColwiseParallel()
                plan["self_attn.o_proj"] = RowwiseParallel()
            else:
                logger.warning("Unknown attention projection layout; skipping attention TP sharding.")
    else:
        logger.warning(
            "Skipping TP sharding for all attention projections: "
            "num_attention_heads=%s, num_key_value_heads=%s, tp_world_size=%s",
            num_attention_heads,
            num_key_value_heads,
            tp_world_size,
        )

    return plan


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
    tp_world_size = int(device_mesh[tp_mesh_dim].size())
    layer_plan = _build_llama_layer_tp_plan(model, tp_world_size)

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
    microbatch_size: int,
    seq_len: int,
    activation_dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, object]:
    """
    Split a causal LM into pipeline stages and build a manual
    ``torch.distributed.pipelining.PipelineStage`` for ``ScheduleGPipe``.

    Returns ``(stage_module, pipeline_stage)`` where ``stage_module`` is the
    per-rank ``nn.Module`` and ``pipeline_stage`` is the pipelining wrapper.

    Supports:
    - Llama-style causal LM (e.g. Llama, Mistral)
    - Gemma3 text causal LM (e.g. google/gemma-3-270m)
    """
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None) if cfg is not None else None

    if model_type in ("gemma3", "gemma3_text"):
        try:
            from gemma3_pp_stages import (
                build_pipeline_stage,
                is_gemma3_pp_supported,
                slice_gemma3_for_pp,
            )
        except ImportError:
            from open_diloco.gemma3_pp_stages import (
                build_pipeline_stage,
                is_gemma3_pp_supported,
                slice_gemma3_for_pp,
            )
        if not is_gemma3_pp_supported(model):
            raise RuntimeError("Model config indicates gemma3 but Gemma3 PP checks failed.")
        model = model.to(device)
        stage_module, (layer_start, layer_end) = slice_gemma3_for_pp(
            model, num_stages, stage_index, activation_dtype=activation_dtype
        )
        stage_module = stage_module.to(device)
        pipeline_stage = build_pipeline_stage(
            stage_module,
            stage_index,
            num_stages,
            device,
            microbatch_size=microbatch_size,
            seq_len=seq_len,
            activation_dtype=activation_dtype,
        )
        logger.info(
            f"Pipeline stage {stage_index}/{num_stages}: "
            f"layers [{layer_start}, {layer_end}) (Gemma3 / ScheduleGPipe)"
        )
        return stage_module, pipeline_stage

    try:
        from llama_pp_stages import (
            build_pipeline_stage,
            is_llama_like_pp_supported,
            slice_llama_like_model_for_pp,
        )
    except ImportError:
        from open_diloco.llama_pp_stages import (
            build_pipeline_stage,
            is_llama_like_pp_supported,
            slice_llama_like_model_for_pp,
        )

    if not is_llama_like_pp_supported(model):
        raise RuntimeError(
            "parallelism='pp' only supports Llama-style or Gemma3-style causal LMs in this repo. "
            f"Got model_type={model_type!r}. Use parallelism='tp' or add a model-specific PP implementation."
        )

    model = model.to(device)
    stage_module, (layer_start, layer_end) = slice_llama_like_model_for_pp(
        model, num_stages, stage_index, activation_dtype=activation_dtype
    )
    stage_module = stage_module.to(device)

    pipeline_stage = build_pipeline_stage(
        stage_module,
        stage_index,
        num_stages,
        device,
        microbatch_size=microbatch_size,
        seq_len=seq_len,
        activation_dtype=activation_dtype,
    )

    logger.info(
        f"Pipeline stage {stage_index}/{num_stages}: "
        f"layers [{layer_start}, {layer_end}) (ScheduleGPipe / PipelineStage)"
    )
    return stage_module, pipeline_stage


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

    All ranks must call this (collective). Rank 0 gathers each stage's
    param names, then scatters the corresponding tensors from full_model.
    """
    rank = dist.get_rank(process_group)

    my_param_names = [name for name, _ in pp_model.named_parameters()]
    all_names: Optional[List] = [None] * num_stages if rank == 0 else None
    dist.gather_object(my_param_names, object_gather_list=all_names, dst=0, group=process_group)

    if rank == 0:
        scatter_list: List = [None] * num_stages
        full_params = dict(full_model.named_parameters())
        for stage_i in range(num_stages):
            names = all_names[stage_i] if all_names[stage_i] else []
            scatter_list[stage_i] = {
                name: full_params[name].detach().cpu()
                for name in names
                if name in full_params
            }
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
