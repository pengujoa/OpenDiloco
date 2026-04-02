"""
Llama-family pipeline stages for torch.distributed.pipelining (manual PipelineStage).

Each stage exposes a forward that matches pipelining expectations:
- First stage:  (input_ids, attention_mask) -> (hidden_states, attention_mask)
- Middle/last:  (hidden_states, attention_mask) -> same tuple or logits (last)

Requires transformers with Llama-style modeling (create_causal_mask, LlamaModel layout).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask


def causal_lm_shifted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for causal LM (shifted), HF-style ignore_index=-100."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _prep_embeds_and_mask(
    config,
    hidden_or_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
):
    """Shared mask/position prep matching LlamaModel (training, no KV cache)."""
    inputs_embeds = hidden_or_embeds
    cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(
        config=config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    return inputs_embeds, cache_position, position_ids, causal_mask


class LlamaFamilyPPFirst(nn.Module):
    """First PP stage: token ids -> partial decoder stack."""

    def __init__(self, causal_lm: nn.Module, activation_dtype: torch.dtype):
        super().__init__()
        self.config = causal_lm.config
        self.backbone = causal_lm.model
        self.embed_tokens = self.backbone.embed_tokens
        self.rotary_emb = self.backbone.rotary_emb
        self.layers = self.backbone.layers
        self.activation_dtype = activation_dtype

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask_f = attention_mask.to(dtype=torch.float32)
        inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = None
        inputs_embeds, cache_position, position_ids, causal_mask = _prep_embeds_and_mask(
            self.config, inputs_embeds, attention_mask, past_key_values
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states.to(dtype=self.activation_dtype), attention_mask_f


class LlamaFamilyPPMiddle(nn.Module):
    """Middle PP stage: hidden states -> hidden states."""

    def __init__(self, causal_lm: nn.Module, activation_dtype: torch.dtype):
        super().__init__()
        self.config = causal_lm.config
        self.backbone = causal_lm.model
        self.rotary_emb = self.backbone.rotary_emb
        self.layers = self.backbone.layers
        self.activation_dtype = activation_dtype

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask_l = attention_mask.to(dtype=torch.long)
        hidden_states = hidden_states + attention_mask.to(hidden_states.dtype).unsqueeze(-1) * 0.0
        past_key_values = None
        hidden_states, cache_position, position_ids, causal_mask = _prep_embeds_and_mask(
            self.config, hidden_states, attention_mask_l, past_key_values
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states.to(dtype=self.activation_dtype), attention_mask


class LlamaFamilyPPLast(nn.Module):
    """Last PP stage: hidden states -> logits (loss applied in ScheduleGPipe via loss_fn)."""

    def __init__(self, causal_lm: nn.Module, activation_dtype: torch.dtype):
        super().__init__()
        self.config = causal_lm.config
        self.backbone = causal_lm.model
        self.rotary_emb = self.backbone.rotary_emb
        self.layers = self.backbone.layers
        self.norm = self.backbone.norm
        self.lm_head = causal_lm.lm_head
        self.activation_dtype = activation_dtype

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask_l = attention_mask.to(dtype=torch.long)
        hidden_states = hidden_states + attention_mask.to(hidden_states.dtype).unsqueeze(-1) * 0.0
        past_key_values = None
        hidden_states, cache_position, position_ids, causal_mask = _prep_embeds_and_mask(
            self.config, hidden_states, attention_mask_l, past_key_values
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).to(dtype=self.activation_dtype)
        return logits


def _balanced_pp_boundaries(
    num_layers: int, num_stages: int
) -> list[tuple[int, int]]:
    """Distribute extra layers to middle stages to compensate for
    embed_tokens (first stage) and lm_head (last stage) memory overhead."""
    base = num_layers // num_stages
    remainder = num_layers % num_stages
    extra = [0] * num_stages
    left = remainder
    for i in range(1, num_stages - 1):
        if left <= 0:
            break
        extra[i] = 1
        left -= 1
    for i in range(num_stages - 1, -1, -1):
        if left <= 0:
            break
        if extra[i] == 0:
            extra[i] = 1
            left -= 1
    boundaries: list[tuple[int, int]] = []
    s = 0
    for i in range(num_stages):
        e = s + base + extra[i]
        boundaries.append((s, e))
        s = e
    return boundaries


def slice_llama_like_model_for_pp(
    causal_lm: nn.Module,
    num_stages: int,
    stage_index: int,
    activation_dtype: torch.dtype,
) -> Tuple[nn.Module, Tuple[int, int]]:
    """
    In-place: keep only layers/embed/norm/lm_head needed for this pipeline stage.
    Returns (stage_module, (layer_start, layer_end)) for logging.
    """
    num_layers = len(causal_lm.model.layers)
    boundaries = _balanced_pp_boundaries(num_layers, num_stages)

    layer_start, layer_end = boundaries[stage_index]
    keep = set(range(layer_start, layer_end))
    drop_indices = sorted(set(range(num_layers)) - keep, reverse=True)
    for idx in drop_indices:
        del causal_lm.model.layers[idx]

    if stage_index != 0:
        causal_lm.model.embed_tokens = None
    if stage_index != num_stages - 1:
        causal_lm.lm_head = None
        causal_lm.model.norm = None

    if stage_index == 0:
        submodule = LlamaFamilyPPFirst(causal_lm, activation_dtype=activation_dtype)
    elif stage_index == num_stages - 1:
        submodule = LlamaFamilyPPLast(causal_lm, activation_dtype=activation_dtype)
    else:
        submodule = LlamaFamilyPPMiddle(causal_lm, activation_dtype=activation_dtype)

    return submodule, (layer_start, layer_end)


def build_pipeline_stage(
    stage_module: nn.Module,
    stage_index: int,
    num_stages: int,
    device: torch.device,
    microbatch_size: int,
    seq_len: int,
    activation_dtype: torch.dtype = torch.float32,
) -> "PipelineStage":
    from torch.distributed.pipelining import PipelineStage

    if stage_index == 0:
        input_args = (
            torch.zeros(microbatch_size, seq_len, dtype=torch.long, device=device),
            torch.zeros(microbatch_size, seq_len, dtype=torch.long, device=device),
        )
        hidden = torch.zeros(
            microbatch_size, seq_len, stage_module.config.hidden_size,
            dtype=activation_dtype, device=device,
        )
        output_args = (hidden, torch.zeros(microbatch_size, seq_len, dtype=torch.float32, device=device))
    elif stage_index == num_stages - 1:
        cfg = stage_module.config
        input_args = (
            torch.zeros(microbatch_size, seq_len, cfg.hidden_size, dtype=activation_dtype, device=device),
            torch.zeros(microbatch_size, seq_len, dtype=torch.float32, device=device),
        )
        output_args = torch.zeros(
            microbatch_size, seq_len, cfg.vocab_size, dtype=activation_dtype, device=device
        )
    else:
        cfg = stage_module.config
        input_args = (
            torch.zeros(microbatch_size, seq_len, cfg.hidden_size, dtype=activation_dtype, device=device),
            torch.zeros(microbatch_size, seq_len, dtype=torch.float32, device=device),
        )
        hidden = torch.zeros(
            microbatch_size, seq_len, cfg.hidden_size, dtype=activation_dtype, device=device
        )
        output_args = (hidden, torch.zeros(microbatch_size, seq_len, dtype=torch.float32, device=device))

    return PipelineStage(
        stage_module,
        stage_index,
        num_stages,
        device,
        input_args,
        output_args,
    )


def is_llama_like_pp_supported(model: nn.Module) -> bool:
    """Llama/Mistral-style backbone with LlamaModel.forward-compatible blocks."""
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        return False
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None else None
    # Gemma families use different masking / RoPE plumbing than this Llama-style loop.
    if isinstance(mt, str) and mt.startswith("gemma"):
        return False
    m = model.model
    return all(hasattr(m, name) for name in ("embed_tokens", "layers", "norm", "rotary_emb"))
