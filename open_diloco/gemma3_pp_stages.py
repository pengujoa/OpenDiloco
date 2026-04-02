"""
Gemma3TextModel (gemma-3-*) pipeline stages for torch.distributed.pipelining (manual PipelineStage).

Stage interfaces (tuple form so non-first stages can run with positional args only):
- First stage:  (input_ids, attention_mask) -> (hidden_states, attention_mask)
- Middle stage: (hidden_states, attention_mask) -> (hidden_states, attention_mask)
- Last stage:   (hidden_states, attention_mask) -> logits
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def _gemma3_build_masks(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
):
    # Gemma3 expects a dict of masks keyed by attention type.
    mask_kwargs = {
        "config": config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    sliding_mask_kwargs = dict(mask_kwargs)

    if getattr(config, "use_bidirectional_attention", False):
        # Bidirectional overlay: simplest behavior (allow all attention).
        mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
        sliding_mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)

    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
    }


def _gemma3_prep(
    config,
    hidden_or_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
):
    inputs_embeds = hidden_or_embeds
    cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    masks = _gemma3_build_masks(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        position_ids=position_ids,
    )
    return inputs_embeds, cache_position, position_ids, masks


class Gemma3PPFirst(nn.Module):
    def __init__(self, causal_lm: nn.Module, activation_dtype: torch.dtype):
        super().__init__()
        self.config = causal_lm.config
        self.backbone = causal_lm.model  # Gemma3TextModel
        self.embed_tokens = self.backbone.embed_tokens
        self.rotary_emb = self.backbone.rotary_emb
        self.rotary_emb_local = self.backbone.rotary_emb_local
        self.layers = self.backbone.layers
        self.activation_dtype = activation_dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # IMPORTANT: pipeline stage recv buffers are marked requires_grad_(True).
        # Non-floating tensors (e.g. int attention_mask) cannot require grads.
        # We therefore pass attention_mask between stages as float and cast back.
        attention_mask_f = attention_mask.to(dtype=torch.float32)
        inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = None
        use_cache = False

        hidden_states, cache_position, position_ids, masks = _gemma3_prep(
            self.config, inputs_embeds, attention_mask, past_key_values
        )
        pos_global = self.rotary_emb(hidden_states, position_ids)
        pos_local = self.rotary_emb_local(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings_global=pos_global,
                position_embeddings_local=pos_local,
                attention_mask=masks[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
            )[0]

        return hidden_states.to(dtype=self.activation_dtype), attention_mask_f


class Gemma3PPMiddle(nn.Module):
    def __init__(self, causal_lm: nn.Module, activation_dtype: torch.dtype):
        super().__init__()
        self.config = causal_lm.config
        self.backbone = causal_lm.model
        self.rotary_emb = self.backbone.rotary_emb
        self.rotary_emb_local = self.backbone.rotary_emb_local
        self.layers = self.backbone.layers
        self.activation_dtype = activation_dtype

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask_l = attention_mask.to(dtype=torch.long)
        # Ensure attention_mask participates in autograd so the pipeline runtime
        # can send a (zero) gradient back to the previous stage.
        hidden_states = hidden_states + attention_mask.to(hidden_states.dtype).unsqueeze(-1) * 0.0
        past_key_values = None
        use_cache = False

        hidden_states, cache_position, position_ids, masks = _gemma3_prep(
            self.config, hidden_states, attention_mask_l, past_key_values
        )
        pos_global = self.rotary_emb(hidden_states, position_ids)
        pos_local = self.rotary_emb_local(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings_global=pos_global,
                position_embeddings_local=pos_local,
                attention_mask=masks[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
            )[0]

        return hidden_states.to(dtype=self.activation_dtype), attention_mask


class Gemma3PPLast(nn.Module):
    def __init__(self, causal_lm: nn.Module, activation_dtype: torch.dtype):
        super().__init__()
        self.config = causal_lm.config
        self.backbone = causal_lm.model
        self.rotary_emb = self.backbone.rotary_emb
        self.rotary_emb_local = self.backbone.rotary_emb_local
        self.layers = self.backbone.layers
        self.norm = self.backbone.norm
        self.lm_head = causal_lm.lm_head
        self.activation_dtype = activation_dtype

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask_l = attention_mask.to(dtype=torch.long)
        hidden_states = hidden_states + attention_mask.to(hidden_states.dtype).unsqueeze(-1) * 0.0
        past_key_values = None
        use_cache = False

        hidden_states, cache_position, position_ids, masks = _gemma3_prep(
            self.config, hidden_states, attention_mask_l, past_key_values
        )
        pos_global = self.rotary_emb(hidden_states, position_ids)
        pos_local = self.rotary_emb_local(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings_global=pos_global,
                position_embeddings_local=pos_local,
                attention_mask=masks[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
            )[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).to(dtype=self.activation_dtype)
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        return logits


def is_gemma3_pp_supported(model: nn.Module) -> bool:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None else None
    return isinstance(mt, str) and mt in ("gemma3", "gemma3_text")


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


def slice_gemma3_for_pp(
    causal_lm: nn.Module,
    num_stages: int,
    stage_index: int,
    activation_dtype: torch.dtype,
) -> Tuple[nn.Module, Tuple[int, int]]:
    """
    In-place: keep only layers for this stage. Returns (stage_module, (layer_start, layer_end)).
    """
    num_layers = len(causal_lm.model.layers)
    boundaries = _balanced_pp_boundaries(num_layers, num_stages)

    layer_start, layer_end = boundaries[stage_index]
    keep = set(range(layer_start, layer_end))
    drop = sorted(set(range(num_layers)) - keep, reverse=True)
    for idx in drop:
        del causal_lm.model.layers[idx]

    if stage_index != 0:
        causal_lm.model.embed_tokens = None
    if stage_index != num_stages - 1:
        causal_lm.lm_head = None
        causal_lm.model.norm = None

    if stage_index == 0:
        submodule = Gemma3PPFirst(causal_lm, activation_dtype=activation_dtype)
    elif stage_index == num_stages - 1:
        submodule = Gemma3PPLast(causal_lm, activation_dtype=activation_dtype)
    else:
        submodule = Gemma3PPMiddle(causal_lm, activation_dtype=activation_dtype)

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

    cfg = stage_module.config
    hidden = torch.zeros(
        microbatch_size,
        seq_len,
        cfg.hidden_size,
        dtype=activation_dtype,
        device=device,
    )
    attn = torch.zeros(microbatch_size, seq_len, dtype=torch.float32, device=device)

    if stage_index == 0:
        input_args = (
            torch.zeros(microbatch_size, seq_len, dtype=torch.long, device=device),
            attn,
        )
        output_args = (hidden, attn)
    elif stage_index == num_stages - 1:
        input_args = (hidden, attn)
        output_args = torch.zeros(
            microbatch_size, seq_len, cfg.vocab_size, dtype=activation_dtype, device=device
        )
    else:
        input_args = (hidden, attn)
        output_args = (hidden, attn)

    return PipelineStage(
        stage_module,
        stage_index,
        num_stages,
        device,
        input_args,
        output_args,
    )

