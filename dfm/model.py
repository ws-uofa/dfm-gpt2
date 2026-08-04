from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn
from transformers import GPT2LMHeadModel

from .config import DFMConfig


class MemoryProjector(nn.Module):
    """Shared Qwen-embedding to GPT-2-hidden projection used by traditional DFM."""

    def __init__(self, memory_dim: int, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, memory: Tensor) -> Tensor:
        return self.net(memory)


class GatedMemoryAttention(nn.Module):
    """Token-wise, per-head cross-attention followed by a sigmoid residual gate."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        gate_type: str,
        gate_init: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("GPT-2 hidden size must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_type = gate_type
        self.gate = None
        self.gate_proj = None
        if gate_type == "per_head":
            self.gate = nn.Parameter(torch.full((heads,), float(gate_init)))
        elif gate_type == "token_wise_per_head":
            self.gate_proj = nn.Linear(hidden_dim, heads)
        elif gate_type == "token_wise_per_head_concat":
            self.gate_proj = nn.Linear(hidden_dim * 2, heads)
        elif gate_type != "none":
            raise ValueError(f"unsupported gate_type: {gate_type}")
        if self.gate_proj is not None:
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, gate_init)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        valid: Tensor,
        *,
        return_update: bool = False,
    ) -> Tensor:
        batch, tokens, slots, width = memory.shape
        q = self.q(hidden).view(batch, tokens, self.heads, self.head_dim)
        k = self.k(memory).view(batch, tokens, slots, self.heads, self.head_dim)
        v = self.v(memory).view(batch, tokens, slots, self.heads, self.head_dim)
        scores = torch.einsum("bthd,btshd->bths", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, :, None, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(dim=-1))
        # Avoid NaNs and writes for chunks with no valid memory (the first chunk).
        has_memory = valid.any(dim=-1)
        weights = torch.where(has_memory[:, :, None, None], weights, torch.zeros_like(weights))
        update = torch.einsum("bths,btshd->bthd", weights, v)
        flat_update = update.reshape(batch, tokens, width)
        if self.gate_type == "none":
            gate = 1.0
        elif self.gate_type == "per_head":
            gate = self.gate.sigmoid()[None, None, :, None]
        else:
            gate_input = hidden if self.gate_type == "token_wise_per_head" else torch.cat((hidden, flat_update), dim=-1)
            gate = self.gate_proj(gate_input).sigmoid().unsqueeze(-1)
        update = self.out((update * gate).reshape(batch, tokens, width))
        update = update * has_memory[:, :, None]
        return update if return_update else hidden + update


class TransformerReaderBlock(nn.Module):
    """Pre-norm block matching the reader used by the WikiText experiments."""

    def __init__(self, width: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * ff_multiplier, width),
        )
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor, valid: Tensor, causal: bool) -> Tensor:
        size = states.size(1)
        causal_mask = None
        if causal:
            causal_mask = torch.ones(size, size, dtype=torch.bool, device=states.device).triu(1)
        normalized = self.attn_norm(states)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=~valid,
            need_weights=False,
        )
        states = states + self.attn_dropout(attended)
        return states + self.ff_dropout(self.ff(self.ff_norm(states)))


class TransformerReader(nn.Module):
    """Continuous-prefix reader with the exact recent-experiment parameter layout."""

    def __init__(self, cfg: DFMConfig, hidden_dim: int) -> None:
        super().__init__()
        width = cfg.reader_dim
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.memory_in = nn.Linear(cfg.memory_dim, width)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_in = nn.Linear(hidden_dim, width)
        self.position = nn.Embedding(cfg.memory_slots + 1, width)
        self.blocks = nn.ModuleList(
            TransformerReaderBlock(
                width,
                cfg.reader_heads,
                cfg.reader_ff_multiplier,
                cfg.reader_dropout,
            )
            for _ in range(cfg.reader_layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, hidden_dim)
        self.topology = cfg.reader_topology
        self.write = cfg.reader_write

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        valid: Tensor,
        *,
        return_update: bool = False,
    ) -> Tensor:
        batch, tokens, slots, _ = memory.shape
        safe_memory = torch.where(valid.unsqueeze(-1), memory, torch.zeros_like(memory))
        mem = self.memory_in(self.memory_norm(safe_memory))
        query = self.query_in(self.query_norm(hidden)).unsqueeze(2)
        sequence = torch.cat((mem, query), dim=2)
        positions = torch.arange(slots + 1, device=hidden.device)
        sequence = sequence + self.position(positions)
        flat = sequence.view(batch * tokens, slots + 1, -1)
        reader_valid = torch.cat((valid, torch.ones_like(valid[:, :, :1])), dim=2)
        flat_valid = reader_valid.view(batch * tokens, slots + 1)
        for block in self.blocks:
            flat = block(flat, flat_valid, causal=self.topology == "causal")
        candidate = self.out(self.final_norm(flat[:, -1])).view(batch, tokens, -1)
        has_memory = valid.any(dim=-1, keepdim=True)
        update = candidate if self.write == "residual" else candidate - hidden
        update = update * has_memory
        if return_update:
            return update
        return hidden + update


class DFMForCausalLM(nn.Module):
    """Frozen GPT-2 with memory writes at an exact pre/post-attention boundary."""

    def __init__(self, base: GPT2LMHeadModel, cfg: DFMConfig) -> None:
        super().__init__()
        cfg.validate(base.config.n_layer)
        self.base = base
        self.dfm_config = cfg
        hidden = base.config.n_embd
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        if cfg.architecture == "traditional":
            self.projector = MemoryProjector(
                cfg.memory_dim, hidden, cfg.traditional_projector_hidden
            )
            self.fusion = nn.ModuleDict(
                {
                    str(i): GatedMemoryAttention(
                        hidden,
                        cfg.memory_attention_heads,
                        cfg.gate_type,
                        cfg.gate_init,
                        cfg.memory_attention_dropout,
                    )
                    for i in cfg.fusion_layers
                }
            )
            self.shared_reader = None
            self.post_attention_norms = nn.ModuleDict(
                {
                    str(i): nn.LayerNorm(hidden, eps=base.config.layer_norm_epsilon)
                    for i in cfg.fusion_layers
                }
                if cfg.fusion_timing == "post_attn"
                else {}
            )
        else:
            self.projector = None
            self.post_attention_norms = nn.ModuleDict()
            if cfg.reader_sharing == "shared":
                self.shared_reader = TransformerReader(cfg, hidden)
                self.fusion = nn.ModuleDict()
            else:
                self.shared_reader = None
                self.fusion = nn.ModuleDict(
                    {str(i): TransformerReader(cfg, hidden) for i in cfg.fusion_layers}
                )

        self._memory: tuple[Tensor, Tensor] | None = None
        self._layer_inputs: dict[str, Tensor] = {}
        self._hooks = []
        for layer_idx in cfg.fusion_layers:
            layer = str(layer_idx)
            block = base.transformer.h[layer_idx]
            if cfg.fusion_timing == "pre_attn":
                if cfg.architecture == "transformer_only":
                    self._hooks.append(
                        block.register_forward_pre_hook(self._capture_layer_input(layer))
                    )
                self._hooks.append(block.attn.register_forward_hook(self._pre_attn_hook(layer)))
            else:
                self._hooks.append(
                    block.ln_2.register_forward_pre_hook(self._post_attn_hook(layer))
                )

    @classmethod
    def from_pretrained(cls, path: str, cfg: DFMConfig) -> "DFMForCausalLM":
        return cls(GPT2LMHeadModel.from_pretrained(path), cfg)

    def _reader(self, layer: str) -> nn.Module:
        return self.shared_reader if self.shared_reader is not None else self.fusion[layer]

    def _capture_layer_input(self, layer: str):
        def capture(_module: nn.Module, inputs: tuple[object, ...]) -> None:
            if self._memory is not None:
                self._layer_inputs[layer] = inputs[0]

        return capture

    def _pre_attn_hook(self, layer: str):
        def apply_memory(_module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            if self._memory is None:
                return output
            memory, valid = self._memory
            if self.dfm_config.architecture == "traditional":
                query = inputs[0]  # the same LN(h_l) consumed by base self-attention
                update = self.fusion[layer](
                    query, self.projector(memory), valid, return_update=True
                )
            else:
                query = self._layer_inputs.pop(layer)
                update = self._reader(layer)(query, memory, valid, return_update=True)
            attended = output[0] + update
            return (attended, *output[1:])

        return apply_memory

    def _post_attn_hook(self, layer: str):
        def apply_memory(_module: nn.Module, inputs: tuple[object, ...]) -> tuple[object, ...] | None:
            if self._memory is None:
                return None
            hidden = inputs[0]  # h_l + SA(LN(h_l)), immediately before the MLP
            memory, valid = self._memory
            if self.dfm_config.architecture == "traditional":
                query = self.post_attention_norms[layer](hidden)
                update = self.fusion[layer](
                    query, self.projector(memory), valid, return_update=True
                )
                fused = hidden + update
            else:
                fused = self._reader(layer)(hidden, memory, valid)
            return (fused, *inputs[1:])

        return apply_memory

    @contextmanager
    def _use_memory(self, memory: Tensor, valid: Tensor) -> Iterator[None]:
        if self._memory is not None:
            raise RuntimeError("Nested DFM forward calls are not supported")
        self._memory = (memory, valid)
        try:
            yield
        finally:
            self._memory = None
            self._layer_inputs.clear()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        if memory is None:
            return self.base(input_ids=input_ids, attention_mask=attention_mask).logits
        if memory_mask is None:
            raise ValueError("memory_mask is required when memory is supplied")
        if memory.shape[:3] != memory_mask.shape:
            raise ValueError("memory and memory_mask dimensions do not agree")
        with self._use_memory(memory, memory_mask.bool()):
            return self.base(input_ids=input_ids, attention_mask=attention_mask).logits

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)
