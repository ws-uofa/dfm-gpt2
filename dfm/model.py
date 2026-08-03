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

    def forward(self, hidden: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
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
        return hidden + update * has_memory[:, :, None]


class TransformerReader(nn.Module):
    """Continuous-prefix Transformer; memory tokens precede one GPT-2 query token."""

    def __init__(self, cfg: DFMConfig, hidden_dim: int) -> None:
        super().__init__()
        width = cfg.reader_dim
        self.memory_in = nn.Linear(cfg.memory_dim, width, bias=False)
        self.query_in = nn.Linear(hidden_dim, width, bias=False)
        self.position = nn.Parameter(torch.zeros(cfg.memory_slots + 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=cfg.reader_heads,
            dim_feedforward=width * cfg.reader_ff_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.reader = nn.TransformerEncoder(layer, cfg.reader_layers, norm=nn.LayerNorm(width))
        self.out = nn.Linear(width, hidden_dim, bias=False)
        self.topology = cfg.reader_topology
        self.write = cfg.reader_write

    def forward(self, hidden: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        batch, tokens, slots, _ = memory.shape
        mem = self.memory_in(memory)
        query = self.query_in(hidden).unsqueeze(2)
        sequence = torch.cat((mem, query), dim=2)
        sequence = sequence + self.position[: slots + 1]
        flat = sequence.view(batch * tokens, slots + 1, -1)
        padding = torch.cat((~valid, torch.zeros_like(valid[:, :, :1])), dim=2)
        causal_mask = None
        if self.topology == "causal":
            causal_mask = torch.triu(
                torch.ones(slots + 1, slots + 1, dtype=torch.bool, device=flat.device),
                diagonal=1,
            )
        encoded = self.reader(
            flat,
            mask=causal_mask,
            src_key_padding_mask=padding.view(batch * tokens, slots + 1),
        )
        update = self.out(encoded[:, -1]).view(batch, tokens, -1)
        has_memory = valid.any(dim=-1, keepdim=True)
        fused = hidden + update if self.write == "residual" else update
        return torch.where(has_memory, fused, hidden)


class DFMForCausalLM(nn.Module):
    """Frozen GPT-2 with readable memory modules inserted through block hooks.

    Hooks keep Hugging Face's tested GPT-2 forward implementation intact. The
    context is installed only for the duration of one forward pass.
    """

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
        else:
            self.projector = None
            self.fusion = nn.ModuleDict({str(i): TransformerReader(cfg, hidden) for i in cfg.fusion_layers})

        self._memory: tuple[Tensor, Tensor] | None = None
        self._hooks = [base.transformer.h[i].register_forward_hook(self._hook(str(i))) for i in cfg.fusion_layers]

    @classmethod
    def from_pretrained(cls, path: str, cfg: DFMConfig) -> "DFMForCausalLM":
        return cls(GPT2LMHeadModel.from_pretrained(path), cfg)

    def _hook(self, layer: str):
        def apply_memory(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            if self._memory is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            memory, valid = self._memory
            projected = self.projector(memory) if self.projector is not None else memory
            fused = self.fusion[layer](hidden, projected, valid)
            return (fused, *output[1:]) if isinstance(output, tuple) else fused

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
