from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Architecture = Literal["traditional", "transformer_only"]
LossKind = Literal["ce", "margin"]


@dataclass(frozen=True)
class DFMConfig:
    """The deliberately small set of architecture choices supported here."""

    architecture: Architecture = "traditional"
    fusion_layers: tuple[int, ...] = (0, 2, 5, 8, 10, 11)
    fusion_timing: Literal["pre_attn", "post_attn"] | None = None
    memory_dim: int = 1024
    memory_slots: int = 32

    # Traditional DFM: shared projector, independent per-layer attention/gates.
    traditional_projector_hidden: int = 768
    memory_attention_heads: int = 12
    gate_type: Literal["none", "per_head", "token_wise_per_head", "token_wise_per_head_concat"] = "token_wise_per_head"
    gate_init: float = 0.0  # sigmoid(0) = 0.5
    memory_attention_dropout: float = 0.1

    # Transformer-only DFM: the strongest declared reader configuration.
    reader_dim: int = 256
    reader_layers: int = 2
    reader_heads: int = 8
    reader_ff_multiplier: int = 4
    reader_dropout: float = 0.0
    reader_topology: Literal["causal", "bidirectional"] = "causal"
    reader_write: Literal["residual", "replace"] = "residual"
    reader_sharing: Literal["independent", "shared"] = "independent"

    def __post_init__(self) -> None:
        if self.fusion_timing is None:
            timing = "pre_attn" if self.architecture == "traditional" else "post_attn"
            object.__setattr__(self, "fusion_timing", timing)

    def validate(self, gpt2_layers: int) -> None:
        if self.architecture not in {"traditional", "transformer_only"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if not self.fusion_layers or min(self.fusion_layers) < 0:
            raise ValueError("fusion_layers must contain non-negative indices")
        if max(self.fusion_layers) >= gpt2_layers:
            raise ValueError("fusion layer lies outside GPT-2")
        if self.fusion_timing not in {"pre_attn", "post_attn"}:
            raise ValueError("unsupported fusion_timing")
        if self.reader_dim % self.reader_heads:
            raise ValueError("reader_heads must divide reader_dim")
        if not 0 <= self.reader_dropout < 1:
            raise ValueError("reader_dropout must be in [0, 1)")
        if self.traditional_projector_hidden <= 0:
            raise ValueError("traditional_projector_hidden must be positive")
        if not 0 <= self.memory_attention_dropout < 1:
            raise ValueError("memory_attention_dropout must be in [0, 1)")
        if self.reader_topology not in {"causal", "bidirectional"}:
            raise ValueError("unsupported reader_topology")
        if self.reader_write not in {"residual", "replace"}:
            raise ValueError("unsupported reader_write")
        if self.reader_sharing not in {"independent", "shared"}:
            raise ValueError("unsupported reader_sharing")


@dataclass(frozen=True)
class LossConfig:
    """CE-only training or retrieved-vs-random hinge-margin training."""

    kind: LossKind = "ce"
    margin: float = 0.05
    margin_weight: float = 0.1

    def validate(self) -> None:
        if self.kind not in {"ce", "margin"}:
            raise ValueError(f"Unsupported loss: {self.kind}")
        if self.margin < 0 or self.margin_weight < 0:
            raise ValueError("margin and margin_weight must be non-negative")
