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
    memory_dim: int = 1024
    memory_slots: int = 32

    # Traditional DFM: shared projector, independent per-layer attention/gates.
    traditional_projector_hidden: int = 768
    gate_init: float = 0.0  # sigmoid(0) = 0.5

    # Transformer-only DFM: the strongest declared reader configuration.
    reader_dim: int = 256
    reader_layers: int = 4
    reader_heads: int = 8
    reader_ff_multiplier: int = 4

    def validate(self, gpt2_layers: int) -> None:
        if self.architecture not in {"traditional", "transformer_only"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if not self.fusion_layers or min(self.fusion_layers) < 0:
            raise ValueError("fusion_layers must contain non-negative indices")
        if max(self.fusion_layers) >= gpt2_layers:
            raise ValueError("fusion layer lies outside GPT-2")
        if self.reader_dim % self.reader_heads:
            raise ValueError("reader_heads must divide reader_dim")


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
