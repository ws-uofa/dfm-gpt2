"""Small, explicit implementation of Deep Fusion Memory for GPT-2."""

from .config import DFMConfig, LossConfig
from .model import DFMForCausalLM

__all__ = ["DFMConfig", "DFMForCausalLM", "LossConfig"]
