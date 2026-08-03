import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from dfm.config import DFMConfig
from dfm.model import DFMForCausalLM


def tiny_base() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(GPT2Config(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=4))


@pytest.mark.parametrize("architecture", ["traditional", "transformer_only"])
def test_invalid_memory_is_exact_noop(architecture: str) -> None:
    cfg = DFMConfig(
        architecture=architecture,
        fusion_layers=(0, 1),
        memory_dim=12,
        memory_slots=4,
        memory_attention_heads=4,
        reader_dim=16,
        reader_layers=1,
        reader_heads=4,
    )
    model = DFMForCausalLM(tiny_base(), cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    memory = torch.randn(1, 4, 4, 12)
    invalid = torch.zeros(1, 4, 4, dtype=torch.bool)
    with torch.no_grad():
        off = model(ids, mask)
        no_memory = model(ids, mask, memory, invalid)
    torch.testing.assert_close(no_memory, off, rtol=0, atol=0)


def test_only_memory_modules_are_trainable() -> None:
    model = DFMForCausalLM(
        tiny_base(),
        DFMConfig(
            fusion_layers=(0,),
            memory_dim=12,
            memory_slots=4,
            memory_attention_heads=4,
        ),
    )
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert sum(parameter.numel() for parameter in model.trainable_parameters()) > 0
