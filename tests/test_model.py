import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from dfm.config import DFMConfig
from dfm.model import DFMForCausalLM, GatedMemoryAttention, MemoryProjector, TransformerReader


def tiny_base() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(GPT2Config(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=4))


@pytest.mark.parametrize("architecture", ["traditional", "transformer_only"])
@pytest.mark.parametrize("fusion_timing", ["pre_attn", "post_attn"])
def test_invalid_memory_is_exact_noop(architecture: str, fusion_timing: str) -> None:
    cfg = DFMConfig(
        architecture=architecture,
        fusion_timing=fusion_timing,
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


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_recent_architecture_parameter_counts() -> None:
    traditional = DFMConfig(architecture="traditional")
    projector = MemoryProjector(1024, 768, 768)
    attention = GatedMemoryAttention(768, 12, "token_wise_per_head", 0.0, 0.1)
    traditional_pre = count_parameters(projector) + 6 * count_parameters(attention)
    assert traditional_pre == 15_609_416
    assert traditional_pre + 6 * count_parameters(nn.LayerNorm(768)) == 15_618_632

    transformer = DFMConfig(architecture="transformer_only")
    one_reader = count_parameters(TransformerReader(transformer, hidden_dim=768))
    assert one_reader == 2_248_704
    assert 6 * one_reader == 13_492_224
    assert traditional.fusion_timing == "pre_attn"
    assert transformer.fusion_timing == "post_attn"
    assert (transformer.reader_dim, transformer.reader_layers, transformer.reader_heads) == (256, 2, 8)


class ProbeReader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        _memory: torch.Tensor,
        _valid: torch.Tensor,
        *,
        return_update: bool = False,
    ) -> torch.Tensor:
        self.query = hidden.detach().clone()
        return torch.zeros_like(hidden) if return_update else hidden


@pytest.mark.parametrize("fusion_timing", ["pre_attn", "post_attn"])
def test_transformer_reader_uses_exact_attention_boundary(fusion_timing: str) -> None:
    model = DFMForCausalLM(
        tiny_base(),
        DFMConfig(
            architecture="transformer_only",
            fusion_layers=(0,),
            fusion_timing=fusion_timing,
            memory_dim=12,
            memory_slots=4,
            reader_dim=16,
            reader_layers=1,
            reader_heads=4,
        ),
    ).eval()
    probe = ProbeReader()
    model.fusion["0"] = probe
    captured: dict[str, torch.Tensor] = {}
    block = model.base.transformer.h[0]

    def capture_input(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        captured["input"] = inputs[0].detach().clone()

    def capture_attn(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        captured["attn"] = output[0].detach().clone()

    handles = [block.register_forward_pre_hook(capture_input)]
    if fusion_timing == "post_attn":
        handles.append(block.attn.register_forward_hook(capture_attn))
    ids = torch.tensor([[1, 2, 3, 4]])
    memory = torch.randn(1, 4, 4, 12)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    try:
        with torch.no_grad():
            model(ids, torch.ones_like(ids), memory, valid)
    finally:
        for handle in handles:
            handle.remove()

    assert probe.query is not None
    expected = captured["input"]
    if fusion_timing == "post_attn":
        expected = expected + captured["attn"]
    torch.testing.assert_close(probe.query, expected, rtol=0, atol=0)
