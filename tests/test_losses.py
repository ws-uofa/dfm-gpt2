import torch

from dfm.losses import margin_loss, token_nll
from dfm.train import negative_forward_rng_context


def test_margin_formula() -> None:
    total, penalty = margin_loss(torch.tensor(2.0), torch.tensor(2.02), margin=0.05, weight=0.1)
    torch.testing.assert_close(penalty, torch.tensor(0.03))
    torch.testing.assert_close(total, torch.tensor(2.003))


def test_token_nll_ignores_first_and_masked_targets() -> None:
    logits = torch.zeros(1, 3, 2)
    labels = torch.tensor([[1, 0, -100]])
    torch.testing.assert_close(token_nll(logits, labels), torch.tensor(0.6931472))


def test_negative_forward_rng_isolation() -> None:
    reference = torch.empty(1)
    torch.manual_seed(42)
    expected = torch.rand(3)
    torch.manual_seed(42)
    with negative_forward_rng_context(reference, preserve=True):
        torch.rand(10)
    torch.testing.assert_close(torch.rand(3), expected)
