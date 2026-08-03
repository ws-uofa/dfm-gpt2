from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def token_nll(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean next-token NLL, honoring the usual -100 ignored labels."""

    return F.cross_entropy(
        logits[:, :-1].contiguous().float().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def margin_loss(
    retrieved_nll: Tensor,
    random_nll: Tensor,
    *,
    margin: float,
    weight: float,
) -> tuple[Tensor, Tensor]:
    """Prefer retrieved memory over an aligned real-memory random control."""

    penalty = F.relu(float(margin) + retrieved_nll - random_nll)
    return retrieved_nll + float(weight) * penalty, penalty
