from __future__ import annotations

import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[:, [-1]]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
