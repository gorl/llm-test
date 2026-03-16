from __future__ import annotations

import torch
import torch.nn.functional as F


class LanguageModelingCrossEntropy:
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        return F.cross_entropy(logits.view(batch_size * seq_len, vocab_size), targets.view(batch_size * seq_len))
