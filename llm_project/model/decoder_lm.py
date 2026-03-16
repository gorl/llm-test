from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLM(nn.Module):
    def __init__(self, input_encoder: nn.Module, blocks: list[nn.Module], final_norm: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.input_encoder = input_encoder
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = final_norm
        self.lm_head = lm_head

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.input_encoder(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(logits.view(batch_size * seq_len, vocab_size), targets.view(batch_size * seq_len))

        return logits, loss
