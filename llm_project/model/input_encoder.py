from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, block_size: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        tok = self.token_emb(token_ids)
        pos = self.pos_emb(positions)
        return tok + pos
