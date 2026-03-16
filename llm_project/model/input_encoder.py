from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    """
    GPT-2 style embeddings: token + learned positional embedding
    """

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


class TokenEmbedding(nn.Module):
    """
    Encoder for RoPE models.

    Only token embeddings.
    Positional information is injected later inside attention via RoPE.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_emb(token_ids)