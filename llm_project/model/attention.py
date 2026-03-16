from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, d_model = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_model)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        y = weights @ v
        return self.out_proj(y)
