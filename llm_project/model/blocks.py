from __future__ import annotations

import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, attention: nn.Module, ffn: nn.Module, d_model: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = attention
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
