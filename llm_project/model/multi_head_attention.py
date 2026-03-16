import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # self.q_proj = nn.Linear(d_model, d_model)
        # self.k_proj = nn.Linear(d_model, d_model)
        # self.v_proj = nn.Linear(d_model, d_model)
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1024, 1024, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, C = x.shape

        # 1. Получаем Q, K, V
        # q = self.q_proj(x)  # (B, T, d_model)
        # k = self.k_proj(x)  # (B, T, d_model)
        # v = self.v_proj(x)  # (B, T, d_model)

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # 2. Раскладываем по головам
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # 3. Считаем attention scores
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        # 4. Causal mask: запрещаем смотреть в будущее
        # mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))  # (T, T)
        mask = self.mask[:T, :T]
        
        att = att.masked_fill(~mask, float("-inf"))

        # 5. Превращаем scores в вероятности
        att = F.softmax(att, dim=-1)  # (B, H, T, T)
        att = self.attn_dropout(att)

        # 6. Взвешенно агрегируем V
        out = att @ v  # (B, H, T, D)

        # 7. Склеиваем головы обратно
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)

        # 8. Финальная проекция
        out = self.out_proj(out)  # (B, T, d_model)
        out = self.resid_dropout(out)

        return out