import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE.

    Features:
    - fused QKV projection
    - PyTorch scaled_dot_product_attention
    - rotary positional embeddings applied to q/k
    - no hardcoded causal mask
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout_p = float(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D)
        Splits last dim into pairs and rotates:
        [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(start_dim=-2)

    def _build_rope_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos, sin of shape (1, 1, T, D)
        """
        half_dim = self.head_dim // 2

        # Frequencies for pairs of dimensions
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
        )  # (D/2,)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
        freqs = torch.outer(positions, inv_freq)  # (T, D/2)

        # Duplicate each frequency for even/odd dims: (T, D)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        return cos, sin

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, H, T, D)
        """
        _, _, seq_len, _ = q.shape
        cos, sin = self._build_rope_cache(seq_len, q.device, q.dtype)

        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, T, C)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, T, C), got {tuple(x.shape)}")

        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {C}")

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        q, k = self._apply_rope(q, k)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out