import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        d_model: int,
        use_rmsnorm: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        norm_cls = nn.RMSNorm if use_rmsnorm and hasattr(nn, "RMSNorm") else nn.LayerNorm

        self.norm1 = norm_cls(d_model)
        self.attention = attention
        self.norm2 = norm_cls(d_model)
        self.ffn = ffn
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.attention(self.norm1(x))
        x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x