from __future__ import annotations

import torch
import torch.nn as nn

from llm_project.configs.base import ModelConfig
from llm_project.model.multi_head_attention import MultiHeadAttention
from llm_project.model.blocks import TransformerBlock
from llm_project.model.decoder_lm import DecoderLM
from llm_project.model.ffn import SwiGLUFFN
from llm_project.model.heads import build_lm_head
from llm_project.model.input_encoder import TokenEmbedding


def build_multi_head_model(cfg: ModelConfig) -> DecoderLM:
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")

    if cfg.hidden_dim < cfg.d_model:
        raise ValueError(
            f"hidden_dim ({cfg.hidden_dim}) should usually be >= d_model ({cfg.d_model})"
        )

    if not (0.0 <= cfg.dropout <= 0.3):
        raise ValueError(f"dropout must be in [0.0, 0.3], got {cfg.dropout}")

    input_encoder = TokenEmbedding(
        cfg.vocab_size,
        cfg.d_model,
    )

    blocks = []
    for _ in range(cfg.n_layers):
        attention = MultiHeadAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        ffn = SwiGLUFFN(
            d_model=cfg.d_model,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
        block = TransformerBlock(
            attention=attention,
            ffn=ffn,
            d_model=cfg.d_model,
            use_rmsnorm=True,
            residual_scale=1.0,
        )
        blocks.append(block)

    norm_cls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
    final_norm = norm_cls(cfg.d_model)

    lm_head = build_lm_head(
        cfg.d_model,
        cfg.vocab_size,
    )

    return DecoderLM(
        input_encoder=input_encoder,
        blocks=blocks,
        final_norm=final_norm,
        lm_head=lm_head,
    )