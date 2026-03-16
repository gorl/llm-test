from __future__ import annotations

import torch


from llm_project.configs.base import ModelConfig
from llm_project.model.multi_head_attention import MultiHeadAttention
from llm_project.model.blocks import TransformerBlock
from llm_project.model.decoder_lm import DecoderLM
from llm_project.model.ffn import GELUFFN
from llm_project.model.heads import build_lm_head
from llm_project.model.input_encoder import TokenAndPositionEmbedding


def build_multi_head_model(cfg: ModelConfig) -> DecoderLM:
    input_encoder = TokenAndPositionEmbedding(cfg.vocab_size, cfg.d_model, cfg.block_size)
    
    blocks = []
    for _ in range(cfg.n_layers):
        attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        ffn = GELUFFN(cfg.d_model, cfg.hidden_dim, cfg.dropout)
        block = TransformerBlock(attention, ffn, cfg.d_model)
        blocks.append(block)

    final_norm = torch.nn.LayerNorm(cfg.d_model)
    # lm_head = build_lm_head(cfg.d_model, cfg.vocab_size, tok_w=input_encoder.token_emb.weight, bias=False)
    lm_head = build_lm_head(cfg.d_model, cfg.vocab_size)
    return DecoderLM(input_encoder, blocks, final_norm, lm_head)