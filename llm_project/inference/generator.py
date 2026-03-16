from __future__ import annotations

import torch
from llm_project.inference.decoding import sample_next_token


class Generator:
    def __init__(self, model, block_size: int) -> None:
        self.model = model
        self.block_size = block_size

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 20) -> torch.Tensor:
        self.model.eval()
        idx = prompt_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.model(idx_cond)
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
