from __future__ import annotations

import torch
from llm_project.metrics.perplexity import perplexity_from_loss


class Evaluator:
    def __init__(
        self,
        model,
        sampler,
        batch_size: int,
        eval_steps: int,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.amp_dtype = amp_dtype
        self.use_amp = amp_dtype is not None

    @torch.no_grad()
    def run(self) -> dict[str, float]:
        self.model.eval()
        losses = []

        for _ in range(self.eval_steps):
            xb, yb = self.sampler.next_batch(self.batch_size)

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    _, loss = self.model(xb, yb)
            else:
                _, loss = self.model(xb, yb)

            losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        self.model.train()
        return {"loss": mean_loss, "perplexity": perplexity_from_loss(mean_loss)}