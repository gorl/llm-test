from __future__ import annotations

import torch


class RandomBatchSampler:
    def __init__(self, dataset, device: str) -> None:
        self.dataset = dataset
        self.device = device

    def next_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self.dataset), (batch_size,))
        xs = []
        ys = []
        for idx in indices.tolist():
            x, y = self.dataset.get_item(idx)
            xs.append(x)
            ys.append(y)
        xb = torch.stack(xs).to(self.device)
        yb = torch.stack(ys).to(self.device)
        return xb, yb
