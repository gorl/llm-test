from __future__ import annotations

import numpy as np
import torch


class PackedBatchSampler:
    """
    Random batch sampler for PackedTokenDataset.

    Behavior:
    - choose one shard
    - sample batches from this shard for N consecutive batches
    - then switch to another shard, chosen proportionally to valid window count

    This improves memmap locality and reduces random jumping across files.

    Interface is compatible with Trainer via next_batch(batch_size).
    """

    def __init__(
        self,
        dataset,
        device: str,
        batches_per_shard: int = 64,
    ) -> None:
        self.dataset = dataset
        self.device = device
        self.rng = np.random.default_rng()

        if batches_per_shard <= 0:
            raise ValueError(
                f"batches_per_shard must be positive, got {batches_per_shard}"
            )
        self.batches_per_shard = batches_per_shard

        self._current_shard_idx: int | None = None
        self._batches_left_on_current_shard = 0

    def _pick_new_shard(self) -> int:
        shard_idx = int(
            self.rng.choice(
                self.dataset.valid_shard_indices,
                p=self.dataset.valid_shard_probs,
            )
        )
        return shard_idx

    def _get_active_shard_idx(self) -> int:
        if (
            self._current_shard_idx is None
            or self._batches_left_on_current_shard <= 0
        ):
            self._current_shard_idx = self._pick_new_shard()
            self._batches_left_on_current_shard = self.batches_per_shard

        self._batches_left_on_current_shard -= 1
        return self._current_shard_idx

    def next_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        block_size = self.dataset.block_size
        shard_idx = self._get_active_shard_idx()

        arr = self.dataset.get_array(shard_idx)
        valid_starts = self.dataset.get_valid_start_count(shard_idx)
        if valid_starts <= 0:
            raise RuntimeError(f"Chosen shard {shard_idx} has no valid windows")

        replace = valid_starts < batch_size
        starts = self.rng.choice(valid_starts, size=batch_size, replace=replace)

        xb = np.empty((batch_size, block_size), dtype=np.int64)
        yb = np.empty((batch_size, block_size), dtype=np.int64)

        for i, start in enumerate(starts):
            start = int(start)
            xb[i] = arr[start:start + block_size]
            yb[i] = arr[start + 1:start + 1 + block_size]

        x = torch.from_numpy(xb)
        y = torch.from_numpy(yb)

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)