from __future__ import annotations

import torch


# class LanguageModelingDataset:
#     def __init__(self, tokens: list[int], block_size: int) -> None:
#         self.data = torch.tensor(tokens, dtype=torch.long)
#         self.block_size = block_size
#         if len(self.data) < self.block_size + 2:
#             raise ValueError("Not enough tokens for the chosen block size")

#     def __len__(self) -> int:
#         return len(self.data) - self.block_size - 1

#     def get_item(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         x = self.data[index:index + self.block_size]
#         y = self.data[index + 1:index + self.block_size + 1]
#         return x, y

# from __future__ import annotations
# import torch

import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset

def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "uint16":
        return np.uint16
    if dtype_name == "int32":
        return np.int32
    if dtype_name == "uint32":
        return np.uint32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class PackedTokenDataset:
    """
    Packed token dataset over multiple .bin shards.

    Each shard is a flat array of token ids. Documents are concatenated with EOS.
    The dataset exposes shard metadata and valid random window ranges, while the
    sampler performs the actual random batching.
    """

    def __init__(
        self,
        paths: list[str] | list[Path],
        block_size: int,
        dtype: np.dtype,
    ) -> None:
        self.paths = [str(p) for p in paths]
        self.block_size = int(block_size)
        self.dtype = dtype

        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not self.paths:
            raise ValueError("PackedTokenDataset requires at least one shard path")

        self.arrays = [np.memmap(path, dtype=self.dtype, mode="r") for path in self.paths]
        self.shard_lengths = np.array([len(arr) for arr in self.arrays], dtype=np.int64)

        # Valid start count per shard for windows:
        # x = arr[i:i+block]
        # y = arr[i+1:i+block+1]
        # so need i + block < len(arr)
        self.valid_start_counts = np.maximum(self.shard_lengths - self.block_size, 0)

        self.total_tokens = int(self.shard_lengths.sum())
        self.total_windows = int(self.valid_start_counts.sum())

        if self.total_tokens <= 0:
            raise ValueError("PackedTokenDataset has no tokens")
        if self.total_windows <= 0:
            raise ValueError(
                f"No valid windows for block_size={self.block_size}. "
                f"Shard lengths: {self.shard_lengths.tolist()}"
            )

        valid_mask = self.valid_start_counts > 0
        self.valid_shard_indices = np.nonzero(valid_mask)[0]
        if len(self.valid_shard_indices) == 0:
            raise ValueError("No shards contain enough tokens for one training window")

        valid_counts = self.valid_start_counts[self.valid_shard_indices].astype(np.float64)
        self.valid_shard_probs = valid_counts / valid_counts.sum()

    def __len__(self) -> int:
        # Kept mainly for stats/debugging. Not used for random batching directly.
        return self.total_windows

    @property
    def num_shards(self) -> int:
        return len(self.arrays)

    def get_array(self, shard_idx: int) -> np.memmap:
        return self.arrays[shard_idx]

    def get_valid_start_count(self, shard_idx: int) -> int:
        return int(self.valid_start_counts[shard_idx])


class LanguageModelingDatasetPlain:
    def __init__(self, tokens, block_size: int) -> None:
        self.tokens = tokens
        self.block_size = block_size

        if len(self.tokens) < block_size + 1:
            raise ValueError(
                f"Not enough tokens for block_size={block_size}: got {len(self.tokens)} tokens"
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def get_item(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.tokens[index:index + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[index + 1:index + self.block_size + 1], dtype=torch.long)
        return x, y


class LanguageModelingDataset:
    def __init__(self, docs: list[list[int]], block_size: int) -> None:
        self.docs = [
            torch.tensor(doc, dtype=torch.long)
            for doc in docs
            if len(doc) >= block_size + 1
        ]
        self.block_size = block_size

        if not self.docs:
            raise ValueError("No documents long enough for block_size")

    def __len__(self) -> int:
        return len(self.docs)

    def get_item(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        doc = self.docs[index]

        max_start = len(doc) - self.block_size - 1
        start = torch.randint(0, max_start + 1, (1,)).item()

        x = doc[start:start + self.block_size]
        y = doc[start + 1:start + self.block_size + 1]

        return x, y