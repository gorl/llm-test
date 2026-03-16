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