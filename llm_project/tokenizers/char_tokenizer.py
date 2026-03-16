from __future__ import annotations

from llm_project.tokenizers.base import Tokenizer


class CharTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def fit(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        if not self.stoi:
            raise ValueError("Tokenizer is not fitted")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        tok = cls()
        tok.stoi = dict(state["stoi"])
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        return tok
