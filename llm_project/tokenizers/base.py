from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def fit(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state: dict):
        raise NotImplementedError
