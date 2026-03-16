# from transformers import AutoTokenizer

# from llm_project.tokenizers.base import Tokenizer
# from tqdm import tqdm

# class HFAutoTokenizer(Tokenizer):
#     def __init__(self, pretrained_name: str = "gpt2") -> None:
#         self.pretrained_name = pretrained_name
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

#     def fit(self, text: str) -> None:
#         # Для готового HF tokenizer ничего обучать не нужно.
#         pass

#     def encode_all(self, texts: list[str]) -> list[int]:
#         tokens = []
#         for text in tqdm(texts, desc="Encoding texts"):
#             if not isinstance(text, str):
#                 raise ValueError("All items in texts must be strings")

#             tokens.extend(self.tokenizer(text, add_special_tokens=False)["input_ids"])

#         return tokens

#     def encode(self, text: str) -> list[int]:
#         return self.tokenizer(text, add_special_tokens=False)["input_ids"]

#     def decode(self, ids: list[int]) -> str:
#         return self.tokenizer.decode(ids)

#     @property
#     def vocab_size(self) -> int:
#         return self.tokenizer.vocab_size

#     def state_dict(self) -> dict:
#         return {"pretrained_name": self.pretrained_name}

#     @classmethod
#     def from_state_dict(cls, state: dict) -> "HFAutoTokenizer":
#         return cls(pretrained_name=state["pretrained_name"])


from transformers import AutoTokenizer

from llm_project.tokenizers.base import Tokenizer
from tqdm import tqdm

class HFAutoTokenizer(Tokenizer):
    def __init__(self, pretrained_name: str = "gpt2") -> None:
        self.pretrained_name = pretrained_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def fit(self, text: str) -> None:
        # Для готового HF tokenizer ничего обучать не нужно.
        pass

    def encode_all(self, texts: list[str]) -> list[int]:
        tokens = []
        for text in tqdm(texts, desc="Encoding texts"):
            if not isinstance(text, str):
                raise ValueError("All items in texts must be strings")

            tokens.extend(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        return tokens

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def state_dict(self) -> dict:
        return {"pretrained_name": self.pretrained_name}

    @classmethod
    def from_state_dict(cls, state: dict) -> "HFAutoTokenizer":
        return cls(pretrained_name=state["pretrained_name"])
