
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm


class BPETokenizer:
    def __init__(self, vocab_size: int = 8000):
        self._vocab_size = vocab_size

        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"

        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

        self.bos_id = None
        self.eos_id = None

    def fit(self, texts: list[str]):
        print("Training BPE tokenizer...")

        trainer = BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=[
                self.unk_token,
                self.bos_token,
                self.eos_token,
            ],
        )

        self.tokenizer.train_from_iterator(texts, trainer)

        self.bos_id = self.tokenizer.token_to_id(self.bos_token)
        self.eos_id = self.tokenizer.token_to_id(self.eos_token)

        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_id),
                (self.eos_token, self.eos_id),
            ],
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            return self.tokenizer.encode(text).ids

        old_processor = self.tokenizer.post_processor
        self.tokenizer.post_processor = None
        try:
            ids = self.tokenizer.encode(text).ids
        finally:
            self.tokenizer.post_processor = old_processor

        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def encode_all(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        batch_size: int = 2048,
        show_progress: bool = True,
        progress_desc: str = "Tokenizing",
    ) -> list[list[int]]:
        result: list[list[int]] = []

        old_processor = None
        if not add_special_tokens:
            old_processor = self.tokenizer.post_processor
            self.tokenizer.post_processor = None

        try:
            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc=progress_desc, total=(len(texts) + batch_size - 1) // batch_size)

            for i in iterator:
                batch = texts[i:i + batch_size]
                encodings = self.tokenizer.encode_batch(batch)
                result.extend(enc.ids for enc in encodings)
        finally:
            if not add_special_tokens:
                self.tokenizer.post_processor = old_processor

        return result

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def state_dict(self) -> dict:
        return {
            "tokenizer_json": self.tokenizer.to_str(),
            "vocab_size": self._vocab_size,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "BPETokenizer":
        tok = cls(vocab_size=state.get("vocab_size", 8000))
        tok.tokenizer = Tokenizer.from_str(state["tokenizer_json"])

        tok.tokenizer.pre_tokenizer = ByteLevel()
        tok.tokenizer.decoder = ByteLevelDecoder()

        tok.bos_id = tok.tokenizer.token_to_id(tok.bos_token)
        tok.eos_id = tok.tokenizer.token_to_id(tok.eos_token)

        if tok.bos_id is not None and tok.eos_id is not None:
            tok.tokenizer.post_processor = TemplateProcessing(
                single=f"{tok.bos_token} $A {tok.eos_token}",
                special_tokens=[
                    (tok.bos_token, tok.bos_id),
                    (tok.eos_token, tok.eos_id),
                ],
            )

        return tok

# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import ByteLevel
# from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# from tokenizers.processors import TemplateProcessing



# class BPETokenizer:
#     def __init__(self, vocab_size: int = 8000):
#         self._vocab_size = vocab_size

#         self.bos_token = "[BOS]"
#         self.eos_token = "[EOS]"
#         self.unk_token = "[UNK]"

#         self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
#         self.tokenizer.pre_tokenizer = ByteLevel()
#         self.tokenizer.decoder = ByteLevelDecoder()

#         self.bos_id = None
#         self.eos_id = None

#     def fit(self, texts: list[str]):
#         print("Training BPE tokenizer...")

#         trainer = BpeTrainer(
#             vocab_size=self._vocab_size,
#             special_tokens=[
#                 self.unk_token,
#                 self.bos_token,
#                 self.eos_token,
#             ],
#         )

#         self.tokenizer.train_from_iterator(texts, trainer)

#         # сохраняем id токенов
#         self.bos_id = self.tokenizer.token_to_id(self.bos_token)
#         self.eos_id = self.tokenizer.token_to_id(self.eos_token)

#         # автоматически добавляем BOS/EOS внутри fast tokenizer
#         self.tokenizer.post_processor = TemplateProcessing(
#             single=f"{self.bos_token} $A {self.eos_token}",
#             special_tokens=[
#                 (self.bos_token, self.bos_id),
#                 (self.eos_token, self.eos_id),
#             ],
#         )

#     def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
#         if add_special_tokens:
#             return self.tokenizer.encode(text).ids

#         # временно кодируем без post_processor
#         old_processor = self.tokenizer.post_processor
#         self.tokenizer.post_processor = None
#         try:
#             ids = self.tokenizer.encode(text).ids
#         finally:
#             self.tokenizer.post_processor = old_processor

#         return ids

#     def decode(self, ids: list[int]) -> str:
#         return self.tokenizer.decode(ids)

#     def encode_all(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
#         if add_special_tokens:
#             return [enc.ids for enc in self.tokenizer.encode_batch(texts)]

#         # временно кодируем batch без post_processor
#         old_processor = self.tokenizer.post_processor
#         self.tokenizer.post_processor = None
#         try:
#             return [enc.ids for enc in self.tokenizer.encode_batch(texts)]
#         finally:
#             self.tokenizer.post_processor = old_processor

#     @property
#     def vocab_size(self):
#         return self.tokenizer.get_vocab_size()

#     def state_dict(self) -> dict:
#         return {
#             "tokenizer_json": self.tokenizer.to_str(),
#             "vocab_size": self._vocab_size,
#         }

#     @classmethod
#     def from_state_dict(cls, state: dict) -> "BPETokenizer":
#         tok = cls(vocab_size=state.get("vocab_size", 8000))
#         tok.tokenizer = Tokenizer.from_str(state["tokenizer_json"])

#         # на всякий случай восстанавливаем декодер / pre_tokenizer
#         tok.tokenizer.pre_tokenizer = ByteLevel()
#         tok.tokenizer.decoder = ByteLevelDecoder()

#         tok.bos_id = tok.tokenizer.token_to_id(tok.bos_token)
#         tok.eos_id = tok.tokenizer.token_to_id(tok.eos_token)

#         # восстанавливаем post_processor для автодобавления BOS/EOS
#         if tok.bos_id is not None and tok.eos_id is not None:
#             tok.tokenizer.post_processor = TemplateProcessing(
#                 single=f"{tok.bos_token} $A {tok.eos_token}",
#                 special_tokens=[
#                     (tok.bos_token, tok.bos_id),
#                     (tok.eos_token, tok.eos_id),
#                 ],
#             )

#         return tok
    
# # class BPETokenizer:
# #     def __init__(self, vocab_size: int = 8000):
# #         self._vocab_size = vocab_size

# #         self.bos_token = "[BOS]"
# #         self.eos_token = "[EOS]"
# #         self.unk_token = "[UNK]"

# #         self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
# #         self.tokenizer.pre_tokenizer = ByteLevel()
# #         self.tokenizer.decoder = ByteLevelDecoder()

# #     def fit(self, texts: list[str]):
# #         print("Training BPE tokenizer...")

# #         trainer = BpeTrainer(
# #             vocab_size=self._vocab_size,
# #             special_tokens=[
# #                 self.unk_token,
# #                 self.bos_token,
# #                 self.eos_token,
# #             ],
# #         )

# #         self.tokenizer.train_from_iterator(texts, trainer)

# #         # сохраняем id токенов
# #         self.bos_id = self.tokenizer.token_to_id(self.bos_token)
# #         self.eos_id = self.tokenizer.token_to_id(self.eos_token)

# #     def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
# #         ids = self.tokenizer.encode(text).ids

# #         if add_special_tokens:
# #             ids = [self.bos_id] + ids + [self.eos_id]

# #         return ids

# #     def decode(self, ids: list[int]) -> str:
# #         return self.tokenizer.decode(ids)

# #     def encode_all(self, texts: list[str]) -> list[list[int]]:
# #         return [self.encode(text) for text in texts]

# #     @property
# #     def vocab_size(self):
# #         return self.tokenizer.get_vocab_size()

# #     def state_dict(self) -> dict:
# #         return {
# #             "tokenizer_json": self.tokenizer.to_str(),
# #             "vocab_size": self._vocab_size,
# #         }

# #     @classmethod
# #     def from_state_dict(cls, state: dict) -> "BPETokenizer":
# #         tok = cls(vocab_size=state.get("vocab_size", 8000))
# #         tok.tokenizer = Tokenizer.from_str(state["tokenizer_json"])

# #         tok.bos_id = tok.tokenizer.token_to_id("[BOS]")
# #         tok.eos_id = tok.tokenizer.token_to_id("[EOS]")

# #         return tok