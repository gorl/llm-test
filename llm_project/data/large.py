from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from llm_project.tokenizers.bpe_tokenizer import BPETokenizer


def save_tokenizer(tokenizer, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: Path) -> BPETokenizer:
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def extract_text(row: dict[str, Any]) -> str | None:
    for key in ("text", "content", "document"):
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return None


def source_id_from_cfg(ds_cfg: dict[str, Any]) -> str:
    custom_id = ds_cfg.get("id")
    if isinstance(custom_id, str) and custom_id.strip():
        return custom_id.strip()

    path = ds_cfg["path"]
    name = ds_cfg.get("name")
    split = ds_cfg["split"]
    if name:
        return f"{path}:{name}:{split}"
    return f"{path}:{split}"


def get_progress_total(dataset_info: list[dict[str, Any]]) -> int | None:
    totals: list[int] = []
    for ds_cfg in dataset_info:
        max_docs = ds_cfg.get("max_docs")
        if max_docs is None:
            return None
        totals.append(max_docs)
    return sum(totals)


@dataclass
class SourceState:
    source_id: str
    path: str
    name: str | None
    split: str
    max_docs: int | None
    iterator: Any
    seen_docs: int = 0


def build_source_states(dataset_info: list[dict[str, Any]]) -> list[SourceState]:
    sources: list[SourceState] = []

    for ds_cfg in dataset_info:
        path = ds_cfg["path"]
        name = ds_cfg.get("name")
        split = ds_cfg["split"]
        max_docs = ds_cfg.get("max_docs")

        ds = load_dataset(path, name, split=split, streaming=True)

        sources.append(
            SourceState(
                source_id=source_id_from_cfg(ds_cfg),
                path=path,
                name=name,
                split=split,
                max_docs=max_docs,
                iterator=iter(ds),
            )
        )

    return sources


def iter_merged_rows(
    dataset_info: list[dict[str, Any]],
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """
    Round-robin по всем источникам.
    Возвращает:
      (source_id, source_doc_idx, row)

    source_doc_idx — порядковый номер документа внутри источника, начиная с 0.
    """
    sources = build_source_states(dataset_info)
    alive = list(range(len(sources)))

    while alive:
        next_alive: list[int] = []

        for idx in alive:
            src = sources[idx]

            if src.max_docs is not None and src.seen_docs >= src.max_docs:
                continue

            try:
                row = next(src.iterator)
            except StopIteration:
                continue

            doc_idx = src.seen_docs
            src.seen_docs += 1

            yield src.source_id, doc_idx, row
            next_alive.append(idx)

        alive = next_alive


def batched_texts(
    rows_iter: Iterable[tuple[str, int, dict[str, Any]]],
    batch_size: int = 4096,
    total_texts: int | None = None,
) -> Iterable[list[str]]:
    batch: list[str] = []

    with tqdm(desc="Preparing texts", unit="texts", total=total_texts) as pbar:
        for _source_id, _doc_idx, row in rows_iter:
            text = extract_text(row)
            if text is None:
                continue

            batch.append(text)
            pbar.update(1)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


class BinShardWriter:
    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        shard_size_tokens: int = 1_000_000,
        dtype: np.dtype = np.uint16,
    ) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size_tokens = shard_size_tokens
        self.dtype = dtype

        self.shard_idx = 0
        self.tokens_in_current_shard = 0
        self.total_tokens = 0
        self.total_shards = 0
        self._fp = None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._open_new_shard()

    def _make_path(self, shard_idx: int) -> Path:
        return self.out_dir / f"{self.prefix}_{shard_idx:03d}.bin"

    def _open_new_shard(self) -> None:
        if self._fp is not None:
            self._fp.close()

        path = self._make_path(self.shard_idx)
        self._fp = open(path, "wb")
        self.tokens_in_current_shard = 0
        self.total_shards = self.shard_idx + 1
        print(f"Opened shard: {path}")

    def write_tokens(self, tokens: list[int]) -> None:
        if not tokens:
            return

        arr = np.asarray(tokens, dtype=self.dtype)
        start = 0

        while start < len(arr):
            remaining = self.shard_size_tokens - self.tokens_in_current_shard
            chunk = arr[start:start + remaining]
            chunk.tofile(self._fp)

            written = len(chunk)
            self.tokens_in_current_shard += written
            self.total_tokens += written
            start += written

            if self.tokens_in_current_shard >= self.shard_size_tokens and start < len(arr):
                self.shard_idx += 1
                self._open_new_shard()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def make_doc_split_key(source_id: str, source_doc_idx: int, row: dict[str, Any]) -> str:
    row_id = row.get("id")
    if row_id is not None:
        return f"{source_id}:{row_id}"
    return f"{source_id}:{source_doc_idx}"


def is_val_doc(doc_key: str, val_ratio: float) -> bool:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if val_ratio == 0.0:
        return False

    digest = hashlib.blake2b(doc_key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return value < val_ratio


def encode_stream_to_train_val_bins(
    tokenizer,
    rows_iter: Iterable[tuple[str, int, dict[str, Any]]],
    out_dir: Path,
    batch_size: int = 2048,
    shard_size_tokens: int = 1_000_000,
    total_texts_hint: int | None = None,
    val_ratio: float = 0.01,
) -> dict[str, Any]:
    if tokenizer.vocab_size >= 65536:
        raise ValueError(
            f"Vocab size {tokenizer.vocab_size} does not fit into uint16. "
            f"Use uint32 instead."
        )

    train_writer = BinShardWriter(
        out_dir=out_dir / "train",
        prefix="train",
        shard_size_tokens=shard_size_tokens,
        dtype=np.uint16,
    )
    val_writer = BinShardWriter(
        out_dir=out_dir / "val",
        prefix="val",
        shard_size_tokens=shard_size_tokens,
        dtype=np.uint16,
    )

    eos_id = tokenizer.eos_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_id is None")

    texts_batch: list[str] = []
    batch_source_ids: list[str] = []
    batch_splits: list[str] = []

    total_texts = 0

    split_stats: dict[str, dict[str, Any]] = {
        "train": {
            "total_tokens": 0,
            "total_texts": 0,
            "texts_per_source": {},
            "tokens_per_source": {},
        },
        "val": {
            "total_tokens": 0,
            "total_texts": 0,
            "texts_per_source": {},
            "tokens_per_source": {},
        },
    }

    def flush_batch() -> None:
        nonlocal texts_batch, batch_source_ids, batch_splits
        if not texts_batch:
            return

        encoded = tokenizer.encode_all(
            texts_batch,
            add_special_tokens=False,
            batch_size=batch_size,
            show_progress=False,
        )

        train_tokens: list[int] = []
        val_tokens: list[int] = []

        for src_id, split_name, ids in zip(batch_source_ids, batch_splits, encoded):
            doc_tokens = len(ids) + 1

            split_stats[split_name]["total_texts"] += 1
            split_stats[split_name]["total_tokens"] += doc_tokens

            texts_per_source = split_stats[split_name]["texts_per_source"]
            tokens_per_source = split_stats[split_name]["tokens_per_source"]

            texts_per_source[src_id] = texts_per_source.get(src_id, 0) + 1
            tokens_per_source[src_id] = tokens_per_source.get(src_id, 0) + doc_tokens

            if split_name == "train":
                train_tokens.extend(ids)
                train_tokens.append(eos_id)
            else:
                val_tokens.extend(ids)
                val_tokens.append(eos_id)

        if train_tokens:
            train_writer.write_tokens(train_tokens)
        if val_tokens:
            val_writer.write_tokens(val_tokens)

        texts_batch.clear()
        batch_source_ids.clear()
        batch_splits.clear()

    try:
        with tqdm(desc="Encoding texts", unit="texts", total=total_texts_hint) as pbar:
            for source_id, source_doc_idx, row in rows_iter:
                text = extract_text(row)
                if text is None:
                    continue

                doc_key = make_doc_split_key(source_id, source_doc_idx, row)
                split_name = "val" if is_val_doc(doc_key, val_ratio) else "train"

                texts_batch.append(text)
                batch_source_ids.append(source_id)
                batch_splits.append(split_name)

                total_texts += 1
                pbar.update(1)

                if len(texts_batch) >= batch_size:
                    flush_batch()

            flush_batch()

    finally:
        train_writer.close()
        val_writer.close()

    return {
        "dtype": "uint16",
        "shard_size_tokens": shard_size_tokens,
        "val_ratio": val_ratio,
        "total_texts": total_texts,
        "train": {
            "total_tokens": split_stats["train"]["total_tokens"],
            "total_texts": split_stats["train"]["total_texts"],
            "num_shards": train_writer.total_shards,
            "texts_per_source": split_stats["train"]["texts_per_source"],
            "tokens_per_source": split_stats["train"]["tokens_per_source"],
            "files": [f"train_{i:03d}.bin" for i in range(train_writer.total_shards)],
        },
        "val": {
            "total_tokens": split_stats["val"]["total_tokens"],
            "total_texts": split_stats["val"]["total_texts"],
            "num_shards": val_writer.total_shards,
            "texts_per_source": split_stats["val"]["texts_per_source"],
            "tokens_per_source": split_stats["val"]["tokens_per_source"],
            "files": [f"val_{i:03d}.bin" for i in range(val_writer.total_shards)],
        },
    }


def prepare_data(
    dataset_info: list[dict[str, Any]],
    out_dir: Path,
    reuse_tokenizer: bool = False,
    tokenizer_batch_size: int = 4096,
    encode_batch_size: int = 2048,
    shard_size_tokens: int = 1_000_000,
    val_ratio: float = 0.01,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    total_texts_hint = get_progress_total(dataset_info)

    if not reuse_tokenizer:
        tokenizer = BPETokenizer()

        print("Fitting tokenizer on texts...")
        rows_iter = iter_merged_rows(dataset_info)
        tokenizer.fit(
            batched_texts(
                rows_iter,
                batch_size=tokenizer_batch_size,
                total_texts=total_texts_hint,
            )
        )
        print(f"Vocab size: {tokenizer.vocab_size}")

        print("Saving tokenizer...")
        save_tokenizer(tokenizer, out_dir / "tokenizer.pkl")
    else:
        print("Loading existing tokenizer...")
        tokenizer = load_tokenizer(out_dir / "tokenizer.pkl")
        print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    print("Encoding dataset to train/val bin shards...")
    rows_iter = iter_merged_rows(dataset_info)
    stats = encode_stream_to_train_val_bins(
        tokenizer=tokenizer,
        rows_iter=rows_iter,
        out_dir=out_dir,
        batch_size=encode_batch_size,
        shard_size_tokens=shard_size_tokens,
        total_texts_hint=total_texts_hint,
        val_ratio=val_ratio,
    )

    def build_split_meta(split_name: str) -> dict[str, Any]:
        split_stats = stats[split_name]
        datasets_meta: list[dict[str, Any]] = []

        for ds_cfg in dataset_info:
            sid = source_id_from_cfg(ds_cfg)
            datasets_meta.append(
                {
                    **ds_cfg,
                    "source_id": sid,
                    "encoded_texts": split_stats["texts_per_source"].get(sid, 0),
                    "encoded_tokens": split_stats["tokens_per_source"].get(sid, 0),
                }
            )

        return {
            "split": split_name,
            "datasets": datasets_meta,
            "tokenizer_type": "bpe",
            "vocab_size": tokenizer.vocab_size,
            "dtype": stats["dtype"],
            "total_tokens": split_stats["total_tokens"],
            "total_texts": split_stats["total_texts"],
            "num_shards": split_stats["num_shards"],
            "shard_size_tokens": stats["shard_size_tokens"],
            "files": split_stats["files"],
            "val_ratio": stats["val_ratio"],
        }

    train_meta = build_split_meta("train")
    val_meta = build_split_meta("val")

    print("Writing train/meta.json...")
    with open(out_dir / "train" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=2)

    print("Writing val/meta.json...")
    with open(out_dir / "val" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(val_meta, f, ensure_ascii=False, indent=2)

    common_meta = {
        "tokenizer_type": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "dtype": stats["dtype"],
        "shard_size_tokens": stats["shard_size_tokens"],
        "val_ratio": stats["val_ratio"],
        "total_texts": stats["total_texts"],
        "train": {
            "total_tokens": stats["train"]["total_tokens"],
            "total_texts": stats["train"]["total_texts"],
            "num_shards": stats["train"]["num_shards"],
            "files": [f"train/{name}" for name in stats["train"]["files"]],
        },
        "val": {
            "total_tokens": stats["val"]["total_tokens"],
            "total_texts": stats["val"]["total_texts"],
            "num_shards": stats["val"]["num_shards"],
            "files": [f"val/{name}" for name in stats["val"]["files"]],
        },
    }

    print("Writing root meta.json...")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(common_meta, f, ensure_ascii=False, indent=2)

    print(f"Saved prepared data to: {out_dir}")


def parse_dataset_info(raw: str) -> list[dict[str, Any]]:
    dataset_info = json.loads(raw)
    if not isinstance(dataset_info, list):
        raise ValueError("--datasets-config must be a JSON array")

    for i, item in enumerate(dataset_info):
        if not isinstance(item, dict):
            raise ValueError(f"--datasets-config[{i}] must be an object")
        if "path" not in item:
            raise ValueError(f"--datasets-config[{i}] must contain 'path'")
        if "split" not in item:
            raise ValueError(f"--datasets-config[{i}] must contain 'split'")

        if "max_docs" in item and item["max_docs"] is not None:
            if not isinstance(item["max_docs"], int) or item["max_docs"] <= 0:
                raise ValueError(
                    f"--datasets-config[{i}].max_docs must be positive int or null"
                )

    return dataset_info


def load_dataset_info_from_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return parse_dataset_info(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-config",
        type=str,
        required=True,
        help="Path to JSON config with datasets list",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer-batch-size", type=int, default=4096)
    parser.add_argument("--encode-batch-size", type=int, default=2048)
    parser.add_argument("--shard-size-tokens", type=int, default=1_000_000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--reuse-tokenizer", action="store_true")
    args = parser.parse_args()

    dataset_info = load_dataset_info_from_json(Path(args.datasets_config))
    out_dir = Path(args.output_dir)

    prepare_data(
        dataset_info=dataset_info,
        out_dir=out_dir,
        reuse_tokenizer=args.reuse_tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
        encode_batch_size=args.encode_batch_size,
        shard_size_tokens=args.shard_size_tokens,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()