from __future__ import annotations

import argparse
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
) -> Iterable[tuple[str, dict[str, Any]]]:
    """
    Round-robin по всем источникам.
    Если у источника задан max_docs, после этого лимита он выключается.
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

            src.seen_docs += 1
            yield src.source_id, row
            next_alive.append(idx)

        alive = next_alive


def batched_texts(
    rows_iter: Iterable[tuple[str, dict[str, Any]]],
    batch_size: int = 4096,
    total_texts: int | None = None,
) -> Iterable[list[str]]:
    batch: list[str] = []

    with tqdm(desc="Preparing texts", unit="texts", total=total_texts) as pbar:
        for _source_id, row in rows_iter:
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
        prefix: str = "tokens",
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


def encode_stream_to_bins(
    tokenizer,
    rows_iter: Iterable[tuple[str, dict[str, Any]]],
    out_dir: Path,
    batch_size: int = 2048,
    shard_size_tokens: int = 1_000_000,
    total_texts_hint: int | None = None,
) -> dict[str, Any]:
    if tokenizer.vocab_size >= 65536:
        raise ValueError(
            f"Vocab size {tokenizer.vocab_size} does not fit into uint16. "
            f"Use uint32 instead."
        )

    writer = BinShardWriter(
        out_dir=out_dir,
        prefix="tokens",
        shard_size_tokens=shard_size_tokens,
        dtype=np.uint16,
    )

    eos_id = tokenizer.eos_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_id is None")

    texts_batch: list[str] = []
    batch_source_ids: list[str] = []
    total_texts = 0
    texts_per_source: dict[str, int] = {}
    tokens_per_source: dict[str, int] = {}

    try:
        with tqdm(desc="Encoding texts", unit="texts", total=total_texts_hint) as pbar:
            for source_id, row in rows_iter:
                text = extract_text(row)
                if text is None:
                    continue

                texts_batch.append(text)
                batch_source_ids.append(source_id)
                total_texts += 1
                texts_per_source[source_id] = texts_per_source.get(source_id, 0) + 1
                pbar.update(1)

                if len(texts_batch) >= batch_size:
                    encoded = tokenizer.encode_all(
                        texts_batch,
                        add_special_tokens=False,
                        batch_size=batch_size,
                        show_progress=False,
                    )

                    flat_tokens: list[int] = []
                    for src_id, ids in zip(batch_source_ids, encoded):
                        doc_tokens = len(ids) + 1  # + eos
                        tokens_per_source[src_id] = tokens_per_source.get(src_id, 0) + doc_tokens
                        flat_tokens.extend(ids)
                        flat_tokens.append(eos_id)

                    writer.write_tokens(flat_tokens)
                    texts_batch.clear()
                    batch_source_ids.clear()

            if texts_batch:
                encoded = tokenizer.encode_all(
                    texts_batch,
                    add_special_tokens=False,
                    batch_size=batch_size,
                    show_progress=False,
                )

                flat_tokens: list[int] = []
                for src_id, ids in zip(batch_source_ids, encoded):
                    doc_tokens = len(ids) + 1
                    tokens_per_source[src_id] = tokens_per_source.get(src_id, 0) + doc_tokens
                    flat_tokens.extend(ids)
                    flat_tokens.append(eos_id)

                writer.write_tokens(flat_tokens)
                texts_batch.clear()
                batch_source_ids.clear()

    finally:
        writer.close()

    return {
        "total_tokens": writer.total_tokens,
        "total_texts": total_texts,
        "num_shards": writer.total_shards,
        "shard_size_tokens": shard_size_tokens,
        "dtype": "uint16",
        "texts_per_source": texts_per_source,
        "tokens_per_source": tokens_per_source,
    }


def prepare_data(
    dataset_info: list[dict[str, Any]],
    out_dir: Path,
    tokenizer_batch_size: int = 4096,
    encode_batch_size: int = 2048,
    shard_size_tokens: int = 1_000_000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    total_texts_hint = get_progress_total(dataset_info)

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

    print("Encoding dataset to bin shards...")
    rows_iter = iter_merged_rows(dataset_info)
    stats = encode_stream_to_bins(
        tokenizer=tokenizer,
        rows_iter=rows_iter,
        out_dir=out_dir,
        batch_size=encode_batch_size,
        shard_size_tokens=shard_size_tokens,
        total_texts_hint=total_texts_hint,
    )

    datasets_meta: list[dict[str, Any]] = []
    for ds_cfg in dataset_info:
        sid = source_id_from_cfg(ds_cfg)
        datasets_meta.append(
            {
                **ds_cfg,
                "source_id": sid,
                "encoded_texts": stats["texts_per_source"].get(sid, 0),
                "encoded_tokens": stats["tokens_per_source"].get(sid, 0),
            }
        )

    meta = {
        "datasets": datasets_meta,
        "tokenizer_type": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "dtype": stats["dtype"],
        "total_tokens": stats["total_tokens"],
        "total_texts": stats["total_texts"],
        "num_shards": stats["num_shards"],
        "shard_size_tokens": stats["shard_size_tokens"],
        "files": [f"tokens_{i:03d}.bin" for i in range(stats["num_shards"])],
    }

    print("Writing meta.json...")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved prepared data to: {out_dir}")


def parse_dataset_info(raw: str) -> list[dict[str, Any]]:
    dataset_info = json.loads(raw)
    if not isinstance(dataset_info, list):
        raise ValueError("--datasets must be a JSON array")

    for i, item in enumerate(dataset_info):
        if not isinstance(item, dict):
            raise ValueError(f"--datasets[{i}] must be an object")
        if "path" not in item:
            raise ValueError(f"--datasets[{i}] must contain 'path'")
        if "split" not in item:
            raise ValueError(f"--datasets[{i}] must contain 'split'")

        if "max_docs" in item and item["max_docs"] is not None:
            if not isinstance(item["max_docs"], int) or item["max_docs"] <= 0:
                raise ValueError(f"--datasets[{i}].max_docs must be positive int or null")

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
    args = parser.parse_args()

    dataset_info = load_dataset_info_from_json(Path(args.datasets_config))
    out_dir = Path(args.output_dir)

    prepare_data(
        dataset_info=dataset_info,
        out_dir=out_dir,
        tokenizer_batch_size=args.tokenizer_batch_size,
        encode_batch_size=args.encode_batch_size,
        shard_size_tokens=args.shard_size_tokens,
    )


if __name__ == "__main__":
    main()