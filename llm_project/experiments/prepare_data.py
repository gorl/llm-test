from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from llm_project.data.hfdataset import load_hf_dataset
from llm_project.tokenizers.bpe_tokenizer import BPETokenizer
from llm_project.tokenizers.char_tokenizer import CharTokenizer


def save_tokenizer(tokenizer, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def encode_to_bin(tokenizer, texts: list[str], out_path: Path, batch_size: int = 2048) -> int:
    total_tokens = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with open(out_path, "wb") as f:
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=f"Writing {out_path.name}"):
            batch = texts[i:i + batch_size]

            encoded = tokenizer.encode_all(
                batch,
                add_special_tokens=False,
                batch_size=batch_size,
                show_progress=False,
            )

            flat: list[int] = []
            for ids in encoded:
                flat.extend(ids)
                flat.append(tokenizer.eos_id)

            arr = np.asarray(flat, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += len(arr)

    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"])
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    texts = load_hf_dataset(args.input)
    print(f"Dataset length (texts): {len(texts)}")

    split_idx = int(len(texts) * args.train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    print(f"Train texts: {len(train_texts)}")
    print(f"Val texts: {len(val_texts)}")

    if args.tokenizer == "bpe":
        tokenizer = BPETokenizer()
    else:
        tokenizer = CharTokenizer()

    print("Fitting tokenizer on train texts...")
    tokenizer.fit(train_texts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("Encoding train texts to disk...")
    train_tokens = encode_to_bin(
        tokenizer=tokenizer,
        texts=train_texts,
        out_path=out_dir / "train.bin",
        batch_size=args.batch_size,
    )

    print("Encoding val texts to disk...")
    val_tokens = encode_to_bin(
        tokenizer=tokenizer,
        texts=val_texts,
        out_path=out_dir / "val.bin",
        batch_size=args.batch_size,
    )

    print("Saving tokenizer...")
    save_tokenizer(tokenizer, out_dir / "tokenizer.pkl")

    meta = {
        "input": args.input,
        "tokenizer_type": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "train_texts": len(train_texts),
        "val_texts": len(val_texts),
        "train_split": args.train_split,
    }

    print("Writing meta.json...")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved prepared data to: {out_dir}")


if __name__ == "__main__":
    main()
    
# from __future__ import annotations

# import argparse
# import json
# import pickle
# from pathlib import Path

# import torch

# from llm_project.data.hfdataset import load_hf_dataset
# from llm_project.tokenizers.bpe_tokenizer import BPETokenizer
# from llm_project.tokenizers.char_tokenizer import CharTokenizer


# def save_tokenizer(tokenizer, path: str) -> None:
#     with open(path, "wb") as f:
#         pickle.dump(tokenizer, f)


# def flatten_with_eos(encoded: list[list[int]], eos_id: int) -> list[int]:
#     all_tokens: list[int] = []
#     for ids in encoded:
#         all_tokens.extend(ids)
#         all_tokens.append(eos_id)
#     return all_tokens


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, required=True)
#     parser.add_argument("--output-dir", type=str, required=True)
#     parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"])
#     parser.add_argument("--train-split", type=float, default=0.9)
#     args = parser.parse_args()

#     out_dir = Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print("Loading dataset...")
#     texts = load_hf_dataset(args.input)
#     print(f"Dataset length (texts): {len(texts)}")

#     split_idx = int(len(texts) * args.train_split)
#     train_texts = texts[:split_idx]
#     val_texts = texts[split_idx:]

#     print(f"Train texts: {len(train_texts)}")
#     print(f"Val texts: {len(val_texts)}")

#     if args.tokenizer == "bpe":
#         tokenizer = BPETokenizer()
#     else:
#         tokenizer = CharTokenizer()

#     print("Fitting tokenizer on train texts...")
#     tokenizer.fit(train_texts)
#     print(f"Vocab size: {tokenizer.vocab_size}")

#     print("Encoding train texts...")
#     train_encoded = tokenizer.encode_all(train_texts, add_special_tokens=False)

#     print("Encoding val texts...")
#     val_encoded = tokenizer.encode_all(val_texts, add_special_tokens=False)

#     train_tokens = flatten_with_eos(train_encoded, tokenizer.eos_id)
#     val_tokens = flatten_with_eos(val_encoded, tokenizer.eos_id)
#     all_tokens_count = len(train_tokens) + len(val_tokens)

#     print(f"Total tokens: {all_tokens_count}")
#     print(f"Train tokens: {len(train_tokens)}")
#     print(f"Val tokens: {len(val_tokens)}")

#     train_tensor = torch.tensor(train_tokens, dtype=torch.int16)
#     val_tensor = torch.tensor(val_tokens, dtype=torch.int16)

#     torch.save(train_tensor, out_dir / "train_tokens.pt")
#     torch.save(val_tensor, out_dir / "val_tokens.pt")
#     save_tokenizer(tokenizer, out_dir / "tokenizer.pkl")

#     meta = {
#         "input": args.input,
#         "tokenizer_type": args.tokenizer,
#         "vocab_size": tokenizer.vocab_size,
#         "total_tokens": all_tokens_count,
#         "train_tokens": len(train_tokens),
#         "val_tokens": len(val_tokens),
#         "train_split": args.train_split,
#         "train_texts": len(train_texts),
#         "val_texts": len(val_texts),
#     }

#     with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
#         json.dump(meta, f, ensure_ascii=False, indent=2)

#     print(f"Saved prepared data to: {out_dir}")


# if __name__ == "__main__":
#     main()