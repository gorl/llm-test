from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def load_tokenizer(path: str):
    path = Path(path)

    with open(path, "rb") as f:
        tok = pickle.load(f)

    return tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    hf_tok = tok.tokenizer

    vocab = hf_tok.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    print()
    print("Tokenizer info")
    print("--------------")
    print("vocab_size:", hf_tok.get_vocab_size())
    print("bos_token:", getattr(tok, "bos_token", None))
    print("eos_token:", getattr(tok, "eos_token", None))
    print("unk_token:", getattr(tok, "unk_token", None))
    print("bos_id:", getattr(tok, "bos_id", None))
    print("eos_id:", getattr(tok, "eos_id", None))
    print()

    print("Vocabulary")
    print("----------")

    count = 0
    for token_id in sorted(inv_vocab):
        token = inv_vocab[token_id]

        print(
            f"id={token_id:5d}  "
            f"token={repr(token)}"
        )

        count += 1
        if args.limit and count >= args.limit:
            break

    print()
    print("Total tokens:", len(inv_vocab))

    # merges
    try:
        model = hf_tok.model
        merges = getattr(model, "merges", None)

        if merges:
            print()
            print("BPE merges")
            print("----------")
            for i, m in enumerate(merges[:50]):
                print(m)

            print("...")
            print("total merges:", len(merges))

    except Exception:
        pass


if __name__ == "__main__":
    main()