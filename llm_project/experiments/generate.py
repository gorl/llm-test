from __future__ import annotations

import argparse

import torch

from llm_project.configs.base import ModelConfig
from llm_project.experiments.active import build_model
from llm_project.inference.generator import Generator
from llm_project.tokenizers.bpe_tokenizer import BPETokenizer
from llm_project.tokenizers.char_tokenizer import CharTokenizer
from llm_project.tokenizers.hf_tokenizer import HFAutoTokenizer
from llm_project.training.checkpoint import load_checkpoint, load_model_state


def load_tokenizer(payload: dict):
    tokenizer_type = payload["config"].get("tokenizer_type", "char")

    if tokenizer_type in ("char", "CharTokenizer"):
        return CharTokenizer.from_state_dict(payload["tokenizer"])

    if tokenizer_type in ("hf_auto", "HFAutoTokenizer"):
        return HFAutoTokenizer.from_state_dict(payload["tokenizer"])

    if tokenizer_type in ("bpe", "BPETokenizer"):
        return BPETokenizer.from_state_dict(payload["tokenizer"])

    raise ValueError(f"Unsupported tokenizer_type in checkpoint: {tokenizer_type}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = load_tokenizer(payload)
    model_cfg = ModelConfig(**payload["config"]["model"])
    model = build_model(model_cfg)
    load_model_state(model, payload["model_state"])
    model.eval()

    print("prompt:", args.prompt)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_ids = prompt_ids[:-1]
    print(prompt_ids)

    x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    generator = Generator(model, block_size=model_cfg.block_size)
    out = generator.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(out[0].tolist())
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
