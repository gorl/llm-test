from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import torch

from llm_project.batching.iter_sampler import PackedBatchSampler
from llm_project.configs.base import ModelConfig, TrainConfig
from llm_project.data.lm_dataset import PackedTokenDataset, resolve_numpy_dtype
from llm_project.experiments.active import build_model
from llm_project.training.checkpoint import load_checkpoint, load_model_state
from llm_project.training.trainer import Trainer
from llm_project.utils.seed import set_seed


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_ratio: float = 0.02,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(max_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        if step >= max_steps:
            return min_lr_ratio

        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay = set()
    no_decay = set()

    whitelist_weight_modules = (torch.nn.Linear,)

    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    if hasattr(torch.nn, "RMSNorm"):
        blacklist_weight_modules = blacklist_weight_modules + (torch.nn.RMSNorm,)

    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters(recurse=False):
            full_name = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(full_name)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(full_name)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(full_name)
            else:
                no_decay.add(full_name)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"
    assert len(param_dict.keys() - union_params) == 0, (
        f"Parameters not separated: {param_dict.keys() - union_params}"
    )

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(optim_groups, lr=lr)


def load_tokenizer(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_prepared_data(data_dir: str):
    data_dir = Path(data_dir)

    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        root_meta = json.load(f)

    with open(data_dir / "train" / "meta.json", "r", encoding="utf-8") as f:
        train_meta = json.load(f)

    with open(data_dir / "val" / "meta.json", "r", encoding="utf-8") as f:
        val_meta = json.load(f)

    with open(data_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    dtype_name = root_meta.get("dtype", "uint16")
    dtype = resolve_numpy_dtype(dtype_name)

    train_paths = [str(data_dir / "train" / name) for name in train_meta["files"]]
    val_paths = [str(data_dir / "val" / name) for name in val_meta["files"]]

    return train_paths, val_paths, tokenizer, root_meta, train_meta, val_meta, dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-heads", type=int, default=6)
    parser.add_argument("--d-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        amp_dtype = None

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        device=device,
    )
    set_seed(train_cfg.seed)

    print("Loading prepared data...")
    train_paths, val_paths, tokenizer, meta, train_meta, val_meta, dtype = load_prepared_data(args.prepared_data)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train shards: {len(train_paths)}")
    print(f"Val shards: {len(val_paths)}")
    print(f"Train tokens: {train_meta['total_tokens']}")
    print(f"Val tokens: {val_meta['total_tokens']}")

    train_dataset = PackedTokenDataset(
        paths=train_paths,
        block_size=train_cfg.block_size,
        dtype=dtype,
    )
    val_dataset = PackedTokenDataset(
        paths=val_paths,
        block_size=train_cfg.block_size,
        dtype=dtype,
    )

    train_sampler = PackedBatchSampler(train_dataset, train_cfg.device)
    val_sampler = PackedBatchSampler(val_dataset, train_cfg.device)

    print("Building model...")
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=train_cfg.block_size,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        n_heads=args.d_heads,
        n_layers=args.d_layers,
    )
    print(f"Model config: {model_cfg}")

    model = build_model(model_cfg).to(train_cfg.device)
    model = torch.compile(model)

    optimizer = build_optimizer(model, train_cfg.learning_rate, train_cfg.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        max_steps=train_cfg.max_steps,
    )

    # Чтобы warmup реально стартовал с нулевого lr.
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0

    start_step = 0

    if args.resume_from is not None:
        payload = load_checkpoint(args.resume_from, map_location=train_cfg.device)
        load_model_state(model, payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])

        if "scheduler_state" in payload and payload["scheduler_state"] is not None:
            scheduler.load_state_dict(payload["scheduler_state"])

        start_step = payload["step"] + 1
        print(f"Resumed training from step {start_step} using checkpoint {args.resume_from}")

    config_dict = {
        "train": vars(train_cfg),
        "model": vars(model_cfg),
        "tokenizer_type": type(tokenizer).__name__,
        "prepared_data": args.prepared_data,
        "meta": meta,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "dtype": str(dtype),
        "train_shards": train_paths,
        "val_shards": val_paths,
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        batch_size=train_cfg.batch_size,
        max_steps=train_cfg.max_steps,
        eval_interval=train_cfg.eval_interval,
        eval_steps=train_cfg.eval_steps,
        grad_clip=train_cfg.grad_clip,
        checkpoint_dir=train_cfg.checkpoint_dir,
        tokenizer=tokenizer,
        config=config_dict,
        start_step=start_step,
        amp_dtype=amp_dtype,
    )
    trainer.train()

    print(f"Checkpoint saved to {os.path.join(train_cfg.checkpoint_dir, 'last.pt')}")


if __name__ == "__main__":
    main()