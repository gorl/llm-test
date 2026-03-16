# Tiny LLM Baseline Project

A minimal modular decoder-only language model project in Python/PyTorch.

## What is included

- character tokenizer
- language-modeling dataset
- random batch sampler
- token + positional embeddings
- single-head causal self-attention
- feed-forward network
- transformer block
- decoder-only LM
- trainer with validation and checkpoints
- autoregressive text generation

## Python version

Tested with Python 3.11+

## Create virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

Train on the bundled tiny corpus:

```bash
python -m llm_project.experiments.train_char_gpt --input data/sample.txt
```

After training, generate text:

```bash
python -m llm_project.experiments.generate --checkpoint checkpoints/last.pt --prompt "mama "
```

## Useful commands via Makefile

```bash
make install
make train
make generate PROMPT="mama "
make clean
```

## Project structure

- `llm_project/tokenizers` — tokenizer implementations
- `llm_project/data` — datasets
- `llm_project/batching` — batch samplers
- `llm_project/model` — model building blocks
- `llm_project/training` — trainer and checkpointing
- `llm_project/inference` — generator / decoding
- `llm_project/experiments` — runnable entry points
- `data/sample.txt` — bundled sample corpus

## Notes

- This is intentionally small and readable.
- The tokenizer is char-level to keep the first version simple.
- The default config is CPU-friendly.
- You can replace one component at a time without changing the whole pipeline.
