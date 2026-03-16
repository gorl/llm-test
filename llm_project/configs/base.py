from dataclasses import dataclass
import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    batch_size: int = 32
    block_size: int = 64
    max_steps: int = 1500
    eval_interval: int = 100
    eval_steps: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_split: float = 0.9
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    device: str = detect_device()


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    d_model: int = 128
    n_heads: int = 4
    hidden_dim: int = 512
    dropout: float = 0.05
    n_layers: int = 4

# d_model = 384
# n_heads = 6
# head_dim = 64