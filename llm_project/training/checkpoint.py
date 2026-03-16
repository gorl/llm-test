from __future__ import annotations

import os
import torch


def _strip_orig_mod_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _add_orig_mod_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        (key if key.startswith(prefix) else f"{prefix}{key}"): value
        for key, value in state_dict.items()
    }


def model_state_dict(model) -> dict[str, torch.Tensor]:
    base_model = getattr(model, "_orig_mod", model)
    return base_model.state_dict()


def load_model_state(model, state_dict: dict[str, torch.Tensor]) -> None:
    target_is_compiled = hasattr(model, "_orig_mod")
    has_compiled_keys = any(key.startswith("_orig_mod.") for key in state_dict)

    if target_is_compiled and not has_compiled_keys:
        state_dict = _add_orig_mod_prefix(state_dict)
    elif not target_is_compiled and has_compiled_keys:
        state_dict = _strip_orig_mod_prefix(state_dict)

    model.load_state_dict(state_dict)


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
