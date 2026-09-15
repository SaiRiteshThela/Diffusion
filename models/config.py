from collections.abc import Callable
from pathlib import Path

import torch.nn.functional as F
import yaml

ACTIVATIONS: dict[str, Callable] = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_activation(activation: str | Callable) -> Callable:
    if callable(activation):
        return activation
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}, expected one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[activation]
