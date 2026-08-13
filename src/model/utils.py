from typing import Callable

import torch
from torch import nn

Activation = Callable[[torch.Tensor], torch.Tensor]

# Gate activations by name. In a GLU these name the variants: "silu"/"swish" gives
# SwiGLU, "gelu" GEGLU, "relu" ReGLU, "sigmoid" the original GLU.
ACTIVATIONS: dict[str, Activation] = {
    "silu": torch.nn.functional.silu,
    "swish": torch.nn.functional.silu,
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


def initialize_normal_torch_weights(M, N, std = 0.02) -> nn.Parameter:
    """Initialize weights with a normal distribution"""
    return nn.Parameter(torch.randn(M,N) * std)    # std = 0.02. from gpt. leaf tensor


def resolve_activation(activation: "str | Activation") -> tuple[str, Activation]:
    """
    Look up an activation by name, or pass a callable straight through.
    Returns (name, fn).

    An unknown name is a typo, not a request for the default, so it raises rather
    than silently falling back.
    """
    if callable(activation):
        return getattr(activation, "__name__", "custom"), activation
    if isinstance(activation, str) and activation.lower() in ACTIVATIONS:
        return activation.lower(), ACTIVATIONS[activation.lower()]
    raise ValueError(
        f"Unknown activation {activation!r}. Use a callable or one of: "
        f"{', '.join(sorted(ACTIVATIONS))}."
    )
