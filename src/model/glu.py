"""Gated linear unit feed-forward: the expert body, with or without routing."""

import torch
from torch import nn

from .utils import Activation, resolve_activation


def _affine(x: torch.Tensor, W: torch.Tensor, b: "torch.Tensor | None") -> torch.Tensor:
    """x @ W (+ b), fusing the bias into the GEMM epilogue where the shape allows."""
    if b is None:
        return x @ W
    return torch.addmm(b, x, W) if x.dim() == 2 else x @ W + b


def glu_hidden_dim(d_ff: int) -> int:
    """
    GLU hidden width for a d_ff-wide FFN budget: 2/3 * d_ff.

    A GLU holds three d_model x d_hidden matrices where a vanilla FFN holds two of
    d_model x d_ff, so 3 * d_hidden = 2 * d_ff grounds the parameter count to the FFN
    being replaced. All three share the one width -- gate and up are multiplied
    elementwise, so they cannot differ, and W_down must match what it consumes -- which
    leaves 2/3 as the only ratio that keeps the count fixed. Spend more via d_ff.
    """
    d_hidden = 2 * d_ff // 3
    if d_hidden < 1:
        raise ValueError(f"d_ff ({d_ff}) is too small: 2/3 * d_ff rounds down to {d_hidden}.")
    return d_hidden


def make_glu(act: Activation, fused: bool = True, dynamic: "bool | None" = True):
    """
    Build the pointwise half of a GLU: (gate, up) -> act(gate) * up.

    That is two kernels in eager over a [tokens, d_hidden] tensor, and pointwise work
    is roughly a third of a GLU block's runtime, so it is compiled into one. Pass
    fused=False if inductor/triton is unavailable.

    dynamic=True suits the MoE, where every expert gets a different token count each
    step and shape-specializing would recompile forever. A dense block sees the same
    [B*T, d_model] every step, so it passes dynamic=None and lets inductor specialize.
    """
    def glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return act(gate) * up

    return torch.compile(glu, dynamic=dynamic) if fused else glu


class GLUFeedForward(nn.Module):
    """
    Dense GLU feed-forward block: the same body MoEBlock gives each expert, unrouted.

        h = act(x @ W_gate) * (x @ W_up)        # [d_model] -> [d_hidden]
        y = h @ W_down                          # [d_hidden] -> [d_model]

    Args:
        d_model: model (input/output) width.
        d_ff: the FFN width this replaces, which sets the parameter budget and hence
            the GLU hidden width (see `glu_hidden_dim`). Defaults to 4 * d_model.
        activation: gate activation name or callable (see `utils.ACTIVATIONS`).
        bias: add biases to the projections.
        fused: compile the pointwise GLU (see `make_glu`).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: "int | None" = None,
        activation: "str | Activation" = "silu",
        bias: bool = False,
        fused: bool = True,
        std: float = 0.02,
    ):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.d_model = d_model
        self.d_hidden = glu_hidden_dim(d_ff)
        self.activation, act = resolve_activation(activation)
        # Static shapes here, unlike the MoE: let inductor specialize on them.
        self.glu = make_glu(act, fused, dynamic=None)

        # gate and up are fused into one matrix, so a single matmul feeds both branches.
        self.W_in = nn.Parameter(torch.randn(d_model, 2 * self.d_hidden) * std)
        self.W_out = nn.Parameter(torch.randn(self.d_hidden, d_model) * std)

        # Biases are off by default: GLU FFNs generally drop them at no cost in quality.
        self.b_in = nn.Parameter(torch.zeros(2 * self.d_hidden)) if bias else None
        self.b_out = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {tuple(x.shape)}.")

        # addmm folds the bias into the GEMM epilogue, where `x @ W` then `+ b` is a
        # second kernel over the whole [tokens, 2*d_hidden] result. Only the 2D case:
        # addmm does not broadcast over leading dims, so wider inputs take the plain path.
        h = _affine(x, self.W_in, self.b_in)

        # One matmul produced both branches side by side; split and gate them.
        gate, up = h.chunk(2, dim=-1)
        return _affine(self.glu(gate, up), self.W_out, self.b_out)
