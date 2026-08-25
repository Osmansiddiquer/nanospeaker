import torch
import torch.nn.functional as F
from torch import nn

class RMSNorm(nn.Module):
    """
    Root-mean-square normalization (Zhang & Sennrich, 2019): LayerNorm without the mean.

    Dropping the centering step costs nothing measurable in quality and removes a
    reduction, a subtraction and the shift parameter. What remains is a rescale by the
    RMS of the features, which is the part that actually stabilizes the residual stream.

    `F.rms_norm` is still the right call here, but not for the reason first written down:
    on torch 2.6+cu126 it is *not* one fused kernel. Profiling one call shows generic
    `reduce_kernel` and `elementwise_kernel` launches -- the same pow/mean/rsqrt chain,
    in fp32, just spelled by ATen instead of by us. What it does keep is the autograd
    footprint: writing the maths out holds two fp32 copies of the activation alive per
    call, which at 80 calls in a 20-layer model measured 614 MiB.

    So the memory argument stands and the speed argument never did. The speed is
    recovered instead by compiling the module (see train.py `--compile-norms`), which
    fuses the chain into one inductor kernel and is worth 27% of the training step.
    """

    def __init__(self, d: int, epsilon: float = 1e-6, fused: bool = True):
        super().__init__()
        self.d, self.epsilon, self.fused = d, epsilon, fused
        self.gamma = nn.Parameter(torch.ones(d))       # no beta: RMSNorm does not shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return F.rms_norm(x, (self.d,), self.gamma, self.epsilon)

        # The same arithmetic, written out. Kept for reading, not for running.
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return self.gamma * (x * rms.to(x.dtype))


class LayerNormalization(nn.Module):
    """
    Normalizes the input across the features dimension (last dimension) for each sample 
    in the batch. Additionally applies learnable scaling and shifting parameters 
    (gamma and beta) to the normalized output.

    Different from Batch Normalization, which normalizes across the batch dimension, 
    Layer Normalization is applied independently to each sample in the batch. 
    This makes it more suitable for tasks where the batch size may vary or be small, 
    such as in sequence modeling tasks.
    """
    def __init__(self, d_model: int, epsilon=1e-5, fused: bool = True):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon # ensures that this doesn't blow up when the variance is very small leading to floating point instability. also division by zero.
        self.gamma = nn.Parameter(torch.ones(d_model)) # per-feature learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model)) # per-feature learnable shift
        self.fused = fused

    def forward(self, x:torch.Tensor):
        # F.layer_norm is one fused kernel with a hand-written backward. The expanded
        # form below is the same arithmetic spelled out -- mean, var, subtract, sqrt,
        # divide, scale, shift -- which is ~6 kernels and several full-size temporaries,
        # twice per transformer block. Keep fused=False to read (or debug) the maths.
        if self.fused:
            return F.layer_norm(x, (self.d_model,), self.gamma, self.beta, self.epsilon)

        # x is (d1, d2, d3, ..., d_n). normalize across d_n
        mean = x.mean(dim = -1, keepdim=True)
        # Biased variance (population), matching torch.nn.LayerNorm; sqrt(var + eps)
        # keeps the divisor away from zero instead of dividing by a bare std.
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.epsilon) + self.beta


        