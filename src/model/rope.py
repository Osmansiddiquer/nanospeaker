"""Rotary position embeddings, with the p-RoPE low-frequency truncation."""

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1): the 90-degree half of the 2D rotation, on the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_dim: int):
    """
    Rotate the leading `rot_dim` dims of x by (cos, sin), passing the tail through.

    Written as the plane-by-plane (x1*cos - x2*sin, x2*cos + x1*sin) rather than via
    `rotate_half`, which is the same arithmetic without the intermediate negated copy,
    and emits one concatenation covering both rotated halves and the untouched tail.
    Eager, that is still several kernels and temporaries over a tensor the op only needs
    to read and write once; compiled, inductor collapses it into a single elementwise
    kernel with one output allocation.
    """
    half = rot_dim // 2
    x1, x2 = x[..., :half], x[..., half:rot_dim]
    rotated = (x1 * cos - x2 * sin, x2 * cos + x1 * sin)
    if rot_dim == x.shape[-1]:
        return torch.cat(rotated, dim=-1)
    return torch.cat((*rotated, x[..., rot_dim:]), dim=-1)


def make_rotate(fused: bool = True):
    """
    Build the rotation body, compiled unless asked otherwise.

    dynamic=True because T moves constantly -- every decode step is a different query
    length than the prefill that preceded it -- and recompiling per length would cost
    far more than the fusion saves. Pass fused=False where inductor is unavailable.
    """
    return torch.compile(_rotate, dynamic=True) if fused else _rotate


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary position embedding (Su et al., 2021) with p-RoPE truncation.

    RoPE splits a head into d_head/2 planes and rotates plane j at a position m by
    m * theta_j, with theta_j = base ** (-2j / d_head). A query at m and a key at n then
    meet at an angle theta_j * (m - n), so the attention logit depends on positions only
    through their difference.

    p-RoPE (Barbero et al., 2024, "Round and Round We Go!") keeps the p fraction of
    *highest* frequencies rotating and leaves the rest unrotated. The lowest frequencies
    barely turn over a whole context, so instead of encoding distance they act as an
    almost-constant channel that the model uses to build a decay prior -- which costs it
    the ability to attend far away on those dims. Zeroing their rotation hands those dims
    back as clean NoPE channels; p = 0.75 is the paper's pick, p = 1 is plain RoPE and
    p = 0 is NoPE. The relative property survives: unrotated dims contribute q . k, a term
    with no position in it at all.

    Rotation uses the "rotate half" pairing (dim i with dim i + rot_dim/2) over the
    leading `rot_dim = 2 * round(p * d_head / 2)` dims, the tail passing through
    untouched. Which dim pairs with which is arbitrary -- the projections that feed this
    are learned -- so a contiguous rotated prefix is chosen for locality, matching the
    usual partial-rotary implementations.

    Args:
        d_head: per-head width (`d_k` in the attention block). Odd widths leave the
            final dim unrotated.
        base: the theta base; larger stretches the wavelengths for longer contexts.
        p: fraction of frequencies that keep rotating, from 1 (RoPE) to 0 (NoPE).
        max_seq_len: positions to precompute. Not a limit -- the table grows on demand.

    Shapes: `forward` takes [..., T, d_head] (i.e. per head, as [B, n_heads, T, d_k]).
    One instance is meant to be shared by every layer: the tables are buffers, not
    parameters, and depend only on (d_head, base, p).
    """

    def __init__(
        self,
        d_head: int,
        base: float = 10_000.0,
        p: float = 1.0,
        max_seq_len: int = 4096,
        fused: bool = True,
    ):
        super().__init__()
        if d_head < 2:
            raise ValueError(f"d_head ({d_head}) must be >= 2 to hold a rotation plane.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ({p}) must lie in [0, 1].")

        self.d_head, self.base, self.p = d_head, base, p
        self.rotate = make_rotate(fused)

        # Tables cast to the compute dtype, keyed by it. Casting inside `forward` instead
        # allocated a fresh pair every call, which is invisible next to the unfused
        # rotation but dominates it once the rotation is one kernel.
        self._cast_cache: dict = {}

        # Truncation is over frequencies (rotation planes), so it is always an even
        # number of dims that rotate. round() keeps p = 0.75 of 32 planes at exactly 24.
        n_freqs = d_head // 2
        self.n_rot = round(p * n_freqs)
        self.rot_dim = 2 * self.n_rot

        # theta_j over the *kept* (highest) frequencies. The exponent uses the full
        # d_head, so truncating drops planes rather than rescaling the ones that remain.
        inv_freq = base ** (
            -torch.arange(self.n_rot, dtype=torch.float32) * 2.0 / d_head
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # One entry per rotation plane, not per dim: `_rotate` pairs dim i with i + n_rot
        # and multiplies both by the same angle. Kept in fp32, cast (once) at use: the
        # angles are what need the precision, and they are computed here.
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self._build(max_seq_len, inv_freq.device)

    def _build(self, seq_len: int, device: torch.device) -> None:
        """
        (Re)compute the cos/sin table out to at least `seq_len` positions.

        inv_freq is recomputed rather than read back, so this also repairs the table
        after a `.to(bfloat16)` on the module: buffers convert with everything else, and
        bf16 would quantize the frequencies and the angles they generate to about two
        decimal places. The table is what carries position, so it is never kept below
        fp32 -- fp64 only if the module was explicitly widened to it.
        """
        dtype = torch.float64 if self.cos.dtype == torch.float64 else torch.float32
        inv_freq = self.base ** (
            -torch.arange(self.n_rot, dtype=dtype, device=device) * 2.0 / self.d_head
        )
        self.inv_freq = inv_freq

        pos = torch.arange(seq_len, dtype=dtype, device=device)
        angles = pos[:, None] * inv_freq[None, :]                   # [T, n_rot]
        self.cos, self.sin = angles.cos(), angles.sin()
        self._cast_cache = {}                                       # the old casts are stale

    def _table(self, seq_len: int, device: torch.device):
        """cos/sin covering `seq_len` positions, growing (by doubling) if they do not."""
        if (
            seq_len > self.cos.shape[0]
            or self.cos.device != device
            or self.cos.dtype not in (torch.float32, torch.float64)
        ):
            self._build(max(seq_len, 2 * self.cos.shape[0]), device)
        return self.cos, self.sin

    def _cast(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """The tables in the compute dtype, cast once per dtype rather than per call."""
        cos, sin = self._table(seq_len, device)
        if cos.dtype == dtype:
            return cos, sin
        if dtype not in self._cast_cache:
            self._cast_cache[dtype] = (cos.to(dtype), sin.to(dtype))
        return self._cast_cache[dtype]

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        positions: "torch.Tensor | None" = None,
        max_pos: "int | None" = None,
    ) -> torch.Tensor:
        """
        Rotate [..., T, d_head] by position.

        Args:
            offset: position of x[..., 0, :]; the rest follow contiguously. This is the
                number of tokens already in the KV cache during incremental decoding.
            positions: explicit int positions, [T] or [B, T], overriding `offset`. For
                packed sequences or any layout where position is not row order.
            max_pos: largest value in `positions`, when the caller knows it. Reading it
                off the tensor instead means a device->host copy, which stalls the
                pipeline until every kernel queued behind it has finished.
        """
        if x.shape[-1] != self.d_head:
            raise ValueError(f"expected last dim {self.d_head}, got {tuple(x.shape)}.")
        if self.rot_dim == 0:                      # p == 0: pure NoPE, nothing to do
            return x

        # Cast first, slice second: slicing a cached table is a view, where casting a
        # slice would allocate a fresh pair of tensors on every call.
        T = x.shape[-2]
        if positions is None:
            cos, sin = self._cast(offset + T, x.device, x.dtype)
            cos, sin = cos[offset : offset + T], sin[offset : offset + T]
        else:
            bound = max_pos if max_pos is not None else int(positions.max())
            cos, sin = self._cast(bound + 1, x.device, x.dtype)
            cos, sin = cos[positions], sin[positions]
            if positions.dim() == 2:               # [B, T, rot] -> broadcast over heads
                cos, sin = cos[:, None], sin[:, None]

        return self.rotate(x, cos, sin, self.rot_dim)

    def extra_repr(self) -> str:
        return (
            f"d_head={self.d_head}, base={self.base}, p={self.p}, "
            f"rotated_dims={self.rot_dim}/{self.d_head}"
        )
