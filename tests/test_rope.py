"""Tests for rotary position embeddings and the p-RoPE truncation."""

import pytest
import torch

from src.model.rope import RotaryPositionalEmbedding, rotate_half

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make(d_head=64, **kw):
    """fused=False keeps the bulk of the suite off the inductor compile path."""
    return RotaryPositionalEmbedding(d_head, **{"fused": False, **kw}).double()


# --- geometry ----------------------------------------------------------------------

def test_rotation_preserves_norm():
    """A rotation only turns the head; its length must come out unchanged."""
    rope = make()
    x = torch.randn(2, 4, 16, 64, dtype=torch.double)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-12)


def test_position_zero_is_identity():
    rope = make()
    x = torch.randn(2, 4, 1, 64, dtype=torch.double)
    assert torch.allclose(rope(x), x, atol=1e-12)


def test_logits_depend_only_on_relative_position():
    """The whole point: q at m and k at n meet at an angle set by m - n alone."""
    rope = make(p=0.75)
    q, k = torch.randn(1, 1, 1, 64, dtype=torch.double), torch.randn(1, 1, 1, 64, dtype=torch.double)

    def logit(m, n):
        return (rope(q, offset=m) * rope(k, offset=n)).sum()

    assert torch.allclose(logit(7, 3), logit(104, 100), atol=1e-10)
    assert not torch.allclose(logit(7, 3), logit(7, 4), atol=1e-6)


def test_offset_matches_slicing_a_longer_sequence():
    """Decoding a token at position t must equal running the prefix and taking row t."""
    rope = make()
    x = torch.randn(2, 4, 12, 64, dtype=torch.double)
    full = rope(x)
    for t in (0, 5, 11):
        step = rope(x[:, :, t : t + 1], offset=t)
        assert torch.allclose(step, full[:, :, t : t + 1], atol=1e-12)


def test_explicit_positions_match_offset():
    rope = make()
    x = torch.randn(2, 4, 6, 64, dtype=torch.double)
    pos = torch.arange(9, 15)
    assert torch.allclose(rope(x, positions=pos), rope(x, offset=9), atol=1e-12)

    # Per-sequence positions broadcast over heads, so row order need not be position.
    per_batch = pos.expand(2, 6)
    assert torch.allclose(rope(x, positions=per_batch), rope(x, offset=9), atol=1e-12)


def test_rotate_half_is_a_quarter_turn():
    x = torch.randn(3, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x)


# --- p-RoPE ------------------------------------------------------------------------

@pytest.mark.parametrize("p, expected", [(1.0, 64), (0.75, 48), (0.5, 32), (0.0, 0)])
def test_truncation_keeps_p_of_the_frequencies(p, expected):
    assert make(p=p).rot_dim == expected


def test_p_one_is_plain_rope_and_p_zero_is_nope():
    x = torch.randn(2, 4, 9, 64, dtype=torch.double)
    full = make(p=1.0)
    assert full.rot_dim == full.d_head
    assert torch.allclose(make(p=0.0)(x, offset=5), x)          # nothing rotates
    assert not torch.allclose(full(x, offset=5), x)


def test_truncated_dims_pass_through_untouched():
    """The dropped low frequencies must become clean NoPE channels, not damaged ones."""
    rope = make(p=0.75)
    x = torch.randn(2, 4, 9, 64, dtype=torch.double)
    out = rope(x, offset=13)
    assert torch.allclose(out[..., rope.rot_dim :], x[..., rope.rot_dim :], atol=1e-12)
    assert not torch.allclose(out[..., : rope.rot_dim], x[..., : rope.rot_dim])


def test_truncation_matches_full_rope_on_the_kept_dims():
    """p only removes rotations; the frequencies it keeps are the same ones as p = 1."""
    part, full = make(p=0.5), make(p=1.0)
    x = torch.randn(2, 4, 9, 64, dtype=torch.double)
    d = part.rot_dim

    # rotate_half pairs dim i with i + rot_dim/2, so the kept planes of the truncated
    # module are dims (0..d/2) x (d/2..d), which under full RoPE are dims i and i + 32.
    keep = torch.cat((torch.arange(d // 2), torch.arange(32, 32 + d // 2)))
    x_full = torch.zeros_like(x)
    x_full[..., keep] = x[..., :d]
    assert torch.allclose(part(x, offset=6)[..., :d], full(x_full, offset=6)[..., keep])


def test_p_rope_logits_still_relative():
    """The unrotated dims contribute a constant q . k, so relativity survives."""
    rope = make(p=0.75)
    q, k = torch.randn(1, 1, 1, 64, dtype=torch.double), torch.randn(1, 1, 1, 64, dtype=torch.double)
    pairs = [(3, 1), (50, 48), (300, 298)]
    logits = [(rope(q, offset=m) * rope(k, offset=n)).sum() for m, n in pairs]
    assert all(torch.allclose(logits[0], v, atol=1e-9) for v in logits)


def test_low_frequencies_are_the_ones_dropped():
    """Truncation must cut the slow planes; the fast ones carry the local signal."""
    rope = make(p=0.5, base=10_000.0)
    full = make(p=1.0, base=10_000.0)
    assert torch.allclose(rope.inv_freq, full.inv_freq[: rope.n_rot])
    assert rope.inv_freq[0] > rope.inv_freq[-1]                  # high -> low


# --- table management --------------------------------------------------------------

def test_table_grows_past_max_seq_len():
    rope = make(max_seq_len=8)
    x = torch.randn(1, 1, 40, 64, dtype=torch.double)
    out = rope(x)                                                # must not raise
    assert rope.cos.shape[0] >= 40
    assert torch.allclose(out[:, :, :4], rope(x[:, :, :4]), atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_runs_on_device_and_dtype(device, dtype):
    rope = RotaryPositionalEmbedding(32, p=0.75).to(device)
    x = torch.randn(2, 3, 5, 32, device=device, dtype=dtype)
    out = rope(x, offset=2)
    assert out.dtype == dtype and out.device.type == device


def test_odd_head_width_leaves_the_last_dim_alone():
    rope = make(d_head=7, p=1.0)
    x = torch.randn(1, 1, 4, 7, dtype=torch.double)
    assert rope.rot_dim == 6
    assert torch.allclose(rope(x, offset=3)[..., 6:], x[..., 6:])


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(1)
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(64, p=1.5)
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(64)(torch.randn(1, 1, 2, 32))


def test_low_precision_module_keeps_a_full_precision_table():
    """`model.to(bfloat16)` must not quantize the angles: the table carries position."""
    rope = RotaryPositionalEmbedding(64, p=0.75).to(torch.bfloat16)
    x = torch.randn(1, 1, 8, 64, dtype=torch.bfloat16)
    out = rope(x, offset=300)

    assert rope.cos.dtype == torch.float32                  # rebuilt, not left in bf16
    assert out.dtype == torch.bfloat16                      # cast back at use

    # Same rotation as a full-precision module, up to bf16's own resolution.
    ref = RotaryPositionalEmbedding(64, p=0.75)(x.float(), offset=300)
    assert torch.allclose(out.float(), ref, atol=2e-2)


def test_double_module_builds_a_double_table():
    rope = RotaryPositionalEmbedding(64).double()
    rope(torch.randn(1, 1, 5000, 64, dtype=torch.double))   # forces a rebuild
    assert rope.cos.dtype == torch.float64


# --- the fused rotation ------------------------------------------------------------

@pytest.mark.parametrize("p", [1.0, 0.75])
def test_compiled_rotation_matches_the_eager_one(p):
    """The fused path is an optimization, so it has to be numerically the same op."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 40, 64)
    fused = RotaryPositionalEmbedding(64, p=p, fused=True)
    eager = RotaryPositionalEmbedding(64, p=p, fused=False)
    assert torch.allclose(fused(x, offset=7), eager(x, offset=7), atol=1e-6)


def test_rotation_form_matches_rotate_half():
    """_rotate inlines rotate_half's algebra; pin that they agree."""
    torch.manual_seed(0)
    rope = make(p=0.75)
    x = torch.randn(1, 2, 6, 64, dtype=torch.double)
    cos, sin = rope._cast(20, x.device, x.dtype)
    cos, sin = cos[3:9], sin[3:9]

    d = rope.rot_dim
    dup = torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)
    reference = torch.cat(
        (x[..., :d] * dup[0] + rotate_half(x[..., :d]) * dup[1], x[..., d:]), dim=-1
    )
    assert torch.allclose(rope(x, offset=3), reference, atol=1e-12)


def test_cast_tables_are_reused_not_rebuilt():
    rope = make(p=1.0)
    x = torch.randn(1, 1, 4, 64)                       # float32 against a float64 table
    rope(x), rope(x)
    assert torch.float32 in rope._cast_cache
    assert rope._cast_cache[torch.float32][0] is rope._cast_cache[torch.float32][0]


def test_growing_the_table_invalidates_the_casts():
    rope = make(max_seq_len=8)
    rope(torch.randn(1, 1, 4, 64))                     # populates the float32 cast
    rope(torch.randn(1, 1, 200, 64))                   # forces a rebuild
    cached = rope._cast_cache[torch.float32][0]
    assert cached.shape[0] == rope.cos.shape[0] >= 200


def test_max_pos_avoids_reading_the_positions_tensor():
    rope = make()
    x = torch.randn(2, 4, 6, 64, dtype=torch.double)
    pos = torch.arange(9, 15)
    assert torch.allclose(rope(x, positions=pos, max_pos=14), rope(x, positions=pos))
