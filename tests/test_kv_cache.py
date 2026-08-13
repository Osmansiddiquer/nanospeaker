"""Tests for the KV cache and the cached/RoPE paths through multi-head attention."""

import pytest
import torch

from src.model.kv_cache import KVCache, make_kv_caches
from src.model.mha import OptimizedMultiHeadAttention, causal_mask, ring_mask
from src.model.rope import RotaryPositionalEmbedding

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make_mha(rope=True, mask=True, d_k=16, **kw):
    r = RotaryPositionalEmbedding(d_k, p=0.75).double() if rope else None
    return OptimizedMultiHeadAttention(
        **{"d_model": 32, "n_heads": 4, "d_k": d_k, "d_v": d_k, "mask": mask, "rope": r, **kw}
    ).double()


def kv(B=2, H=4, T=3, d=8, dtype=torch.double, device="cpu"):
    return (
        torch.randn(B, H, T, d, dtype=dtype, device=device),
        torch.randn(B, H, T, d, dtype=dtype, device=device),
    )


# --- cache mechanics ---------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_appends_in_order(device):
    cache = KVCache(max_seq_len=16)
    k1, v1 = kv(T=3, device=device)
    k2, v2 = kv(T=1, device=device)
    cache.update(k1, v1)
    k_all, v_all = cache.update(k2, v2)

    assert cache.pos == len(cache) == 4
    assert torch.equal(k_all, torch.cat((k1, k2), dim=2))
    assert torch.equal(v_all, torch.cat((v1, v2), dim=2))


def test_returns_only_the_filled_prefix():
    cache = KVCache(max_seq_len=64)
    k_all, _ = cache.update(*kv(T=5))
    assert k_all.shape[2] == 5 and cache.k.shape[2] == 64


def test_allocates_lazily_from_the_first_update():
    cache = KVCache(max_seq_len=10)
    assert cache.k is None
    k, v = kv(B=3, H=2, T=1, d=8)
    cache.update(k, torch.randn(3, 2, 1, 5, dtype=torch.double))     # d_v != d_k
    assert cache.k.shape == (3, 2, 10, 8) and cache.v.shape == (3, 2, 10, 5)
    assert cache.k.dtype == torch.double


def test_reset_keeps_the_buffers():
    cache = KVCache(max_seq_len=8)
    cache.update(*kv(T=4))
    buf = cache.k
    cache.reset()
    assert cache.pos == 0 and cache.k is buf
    k, _ = kv(T=2)
    assert torch.equal(cache.update(k, k)[0], k)                     # overwrites cleanly


def test_reorder_gathers_the_batch():
    cache = KVCache(max_seq_len=8)
    k, v = kv(B=3, T=2)
    cache.update(k, v)
    cache.reorder(torch.tensor([2, 2, 0]))
    k_all, _ = cache.update(*kv(B=3, T=1))
    assert torch.equal(k_all[:, :, :2], k[[2, 2, 0]])


def test_overflow_is_an_error_not_a_silent_drop():
    cache = KVCache(max_seq_len=4)
    cache.update(*kv(T=3))
    with pytest.raises(ValueError, match="overflow"):
        cache.update(*kv(T=2))
    assert cache.pos == 3                                            # left usable


def test_rejects_shape_mismatches():
    with pytest.raises(ValueError, match="must be >= 1"):
        KVCache(0)
    with pytest.raises(ValueError, match=r"\[B, n_heads, T, d\]"):
        KVCache(8).update(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="disagree"):
        KVCache(8).update(torch.randn(2, 4, 3, 8), torch.randn(2, 4, 2, 8))

    cache = KVCache(8)
    cache.update(*kv(B=2, T=1))
    with pytest.raises(ValueError, match="allocated for"):
        cache.update(*kv(B=3, T=1))


def test_make_kv_caches_are_independent():
    caches = make_kv_caches(3, max_seq_len=8)
    caches[0].update(*kv(T=2))
    assert [c.pos for c in caches] == [2, 0, 0]


# --- masking -----------------------------------------------------------------------

def test_causal_mask_is_bottom_right_aligned():
    """Cached queries sit at the end of the key sequence, not the start."""
    m = causal_mask(T=2, S=5, offset=3, device="cpu")
    assert m.tolist() == [[True] * 4 + [False], [True] * 5]
    assert causal_mask(3, 3, 0, "cpu").tolist() == torch.ones(3, 3).tril().bool().tolist()


# --- attention with a cache --------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rope", [True, False])
def test_incremental_decode_matches_full_forward(device, rope):
    """Token by token through the cache must equal one causal pass over the sequence."""
    mha = make_mha(rope=rope).to(device)
    x = torch.randn(2, 7, 32, dtype=torch.double, device=device)
    full = mha(x)

    cache = KVCache(max_seq_len=16)
    steps = [mha(x[:, t : t + 1], cache=cache) for t in range(7)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_prefill_then_decode_matches_full_forward(device):
    """The realistic split: a prompt in one pass, then one token at a time."""
    mha = make_mha().to(device)
    x = torch.randn(2, 9, 32, dtype=torch.double, device=device)
    full = mha(x)

    cache = KVCache(max_seq_len=32)
    out = [mha(x[:, :5], cache=cache)]
    out += [mha(x[:, t : t + 1], cache=cache) for t in range(5, 9)]
    assert torch.allclose(torch.cat(out, dim=1), full, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_prefill_matches_full_forward(device):
    """Multi-token steps at a nonzero offset: the path that needs the explicit mask."""
    mha = make_mha().to(device)
    x = torch.randn(2, 12, 32, dtype=torch.double, device=device)
    full = mha(x)

    cache = KVCache(max_seq_len=16)
    out = [mha(x[:, i : i + 4], cache=cache) for i in range(0, 12, 4)]
    assert torch.allclose(torch.cat(out, dim=1), full, atol=1e-10)


def test_attention_list_path_matches_fast_path():
    """The explicit-map fallback must agree with SDPA, cached or not."""
    mha = make_mha()
    x = torch.randn(2, 6, 32, dtype=torch.double)
    A, out = mha(x, get_attention_list=True)
    assert torch.allclose(out, mha(x), atol=1e-12)
    assert len(A) == mha.n_heads and A[0].shape == (2, 6, 6)
    assert torch.allclose(A[0].sum(-1), torch.ones(2, 6, dtype=torch.double))
    assert torch.allclose(A[0].triu(1), torch.zeros(2, 6, 6, dtype=torch.double))  # causal

    cache = KVCache(max_seq_len=8)
    mha(x[:, :4], cache=cache)
    A_step, out_step = mha(x[:, 4:5], cache=cache, get_attention_list=True)
    assert A_step[0].shape == (2, 1, 5)                       # attends the whole prefix

    cache.reset()
    assert torch.allclose(out_step, mha(x[:, :5], cache=cache)[:, -1:], atol=1e-10)


def test_cache_is_not_needed_for_the_uncached_path():
    """Training keeps the old signature: no cache, no positions to thread."""
    mha = make_mha(rope=False, mask=False)
    x = torch.randn(2, 6, 32, dtype=torch.double)
    assert mha(x).shape == (2, 6, 32)


def test_rope_moves_the_output():
    """A block with RoPE must not be a no-op relative to one without it."""
    torch.manual_seed(0)
    with_rope = make_mha(rope=True)
    torch.manual_seed(0)
    without = make_mha(rope=False)
    x = torch.randn(2, 6, 32, dtype=torch.double)
    assert not torch.allclose(with_rope(x), without(x))


def test_rope_width_must_match_d_k():
    with pytest.raises(ValueError, match="d_head"):
        OptimizedMultiHeadAttention(
            32, 4, d_k=16, d_v=16, rope=RotaryPositionalEmbedding(8)
        )


# --- grouped-query attention -------------------------------------------------------

def make_gqa(n_kv_heads, n_heads=8, d_k=16, **kw):
    return OptimizedMultiHeadAttention(
        **{"d_model": 32, "n_heads": n_heads, "d_k": d_k, "d_v": d_k, "mask": True,
           "n_kv_heads": n_kv_heads, **kw}
    ).double()


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4, 8])
def test_kv_projections_shrink_with_the_group_count(n_kv_heads):
    """The point of GQA: fewer KV heads, so smaller K/V weights and a smaller cache."""
    m = make_gqa(n_kv_heads)
    assert m.W_q.shape == (32, 8 * 16)
    assert m.W_k.shape == m.W_v.shape == (32, n_kv_heads * 16)
    assert m.n_groups == 8 // n_kv_heads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_kv_heads", [1, 2, 8])
def test_gqa_fast_path_matches_the_explicit_maps(device, n_kv_heads):
    """enable_gqa must broadcast KV heads the same way the reference expansion does."""
    m = make_gqa(n_kv_heads).to(device)
    x = torch.randn(2, 6, 32, dtype=torch.double, device=device)
    _, explicit = m(x, get_attention_list=True)
    assert torch.allclose(m(x), explicit, atol=1e-10)


@pytest.mark.parametrize("n_kv_heads", [1, 2])
def test_gqa_cache_stores_only_kv_heads(n_kv_heads):
    m = make_gqa(n_kv_heads)
    cache = KVCache(max_seq_len=8)
    m(torch.randn(2, 3, 32, dtype=torch.double), cache=cache)
    assert cache.k.shape == (2, n_kv_heads, 8, 16)


@pytest.mark.parametrize("n_kv_heads", [1, 4])
def test_gqa_decodes_incrementally(n_kv_heads):
    m = make_gqa(n_kv_heads)
    x = torch.randn(2, 5, 32, dtype=torch.double)
    cache = KVCache(max_seq_len=8)
    steps = [m(x[:, t : t + 1], cache=cache) for t in range(5)]
    assert torch.allclose(torch.cat(steps, dim=1), m(x), atol=1e-10)


def test_mqa_is_gqa_with_one_kv_head():
    """MQA and MHA are the same block at the two ends of n_kv_heads."""
    assert make_gqa(1).n_groups == 8 and make_gqa(8).n_groups == 1


def test_n_kv_heads_must_divide_n_heads():
    for bad in (0, 3, 5, 16):
        with pytest.raises(ValueError, match="n_kv_heads"):
            make_gqa(bad)


# --- sliding-window attention ------------------------------------------------------

def test_window_mask_keeps_only_the_band():
    m = causal_mask(T=4, S=4, offset=0, device="cpu", window=2)
    assert m.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, True, True],
    ]


def test_window_mask_with_an_offset():
    """A cached decode step sees the window ending at its own absolute position."""
    m = causal_mask(T=1, S=6, offset=5, device="cpu", window=3)
    assert m.tolist() == [[False, False, False, True, True, True]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("window", [1, 3, 8])
def test_windowed_attention_matches_the_explicit_maps(device, window):
    mha = make_mha(mask=True, window=window).to(device)
    x = torch.randn(2, 10, 32, dtype=torch.double, device=device)
    A, explicit = mha(x, get_attention_list=True)
    assert torch.allclose(mha(x), explicit, atol=1e-10)

    # Nothing outside the band may carry weight.
    banded = causal_mask(10, 10, 0, device, window)
    assert torch.allclose(A[0][:, ~banded], torch.zeros(1, dtype=torch.double, device=device))


def test_window_of_one_attends_only_to_self():
    """Degenerate but sharp: the output must be V of the token itself, projected."""
    mha = make_mha(rope=False, mask=True, window=1)
    x = torch.randn(2, 5, 32, dtype=torch.double)
    A, _ = mha(x, get_attention_list=True)
    assert torch.allclose(A[0], torch.eye(5, dtype=torch.double).expand(2, 5, 5))


def test_wide_window_equals_full_causal_attention():
    torch.manual_seed(0)
    windowed = make_mha(mask=True, window=64)
    torch.manual_seed(0)
    full = make_mha(mask=True)
    x = torch.randn(2, 9, 32, dtype=torch.double)
    assert torch.allclose(windowed(x), full(x), atol=1e-12)


def test_narrow_window_changes_the_output():
    torch.manual_seed(0)
    windowed = make_mha(mask=True, window=2)
    torch.manual_seed(0)
    full = make_mha(mask=True)
    x = torch.randn(2, 9, 32, dtype=torch.double)
    assert not torch.allclose(windowed(x), full(x))


@pytest.mark.parametrize("device", DEVICES)
def test_windowed_incremental_decode_matches_full_forward(device):
    """The cache keeps every key, so the window has to be enforced by position."""
    mha = make_mha(mask=True, window=3).to(device)
    x = torch.randn(2, 9, 32, dtype=torch.double, device=device)
    cache = KVCache(max_seq_len=16)
    steps = [mha(x[:, t : t + 1], cache=cache) for t in range(9)]
    assert torch.allclose(torch.cat(steps, dim=1), mha(x), atol=1e-10)


def test_windowed_chunked_prefill_matches_full_forward():
    mha = make_mha(mask=True, window=4)
    x = torch.randn(2, 12, 32, dtype=torch.double)
    cache = KVCache(max_seq_len=16)
    out = [mha(x[:, i : i + 3], cache=cache) for i in range(0, 12, 3)]
    assert torch.allclose(torch.cat(out, dim=1), mha(x), atol=1e-10)


def test_window_and_gqa_compose():
    m = OptimizedMultiHeadAttention(32, 8, 16, 16, mask=True, n_kv_heads=2, window=3).double()
    x = torch.randn(2, 7, 32, dtype=torch.double)
    _, explicit = m(x, get_attention_list=True)
    assert torch.allclose(m(x), explicit, atol=1e-10)


def test_bad_window_configs_raise():
    with pytest.raises(ValueError, match="window"):
        OptimizedMultiHeadAttention(32, 4, 8, 8, mask=True, window=0)
    with pytest.raises(ValueError, match="mask=True"):
        OptimizedMultiHeadAttention(32, 4, 8, 8, mask=False, window=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the flex path is CUDA-only")
@pytest.mark.parametrize("n_kv_heads", [4, 1])
def test_flex_window_matches_the_materialized_mask(n_kv_heads):
    """The block-sparse kernel and the masked SDPA must compute the same attention."""
    torch.manual_seed(0)
    m = OptimizedMultiHeadAttention(
        64, 4, 16, 16, mask=True, window=48, n_kv_heads=n_kv_heads
    ).cuda()
    x = torch.randn(1, 192, 64, device="cuda")

    assert m._use_flex(x, 192), "flex path did not engage on a config built for it"
    flex_out = m(x)

    m.flex = False                                    # same block, materialized mask
    assert not m._use_flex(x, 192)
    assert torch.allclose(flex_out, m(x), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the flex path is CUDA-only")
def test_flex_is_declined_where_it_would_be_slower():
    """Short query blocks, float64 and CPU all belong on the materialized-mask path."""
    m = OptimizedMultiHeadAttention(64, 4, 16, 16, mask=True, window=48).cuda()
    x = torch.randn(1, 192, 64, device="cuda")
    assert not m._use_flex(x, 1)                      # decode step
    assert not m._use_flex(x.double(), 192)           # unsupported dtype
    assert not m._use_flex(x.cpu(), 192)              # CPU flex is ~25x slower


# --- rolling (window-sized) cache --------------------------------------------------

def test_rolling_cache_holds_only_the_window():
    cache = KVCache(max_seq_len=1024, window=4)
    for _ in range(10):
        cache.update(*kv(T=1))
    assert cache.capacity == 4 and cache.k.shape[2] == 4      # 1024 never allocated
    assert cache.pos == 10                                    # absolute, keeps counting


def test_rolling_cache_keeps_the_most_recent_tokens():
    cache = KVCache(max_seq_len=64, window=3)
    steps = [kv(T=1) for _ in range(5)]
    for k, v in steps:
        k_all, _ = cache.update(k, v)

    assert k_all.shape[2] == 3
    # Ring order, so compare as a set of positions rather than a sequence.
    assert sorted(cache.positions().tolist()) == [2, 3, 4]
    stored = {tuple(k_all[0, 0, j].tolist()) for j in range(3)}
    assert stored == {tuple(steps[i][0][0, 0, 0].tolist()) for i in (2, 3, 4)}


def test_rolling_cache_fills_before_it_wraps():
    cache = KVCache(max_seq_len=64, window=4, max_chunk=2)
    k_all, _ = cache.update(*kv(T=2))
    assert k_all.shape[2] == 2 and cache.positions().tolist() == [0, 1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("window", [2, 3, 8])
def test_windowed_decode_through_a_rolling_cache_matches_full_forward(device, window):
    """The real test: a ring that drops evicted keys must still decode identically."""
    mha = make_mha(mask=True, window=window).to(device)
    x = torch.randn(2, 10, 32, dtype=torch.double, device=device)
    full = mha(x)

    cache = KVCache(max_seq_len=64, window=window)
    steps = [mha(x[:, t : t + 1], cache=cache) for t in range(10)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_rolling_and_full_caches_decode_the_same(device):
    """Same block, same window; only how much history the cache keeps differs."""
    mha = make_mha(mask=True, window=4).to(device)
    x = torch.randn(2, 12, 32, dtype=torch.double, device=device)

    rolling = KVCache(max_seq_len=64, window=4)
    full = KVCache(max_seq_len=64)
    a = torch.cat([mha(x[:, t : t + 1], cache=rolling) for t in range(12)], dim=1)
    b = torch.cat([mha(x[:, t : t + 1], cache=full) for t in range(12)], dim=1)
    assert torch.allclose(a, b, atol=1e-10)
    assert rolling.k.shape[2] == 4 and full.k.shape[2] == 64      # 16x less KV memory


def test_rolling_cache_rejects_a_step_it_cannot_serve():
    """A T-token step needs window + T - 1 slots; refuse rather than drop keys silently."""
    cache = KVCache(max_seq_len=64, window=4)
    with pytest.raises(ValueError, match="max_chunk >= 4"):
        cache.update(*kv(T=4))


def test_rolling_cache_serves_a_chunk_when_sized_for_it():
    mha = make_mha(mask=True, window=4)
    x = torch.randn(2, 12, 32, dtype=torch.double)
    cache = KVCache(max_seq_len=64, window=4, max_chunk=3)
    out = [mha(x[:, i : i + 3], cache=cache) for i in range(0, 12, 3)]
    assert torch.allclose(torch.cat(out, dim=1), mha(x), atol=1e-10)


def test_ring_mask_uses_positions_not_slots():
    pos = torch.tensor([4, 5, 2, 3])                       # a wrapped ring
    m = ring_mask(T=1, offset=5, key_pos=pos, window=3)
    assert m.tolist() == [[True, True, False, True]]       # positions 3, 4, 5 are in band


def test_block_and_cache_windows_must_agree():
    mha = make_mha(mask=True, window=4)
    with pytest.raises(ValueError, match="rolls a 8-token window"):
        mha(torch.randn(2, 1, 32, dtype=torch.double), cache=KVCache(64, window=8))


def test_reset_clears_the_ring_positions():
    cache = KVCache(max_seq_len=64, window=3)
    for _ in range(5):
        cache.update(*kv(T=1))
    cache.reset()
    assert cache.pos == 0 and cache.positions().numel() == 0


# --- the mask caches ----------------------------------------------------------------

def test_mask_caches_are_bounded():
    """Chunked prefill and decode both walk the key, so an uncapped cache is a leak."""
    from src.model import mha as mha_mod

    mha_mod._MASKS.clear()
    for offset in range(200):
        causal_mask(2, 300, offset, "cpu", window=4)
    assert len(mha_mod._MASKS) <= mha_mod._CACHE_SIZE


def test_mask_cache_returns_the_same_tensor_for_the_same_shape():
    from src.model import mha as mha_mod

    mha_mod._MASKS.clear()
    a = causal_mask(4, 4, 0, "cpu")
    assert causal_mask(4, 4, 0, "cpu") is a
