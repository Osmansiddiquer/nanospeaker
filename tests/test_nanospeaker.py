"""Tests for the assembled nanoSpeaker decoder."""

import pytest
import torch

from src.model.moe import MoEBlock
from src.model.nanospeaker import NanoSpeaker, NanoSpeakerConfig

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def small(**kw):
    """A miniature of the real shape: same structure, small enough to test on CPU."""
    return NanoSpeakerConfig(**{
        "vocab_size": 128, "d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "d_head": 8, "max_seq_len": 64, "n_experts": 6, "n_shared": 1, "top_k": 2,
        "d_expert": 8, "checkpoint": False, **kw,
    })


# --- shape and accounting ----------------------------------------------------------

def test_defaults_are_the_settled_architecture():
    cfg = NanoSpeakerConfig()
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads) == (20, 576, 9, 3)
    assert (cfg.n_experts, cfg.top_k, cfg.n_shared, cfg.d_expert) == (76, 6, 2, 128)
    assert cfg.n_heads * cfg.d_head == cfg.d_model      # heads tile the residual stream
    assert cfg.capacity_factor is None                  # dropless


def test_active_params_count_only_the_routed_experts():
    m = NanoSpeaker(small())
    cfg = m.cfg
    idle = cfg.n_layers * 3 * cfg.d_model * cfg.d_expert * (cfg.n_experts - cfg.top_k)
    assert m.n_active_params() == m.n_params() - idle
    assert m.n_active_params() < m.n_params()


def test_unembed_is_tied_to_the_embedding():
    m = NanoSpeaker(small())
    assert not any(p is not m.embed.weight and p.shape == m.embed.weight.shape
                   for n, p in m.named_parameters() if "embed" not in n)
    logits = m(torch.randint(0, 128, (2, 6)))
    assert logits.shape == (2, 6, 128)


@pytest.mark.parametrize("device", DEVICES)
def test_forward_shapes_and_loss(device):
    m = NanoSpeaker(small()).to(device)
    idx = torch.randint(0, 128, (2, 8), device=device)
    assert m(idx).shape == (2, 8, 128)

    loss, aux = m(idx, idx)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert float(aux) > 0                               # balancing loss is live


# --- gradients ---------------------------------------------------------------------

def test_gradients_reach_every_part_of_the_model():
    m = NanoSpeaker(small())
    loss, aux = m(torch.randint(0, 128, (2, 8)), torch.randint(0, 128, (2, 8)))
    (loss + aux).backward()

    for name in ("embed.weight", "blocks.0.attn.W_q", "blocks.0.attn.W_O",
                 "blocks.0.ffn.W_in", "blocks.0.ffn.W_sh_in", "blocks.0.ffn.router.W",
                 "blocks.0.norm_attn.gamma", "norm_out.gamma"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient at {name}"


def test_qk_norm_is_on_and_learns():
    m = NanoSpeaker(small())
    assert m.blocks[0].attn.q_norm is not None
    loss, _ = m(torch.randint(0, 128, (2, 8)), torch.randint(0, 128, (2, 8)))
    loss.backward()
    assert m.blocks[0].attn.q_norm.gamma.grad.abs().sum() > 0


def test_unckpt_blocks_are_spread_not_clustered():
    cfg = NanoSpeakerConfig(ckpt_skip=2)
    assert sorted(cfg.unckpt_blocks) == [0, 10]          # 20 layers, stride 10
    assert NanoSpeakerConfig().unckpt_blocks == frozenset()
    assert len(NanoSpeakerConfig(ckpt_skip=6).unckpt_blocks) == 6


def test_skipping_checkpointing_on_some_blocks_changes_nothing():
    """A memory-for-recompute trade, per block. The arithmetic must be untouched."""
    torch.manual_seed(0)
    all_ckpt = NanoSpeaker(small(checkpoint=True)).double().train()
    torch.manual_seed(0)
    partial = NanoSpeaker(small(checkpoint=True, ckpt_skip=1)).double().train()

    idx = torch.randint(0, 128, (2, 8))
    a, _ = all_ckpt(idx, idx)
    b, _ = partial(idx, idx)
    a.backward(), b.backward()
    assert torch.allclose(a, b, atol=1e-10)
    assert torch.allclose(all_ckpt.embed.weight.grad, partial.embed.weight.grad, atol=1e-10)


def test_checkpointing_does_not_change_the_result():
    """It is a memory trade, so the numbers must be identical either way."""
    torch.manual_seed(0)
    plain = NanoSpeaker(small(checkpoint=False)).double().train()
    torch.manual_seed(0)
    ckpt = NanoSpeaker(small(checkpoint=True)).double().train()

    idx = torch.randint(0, 128, (2, 8))
    a, _ = plain(idx, idx)
    b, _ = ckpt(idx, idx)
    assert torch.allclose(a, b, atol=1e-10)


# --- inference ---------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_cached_decode_matches_a_full_forward(device):
    """One token at a time through the caches must equal one pass over the sequence."""
    m = NanoSpeaker(small()).to(device).double().eval()
    idx = torch.randint(0, 128, (2, 7), device=device)
    with torch.no_grad():
        full = m(idx)
        caches = m.caches(16)
        steps = [m(idx[:, t : t + 1], caches=caches) for t in range(7)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-9)


def test_generate_extends_the_prompt():
    m = NanoSpeaker(small()).eval()
    idx = torch.randint(0, 128, (2, 5))
    out = m.generate(idx, 4, temperature=0.0)
    assert out.shape == (2, 9)
    assert torch.equal(out[:, :5], idx)                 # prompt preserved
    assert (out < 128).all()


def test_generate_is_deterministic_at_zero_temperature():
    m = NanoSpeaker(small()).eval()
    idx = torch.randint(0, 128, (1, 4))
    assert torch.equal(m.generate(idx, 3, temperature=0.0),
                       m.generate(idx, 3, temperature=0.0))


# --- the MoE wiring ----------------------------------------------------------------

def test_experts_are_dropless_and_shared_experts_present():
    m = NanoSpeaker(small())
    ffn = m.blocks[0].ffn
    assert isinstance(ffn, MoEBlock)
    assert ffn.capacity_factor is None and ffn.n_shared == 1
    assert ffn.W_sh_in is not None

    m(torch.randint(0, 128, (2, 8)))
    assert int(ffn.dropped) == 0                        # nothing is ever turned away


def test_router_gates_are_renormalized():
    """
    The selected gates sum to 1. Unnormalized, top-6 of 76 summed to ~0.08 at init,
    scaling every routed FFN output to a twelfth of itself -- so the model could reduce
    loss by sharpening the router rather than by learning anything in the experts.
    """
    assert NanoSpeaker(small()).blocks[0].ffn.router.normalize_weights is True


# --- interleaved local/global attention --------------------------------------------

def test_layer_windows_are_three_to_one():
    cfg = NanoSpeakerConfig(window=512, global_every=4)
    w = cfg.layer_windows()
    assert len(w) == 20
    assert [i for i, x in enumerate(w) if x is None] == [0, 4, 8, 12, 16]
    assert all(x == 512 for x in w if x is not None)
    assert NanoSpeakerConfig().layer_windows() == [None] * 20      # default untouched


def test_windowed_layer_actually_masks():
    """
    A windowed layer must not see past its span. Compare against the same weights with
    the window removed: if the logits agree, the mask never applied.
    """
    torch.manual_seed(0)
    cfg = small(window=4, global_every=None, max_seq_len=32)
    m = NanoSpeaker(cfg).double().eval()
    torch.manual_seed(0)
    full = NanoSpeaker(small(window=None, max_seq_len=32)).double().eval()

    idx = torch.randint(0, 128, (1, 24))
    with torch.no_grad():
        assert not torch.allclose(m(idx), full(idx), atol=1e-6)


def test_mixed_window_stack_trains():
    m = NanoSpeaker(small(window=4, global_every=2, checkpoint=True)).train()
    loss, aux = m(torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16)))
    (loss + aux).backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_mixed_window_cached_decode_matches_full_forward():
    """Per-layer caches: a global layer handed a windowed ring buffer loses history."""
    m = NanoSpeaker(small(window=8, global_every=2, max_seq_len=32)).double().eval()
    idx = torch.randint(0, 128, (2, 7))
    with torch.no_grad():
        full = m(idx)
        caches = m.caches(16)
        steps = [m(idx[:, t : t + 1], caches=caches) for t in range(7)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-9)


def test_caches_carry_the_per_layer_window():
    m = NanoSpeaker(small(window=8, global_every=2, max_seq_len=32, n_layers=4))
    assert [c.window for c in m.caches(32)] == [None, 8, None, 8]
