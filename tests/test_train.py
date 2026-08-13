"""Tests for the training machinery: schedule, optimizer, resume, metrics."""

import argparse

import numpy as np
import pytest
import torch

from src.data.loader import MixtureLoader
from src.model.nanospeaker import NanoSpeaker, NanoSpeakerConfig
from src.train.metrics import MetricsLogger, moe_health, snapshot, update_norm_ratio
from src.train.optim import (Muon, build_optimizers, muon_parameter_names,
                             wsd_lr, zeropower_via_newtonschulz5)
from src.train.train import load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def tiny_cfg(**kw):
    return NanoSpeakerConfig(**{
        "vocab_size": 512, "d_model": 64, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
        "d_head": 32, "max_seq_len": 64, "n_experts": 8, "n_shared": 1, "top_k": 2,
        "d_expert": 16, "checkpoint": False, **kw})


@pytest.fixture
def corpus(tmp_path):
    """Two tiny tokenized sources, shaped like the real ones."""
    for name, n in (("alpha", 20_000), ("beta", 12_000)):
        ids = np.random.default_rng(0).integers(0, 512, n, dtype=np.uint16)
        ids.tofile(tmp_path / f"{name}.bin")
        np.array([0, n], dtype=np.uint64).tofile(tmp_path / f"{name}.idx")
    return tmp_path


def loader(corpus, **kw):
    return MixtureLoader(**{
        "token_dir": str(corpus), "seq_len": 32, "micro_batch": 4, "total_steps": 100,
        "stable_mix": {"alpha": 0.75, "beta": 0.25}, "decay_mix": {"alpha": 1.0}, **kw})


# --- the WSD schedule --------------------------------------------------------------

def test_wsd_is_flat_then_decays_to_zero():
    total = 1000
    assert wsd_lr(0, total) == 1.0                       # no warmup by default
    assert wsd_lr(799, total) == 1.0                     # flat through the stable phase
    assert wsd_lr(800, total) == pytest.approx(1.0)      # knee
    assert wsd_lr(999, total) == pytest.approx(0.0, abs=1e-4)


def test_wsd_decay_is_monotone():
    xs = [wsd_lr(s, 1000) for s in range(800, 1000)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))


def test_wsd_warmup_when_asked():
    assert wsd_lr(0, 1000, warmup=10) == pytest.approx(0.1)
    assert wsd_lr(9, 1000, warmup=10) == pytest.approx(1.0)


def test_wsd_depends_only_on_the_step():
    """Computed, not accumulated -- so a resume cannot land out of phase."""
    assert wsd_lr(850, 1000) == wsd_lr(850, 1000)


# --- Newton-Schulz -----------------------------------------------------------------

def test_newton_schulz_compresses_the_spectrum():
    """
    Five quintic steps do not produce an exactly orthogonal matrix, and are not meant
    to: they pull the singular values into a band around 1. That conditioning is the
    whole effect, so it is what gets asserted rather than gram == I.
    """
    # Deliberately ill-conditioned: a random Gaussian is already well behaved, so it
    # would show almost nothing. Spread the spectrum over two decades instead.
    u, _ = torch.linalg.qr(torch.randn(64, 16))
    v, _ = torch.linalg.qr(torch.randn(16, 16))
    g = u @ torch.diag(torch.logspace(-1, 1, 16)) @ v.T
    before = torch.linalg.svdvals(g)
    after = torch.linalg.svdvals(zeropower_via_newtonschulz5(g).float())

    assert before.max() / before.min() > 50                # ~100:1 going in
    assert after.min() > 0.5 and after.max() < 1.5         # a band around 1 coming out
    assert after.max() / after.min() < 3


def test_newton_schulz_handles_both_orientations():
    for shape in ((64, 16), (16, 64)):
        out = zeropower_via_newtonschulz5(torch.randn(*shape))
        assert out.shape == shape and torch.isfinite(out).all()


def test_newton_schulz_batches_over_stacked_experts():
    """Experts arrive as [E, d_model, d_expert]; each slice must be handled separately."""
    stacked = torch.randn(8, 64, 16)
    out = zeropower_via_newtonschulz5(stacked)
    assert out.shape == stacked.shape
    single = zeropower_via_newtonschulz5(stacked[3])
    assert torch.allclose(out[3].float(), single.float(), atol=1e-2)


# --- parameter routing -------------------------------------------------------------

def test_muon_takes_hidden_matrices_only():
    m = NanoSpeaker(tiny_cfg())
    names = muon_parameter_names(m)
    assert not any("embed" in n or "router" in n for n in names)
    assert all(dict(m.named_parameters())[n].ndim >= 2 for n in names)
    assert any("W_in" in n for n in names) and any("W_q" in n for n in names)


def test_every_parameter_lands_in_exactly_one_optimizer():
    m = NanoSpeaker(tiny_cfg())
    optims, (n_muon, n_embed, n_other) = build_optimizers(m)
    owned = sum(len(g["params"]) for o in optims.values() for g in o.param_groups)
    assert owned == len(list(m.parameters())) == n_muon + n_embed + n_other


# --- resume ------------------------------------------------------------------------

def test_checkpoint_resume_is_bit_exact(tmp_path):
    cfg = tiny_cfg()
    m = NanoSpeaker(cfg)
    optims, _ = build_optimizers(m)
    x = torch.randint(0, 512, (2, 32))

    def step(model, opts):
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        loss, aux = model(x, x)
        (loss + aux).backward()
        for o in opts.values():
            o.step()
        return loss.item()

    for _ in range(2):
        step(m, optims)
    path = tmp_path / "ck" / "step_000002.pt"
    save_checkpoint(path, m, optims, 2, 1234, cfg, argparse.Namespace(steps=10))

    torch.manual_seed(999)                                # deliberately different init
    m2 = NanoSpeaker(cfg)
    optims2, _ = build_optimizers(m2)
    assert load_checkpoint(path, m2, optims2, "cpu") == (2, 1234)

    # Optimizer state must survive too, so the *next* step agrees as well.
    assert step(m, optims) == step(m2, optims2)
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_muon_momentum_stays_bf16_across_resume(tmp_path):
    """load_state_dict casts state to the parameter dtype; that would double the cost."""
    cfg = tiny_cfg()
    m = NanoSpeaker(cfg)
    optims, _ = build_optimizers(m)
    loss, aux = m(torch.randint(0, 512, (2, 32)), torch.randint(0, 512, (2, 32)))
    (loss + aux).backward()
    optims["muon"].step()

    path = tmp_path / "ck" / "step_000001.pt"
    save_checkpoint(path, m, optims, 1, 0, cfg, argparse.Namespace())
    m2 = NanoSpeaker(cfg)
    optims2, _ = build_optimizers(m2)
    load_checkpoint(path, m2, optims2, "cpu")

    dtypes = {s["momentum"].dtype for s in optims2["muon"].state.values()}
    assert dtypes == {torch.bfloat16}


def test_latest_pointer_is_written_after_the_checkpoint(tmp_path):
    cfg = tiny_cfg()
    m = NanoSpeaker(cfg)
    optims, _ = build_optimizers(m)
    path = tmp_path / "ck" / "step_000500.pt"
    save_checkpoint(path, m, optims, 500, 0, cfg, argparse.Namespace())
    assert (path.parent / "latest.txt").read_text().strip() == "step_000500.pt"


# --- the loader --------------------------------------------------------------------

def test_batches_are_a_pure_function_of_step(corpus):
    """The whole resume design rests on this: no state carried between steps."""
    a, ay, amix = loader(corpus).batch(9000, 3)
    b, by, bmix = loader(corpus).batch(9000, 3)          # fresh object
    assert torch.equal(a, b) and torch.equal(ay, by) and amix == bmix


def test_different_steps_draw_different_data(corpus):
    a, _, _ = loader(corpus).batch(10, 0)
    assert not torch.equal(a, loader(corpus).batch(11, 0)[0])
    assert not torch.equal(a, loader(corpus).batch(10, 1)[0])


def test_targets_are_inputs_shifted_by_one(corpus):
    x, y, _ = loader(corpus).batch(5, 0)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_phase_switches_at_decay_start(corpus):
    L = loader(corpus)
    assert L.phase(79) == "stable" and L.phase(80) == "decay"
    assert L.mix(79) == pytest.approx({"alpha": 0.75, "beta": 0.25})
    assert L.mix(85) == {"alpha": 1.0}


def test_realized_mix_matches_the_ratio_within_the_phase(corpus):
    L = loader(corpus)
    tally = {}
    for s in range(60):                                   # stable steps only
        _, _, mix = L.batch(s, 0)
        for k, v in mix.items():
            tally[k] = tally.get(k, 0) + v / 60
    assert tally["alpha"] == pytest.approx(0.75, abs=0.1)


def test_missing_source_is_a_clear_error(corpus):
    with pytest.raises(FileNotFoundError, match="tokenize_corpus"):
        loader(corpus, stable_mix={"nonexistent": 1.0})


# --- metrics -----------------------------------------------------------------------

def test_metrics_logger_writes_one_json_object_per_line(tmp_path):
    import json
    log = MetricsLogger(tmp_path / "metrics.jsonl", sync_every=1)
    log.log(step=0, loss=3.0)
    log.log(step=1, loss=2.5)
    log.close()
    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1]


def test_loss_ema_is_smoother_than_the_signal(tmp_path):
    """Seeded at the first value, so it lags a swinging signal rather than tracking it."""
    log = MetricsLogger(tmp_path / "m.jsonl", ema_beta=0.9)
    values = [10.0, 0.0, 10.0, 0.0]
    for v in values:
        last = log.update_ema(v)
    log.close()
    assert last == pytest.approx(8.19, abs=1e-2)          # never reaches either extreme
    assert min(values) < last < max(values)


def test_moe_health_reports_router_state():
    m = NanoSpeaker(tiny_cfg())
    m(torch.randint(0, 512, (2, 32)))
    h = moe_health(m)
    assert {"expert_load_cv", "experts_unused", "router_entropy"} <= set(h)
    assert h["expert_load_cv"] >= 0 and 0 <= h["router_entropy"] <= 1


def test_update_norm_ratio_is_zero_without_a_step():
    m = NanoSpeaker(tiny_cfg())
    names = muon_parameter_names(m)
    assert update_norm_ratio(m, snapshot(m, names)) == pytest.approx(0.0)


def test_update_norm_ratio_grows_after_a_step():
    m = NanoSpeaker(tiny_cfg())
    optims, _ = build_optimizers(m)
    names = muon_parameter_names(m)
    before = snapshot(m, names)
    loss, aux = m(torch.randint(0, 512, (2, 32)), torch.randint(0, 512, (2, 32)))
    (loss + aux).backward()
    optims["muon"].step()
    assert update_norm_ratio(m, before) > 0
