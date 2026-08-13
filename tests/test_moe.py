"""Tests for the MoE block, its router, and the GLU body they share."""

import pytest
import torch

from src.model.glu import GLUFeedForward, glu_hidden_dim
from src.model.moe import MoEBlock
from src.model.router import TopKRouter
from src.model.utils import ACTIVATIONS

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make(**kw):
    """Small block; fused=False keeps the tests off the inductor compile path."""
    return MoEBlock(**{"d_model": 32, "n_experts": 8, "d_ff": 64, "fused": False, **kw})


# --- routing correctness -----------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_sparse_dispatch_matches_dense_reference(device, top_k, bias):
    """The sorted per-expert dispatch must equal running every expert on every token."""
    m = make(top_k=top_k, bias=bias).to(device).double()
    x = torch.randn(3, 17, 32, device=device, dtype=torch.double)
    assert torch.allclose(m(x), m.dense_forward(x), atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_every_token_reaches_top_k_experts(device):
    """No capacity limit, so the assignment counts must total tokens * top_k exactly."""
    m = make(top_k=3).to(device)
    m(torch.randn(4, 25, 32, device=device))
    assert int(m.expert_counts.sum()) == 4 * 25 * 3


@pytest.mark.parametrize("device", DEVICES)
def test_second_expert_actually_contributes(device):
    """Guards against a dispatch bug where only the top-1 expert's output survives."""
    m = make(top_k=2, normalize_weights=False).to(device).double()
    x = torch.randn(6, 32, device=device, dtype=torch.double)
    top2 = m(x)

    # Same weights, same routing, but only the first expert kept.
    m.router.top_k, m.top_k = 1, 1
    assert not torch.allclose(m(x), top2)


def test_router_weights_sum_to_one_when_normalized():
    r = TopKRouter(32, 8, top_k=3, normalize_weights=True)
    w, _, probs = r(torch.randn(20, 32))
    assert torch.allclose(w.sum(-1), torch.ones(20), atol=1e-6)
    assert torch.allclose(probs.sum(-1), torch.ones(20), atol=1e-6)   # full softmax


def test_unnormalized_router_weights_are_raw_probabilities():
    r = TopKRouter(32, 8, top_k=3, normalize_weights=False)
    w, idx, probs = r(torch.randn(20, 32))
    assert torch.allclose(w, probs.gather(1, idx), atol=1e-6)


def test_balance_loss_penalizes_a_collapsed_router():
    """
    The loss is f . P, so it punishes load and confidence agreeing on one expert.
    Uniform load with uniform probs is the minimum; both piled on expert 0 is far worse.
    """
    r = TopKRouter(32, 4, top_k=1)
    uniform = r.balance_loss(torch.full((16, 4), 0.25), torch.tensor([4, 4, 4, 4]))
    collapsed = r.balance_loss(
        torch.tensor([[0.7, 0.1, 0.1, 0.1]]).expand(16, 4), torch.tensor([16, 0, 0, 0])
    )
    assert collapsed > uniform


def test_balance_loss_is_blind_to_load_when_the_router_is_undecided():
    """Corner of the same formula worth pinning: uniform P makes every split score alike."""
    r = TopKRouter(32, 4, top_k=1)
    probs = torch.full((16, 4), 0.25)
    assert r.balance_loss(probs, torch.tensor([4, 4, 4, 4])) == pytest.approx(
        float(r.balance_loss(probs, torch.tensor([16, 0, 0, 0])))
    )


def test_router_noise_explores_in_training_and_is_off_at_eval():
    """Noise must widen which experts get picked, and never perturb evaluation."""
    r = TopKRouter(32, 8, top_k=1, noise_std=5.0)
    x = torch.randn(64, 32)

    r.train()
    picks = {tuple(r(x)[1].flatten().tolist()) for _ in range(5)}
    assert len(picks) > 1, "noise did not change routing between training steps"

    r.eval()
    assert torch.equal(r(x)[1], r(x)[1])


def test_router_is_deterministic_without_noise():
    r = TopKRouter(32, 8, top_k=2).train()
    x = torch.randn(16, 32)
    assert torch.equal(r(x)[0], r(x)[0])


def test_noise_spreads_load_across_more_experts():
    """The reason to switch it on: a collapsed router still reaches every expert."""
    torch.manual_seed(0)
    x = torch.randn(256, 32)

    quiet, noisy = make(top_k=1, noise_std=0.0).train(), make(top_k=1, noise_std=3.0).train()
    noisy.router.W.data = quiet.router.W.data          # same (arbitrarily skewed) router
    quiet(x), noisy(x)
    assert int((noisy.expert_counts > 0).sum()) >= int((quiet.expert_counts > 0).sum())


def test_aux_loss_coef_zero_disables_the_loss():
    m = make(aux_loss_coef=0.0)
    m(torch.randn(4, 32))
    assert float(m.aux_loss) == 0.0


# --- gradients ---------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_gradients_reach_every_expert_and_the_router(device):
    m = make().to(device)
    x = torch.randn(4, 64, 32, device=device, requires_grad=True)
    (m(x).square().mean() + m.aux_loss).backward()

    assert x.grad.abs().sum() > 0
    assert m.router.W.grad.abs().sum() > 0                       # via the aux loss + weights
    per_expert = m.W_in.grad.flatten(1).abs().sum(1)
    assert (per_expert > 0).all(), f"experts without gradient: {(per_expert == 0).nonzero()}"


def test_backward_on_an_empty_batch_does_not_break_the_graph():
    m = make()
    x = torch.zeros(0, 32, requires_grad=True)
    m(x).sum().backward()                                        # must not raise


# --- shapes and dtypes -------------------------------------------------------------

@pytest.mark.parametrize("shape", [(32,), (5, 32), (2, 7, 32), (2, 3, 5, 32)])
def test_leading_dims_are_preserved(shape):
    assert make()(torch.randn(*shape)).shape == shape


def test_empty_input_returns_empty():
    assert make()(torch.zeros(0, 32)).shape == (0, 32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_dtype_is_preserved_and_output_is_finite(dtype):
    m = make().to(dtype)
    y = m(torch.randn(2, 8, 32, dtype=dtype))
    assert y.dtype == dtype and torch.isfinite(y).all()


# --- configuration and activations -------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"top_k": 9},            # more experts per token than exist
    {"top_k": 0},
    {"n_experts": 0},
    {"d_model": 0},
    {"d_ff": 0},
    {"d_ff": 1},             # 2/3 * d_ff rounds the GLU hidden width down to zero
])
def test_impossible_configs_raise(kw):
    with pytest.raises(ValueError):
        make(**kw)


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_every_named_activation_runs(name):
    m = make(activation=name)
    assert torch.isfinite(m(torch.randn(2, 5, 32))).all()


@pytest.mark.parametrize("bad", ["swiglu", "SiLU ", "", None, 7])
def test_unknown_activation_raises_instead_of_defaulting(bad):
    with pytest.raises(ValueError):
        make(activation=bad)


def test_activation_accepts_a_callable_and_is_case_insensitive():
    assert make(activation="GELU").activation == "gelu"

    # Same seed -> same weights, so a hand-written SiLU must reproduce the named one.
    torch.manual_seed(1)
    named = make(activation="silu")
    torch.manual_seed(1)
    custom = make(activation=lambda t: t * torch.sigmoid(t))
    x = torch.randn(4, 32)
    assert torch.allclose(named(x), custom(x), atol=1e-6)


# --- the shared GLU body -----------------------------------------------------------

@pytest.mark.parametrize("d_model,d_ff", [(512, 2048), (256, 1024), (768, 3072)])
def test_glu_costs_the_same_as_the_ffn_it_replaces(d_model, d_ff):
    """The reason the width is pinned at 2/3: three matrices must cost what two did."""
    glu_params = 3 * d_model * glu_hidden_dim(d_ff)
    ffn_params = 2 * d_model * d_ff
    assert glu_params == pytest.approx(ffn_params, rel=1e-3)   # rel: integer truncation


def test_moe_expert_matches_the_dense_glu_parameter_count():
    m = make(n_experts=4, d_ff=2048)
    per_expert = (m.W_in.numel() + m.W_out.numel()) // m.n_experts
    assert per_expert == pytest.approx(2 * m.d_model * m.d_ff, rel=1e-3)


def test_moe_with_one_expert_equals_the_dense_glu_block():
    """A 1-expert top-1 MoE is just a GLU FFN, so weights transplant one-for-one."""
    m = make(n_experts=1, top_k=1, normalize_weights=False).double()
    g = GLUFeedForward(32, d_ff=64, fused=False).double()
    g.W_in.data, g.W_out.data = m.W_in[0].data, m.W_out[0].data

    x = torch.randn(6, 32, dtype=torch.double)
    # top-1 of one expert routes everything to it with weight = softmax over 1 = 1.
    assert torch.allclose(m(x), g(x), atol=1e-12)


# --- shared experts ----------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_shared", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_shared_experts_match_the_dense_reference(device, n_shared, bias):
    """The always-on path must be identical whichever way the routed experts are run."""
    m = make(n_shared=n_shared, bias=bias).to(device).double()
    x = torch.randn(3, 17, 32, device=device, dtype=torch.double)
    assert torch.allclose(m(x), m.dense_forward(x), atol=1e-12)


def test_shared_expert_is_a_plain_glu_over_every_token():
    """With the routed experts silenced, what is left must be exactly one GLU."""
    m = make(n_shared=1).double()
    with torch.no_grad():
        m.W_out.zero_()                                   # routed contribution -> 0
    x = torch.randn(6, 32, dtype=torch.double)

    gate, up = (x @ m.W_sh_in[0]).chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate) * up) @ m.W_sh_out[0]
    assert torch.allclose(m(x), expected, atol=1e-12)


def test_shared_expert_changes_the_output():
    torch.manual_seed(0)
    plain = make(top_k=1)
    torch.manual_seed(0)
    shared = make(top_k=1, n_shared=1)                    # same routed weights, same routing

    x = torch.randn(4, 32)
    assert torch.equal(shared.W_in, plain.W_in)
    assert not torch.allclose(shared(x), plain(x))


def test_shared_expert_costs_extra_parameters_on_top_of_d_ff():
    """Shared capacity is added to the block, not carved out of the routed experts."""
    plain, shared = make(), make(n_shared=2)
    expert = 3 * shared.d_model * shared.d_hidden
    assert shared.W_in.shape == plain.W_in.shape          # routed experts unchanged
    assert sum(p.numel() for p in shared.parameters()) == \
        sum(p.numel() for p in plain.parameters()) + 2 * expert


def test_every_token_reaches_the_shared_expert():
    """Even a token routed to one expert must produce gradient on the shared weights."""
    m = make(top_k=1, n_shared=1)
    x = torch.randn(1, 32, requires_grad=True)
    m(x).square().sum().backward()

    assert m.W_sh_in.grad.abs().sum() > 0
    assert m.W_sh_out.grad.abs().sum() > 0
    assert int((m.W_in.grad.flatten(1).abs().sum(1) > 0).sum()) == 1   # one routed expert


def test_no_shared_expert_by_default():
    m = make()
    assert m.n_shared == 0 and m.W_sh_in is None and m.W_sh_out is None


def test_negative_n_shared_raises():
    with pytest.raises(ValueError):
        make(n_shared=-1)


# --- expert capacity ---------------------------------------------------------------

def collapse(m):
    """A zero router sends every token to one expert, so capacity has to bite."""
    with torch.no_grad():
        m.router.W.zero_()          # all logits equal -> top-k takes the lowest indices
    return m


def test_capacity_is_the_fair_share_times_the_factor():
    m = make(top_k=2)               # 8 experts
    assert m.capacity_factor == 1.25
    assert m._capacity(64) == 20    # ceil(1.25 * 64 * 2 / 8)
    assert m._capacity(1) == 1      # never rounds down to nothing


def test_default_is_a_capacity_of_1_25():
    assert MoEBlock(32, 8, d_ff=64, fused=False).capacity_factor == 1.25


def test_overflowing_tokens_are_dropped():
    m = collapse(make(top_k=1))     # every token -> one expert, capacity 1.25 * 64/8 = 10
    x = torch.randn(64, 32)
    out = m(x)

    assert int(m.dropped) == 64 - 10
    assert (out[:10].abs().sum(1) > 0).all()          # the ones that fit went through
    assert torch.allclose(out[10:], torch.zeros(54, 32))   # the rest skipped the FFN


def test_drops_fall_on_the_later_tokens():
    """Stable ordering, so the survivors are the earliest tokens of the run."""
    m = collapse(make(top_k=1))
    out = m(torch.randn(20, 32))                      # capacity 1.25 * 20/8 -> 4
    kept = (out.abs().sum(1) > 0).nonzero().flatten()
    assert kept.tolist() == [0, 1, 2, 3]


def test_dense_reference_drops_the_same_tokens():
    """Otherwise dense_forward stops being a reference for the padded dispatch."""
    m = collapse(make(top_k=1).double())
    x = torch.randn(48, 32, dtype=torch.double)
    sparse = m(x)
    assert int(m.dropped) > 0, "the premise of this test is that capacity bit"
    assert torch.allclose(sparse, m.dense_forward(x), atol=1e-12)


def test_no_capacity_means_no_drops():
    m = collapse(make(top_k=1, capacity_factor=None))
    out = m(torch.randn(64, 32))
    assert (out.abs().sum(1) > 0).all()               # every token still reached expert 0


def test_a_generous_factor_drops_nothing():
    m = collapse(make(top_k=1, capacity_factor=8.0))
    m(torch.randn(64, 32))
    assert int(m.dropped) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_balanced_routing_loses_almost_nothing(device):
    """The usual case: an untrained router is near-uniform, so 1.25 rarely bites."""
    m = make(n_experts=8, top_k=2).to(device)
    m(torch.randn(512, 32, device=device))
    assert int(m.dropped) / (512 * 2) < 0.05


def test_dropped_count_is_refreshed_each_forward():
    m = collapse(make(top_k=1))
    m(torch.randn(64, 32))
    assert int(m.dropped) == 54
    m(torch.randn(16, 32))                            # capacity 1.25 * 16/8 -> 3
    assert int(m.dropped) == 13


def test_capacity_still_reports_routing_counts_not_kept_counts():
    """expert_counts drives the balance loss, which must see the router's decisions."""
    m = collapse(make(top_k=1))
    m(torch.randn(64, 32))
    assert int(m.expert_counts.max()) == 64           # not the 10 that survived


def test_padded_and_ragged_paths_agree_when_nothing_overflows():
    # A factor far above any imbalance, so the two differ only in how they dispatch.
    torch.manual_seed(0)
    padded = make(top_k=2, capacity_factor=100.0).double()
    torch.manual_seed(0)
    ragged = make(top_k=2, capacity_factor=None).double()

    x = torch.randn(4, 16, 32, dtype=torch.double)
    out = padded(x)
    assert int(padded.dropped) == 0 and float(out.abs().sum()) > 0
    assert torch.allclose(out, ragged(x), atol=1e-12)


def test_capacity_survives_shared_experts_and_bias():
    m = collapse(make(top_k=1, n_shared=1, bias=True).double())
    x = torch.randn(48, 32, dtype=torch.double)
    assert torch.allclose(m(x), m.dense_forward(x), atol=1e-12)


def test_dropped_tokens_still_reach_the_shared_expert():
    """A shared expert is not routed, so capacity must not gate it."""
    m = collapse(make(top_k=1, n_shared=1))
    out = m(torch.randn(64, 32))
    assert (out[10:].abs().sum(1) > 0).all()          # dropped by the router, not skipped


def test_bad_capacity_factor_raises():
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="capacity_factor"):
            make(capacity_factor=bad)


# --- router short-circuits ----------------------------------------------------------

def test_probs_are_skipped_when_nothing_consumes_them():
    """The full [N, E] softmax exists only for the aux loss; it should not be built otherwise."""
    r = TopKRouter(32, 8, top_k=2).train()
    assert r(torch.randn(4, 32))[2] is not None                 # training + aux loss on

    assert r.eval()(torch.randn(4, 32))[2] is None              # eval
    assert TopKRouter(32, 8, top_k=2, aux_loss_coef=0.0).train()(
        torch.randn(4, 32))[2] is None                          # aux loss off


@pytest.mark.parametrize("normalize", [True, False])
def test_short_circuit_gives_the_same_weights_as_the_full_softmax(normalize):
    """topk-then-softmax must equal softmax-then-topk-then-renormalize, exactly."""
    torch.manual_seed(0)
    r = TopKRouter(32, 8, top_k=3, normalize_weights=normalize).double()
    x = torch.randn(64, 32, dtype=torch.double)

    r.train()
    w_full, idx_full, probs = r(x)
    assert probs is not None
    r.eval()
    w_short, idx_short, none = r(x)

    assert none is None
    assert torch.equal(idx_full, idx_short)
    # The router deliberately computes in fp32 (see TopKRouter.forward), so the two
    # spellings agree to fp32 epsilon regardless of the module's own dtype.
    assert torch.allclose(w_full, w_short, atol=1e-6)


def test_aux_loss_is_zero_when_probs_are_skipped():
    m = make(aux_loss_coef=0.0)
    m(torch.randn(8, 32))
    assert float(m.aux_loss) == 0.0

    m = make().eval()
    m(torch.randn(8, 32))
    assert float(m.aux_loss) == 0.0


def test_eval_and_training_route_identically_without_noise():
    """The short-circuit must not change which experts win, only how they're computed."""
    torch.manual_seed(0)
    m = make(top_k=2).double()
    x = torch.randn(32, 32, dtype=torch.double)
    m.train()
    trained = m(x)
    m.eval()
    assert torch.allclose(trained, m(x), atol=1e-6)     # fp32 router, fp64 block


def test_dense_forward_is_differentiable_and_reports_the_same_state():
    """It is a fallback path, not only an oracle, so it must behave like forward."""
    m = make(top_k=2)
    x = torch.randn(4, 16, 32, requires_grad=True)

    out = m.dense_forward(x)
    (out.square().mean() + m.aux_loss).backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert m.W_in.grad.abs().sum() > 0
    assert int(m.expert_counts.sum()) == 4 * 16 * 2
    assert float(m.aux_loss) > 0


def test_both_paths_report_the_same_counts_and_loss():
    m = make(top_k=2).double()
    x = torch.randn(3, 17, 32, dtype=torch.double)

    m(x)
    sparse_counts, sparse_loss = m.expert_counts.clone(), float(m.aux_loss)
    m.dense_forward(x)
    assert torch.equal(m.expert_counts, sparse_counts)
    assert float(m.aux_loss) == pytest.approx(sparse_loss)


# --- device-adaptive capacity -------------------------------------------------------

def test_auto_capacity_pads_where_the_kernel_cannot_run():
    """CPU has no fused path, so the eager dispatch still wants static shapes."""
    m = make()
    assert m._capacity_setting == "auto"
    assert m.capacity_factor == 1.25 and not m._kernel_ready()


def test_explicit_capacity_overrides_auto():
    assert make(capacity_factor=None).capacity_factor is None
    assert make(capacity_factor=4.0).capacity_factor == 4.0


def test_auto_capacity_follows_the_module_across_devices(monkeypatch):
    """.to(cuda) is what flips it, so it cannot be decided in __init__."""
    m = make()
    assert m.capacity_factor == 1.25

    # Stand in for a live kernel, so this pins the policy on any machine.
    monkeypatch.setattr(type(m), "_kernel_ready", lambda self: True)
    assert m.capacity_factor is None                      # kernel runs -> no cap needed

    monkeypatch.setattr(type(m), "_kernel_ready", lambda self: False)
    assert m.capacity_factor == 1.25                      # fallback -> pad again


def test_auto_capacity_respects_the_kernel_opt_out():
    m = make(kernel=False)
    assert not m._kernel_ready() and m.capacity_factor == 1.25


def test_fp64_falls_back_to_padding_even_on_cuda():
    """The kernel is fp16/bf16/fp32 only; fp64 must not be left on the ragged loop."""
    m = make().double()
    assert not m._kernel_ready() and m.capacity_factor == 1.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_auto_capacity_on_cuda_matches_the_kernel_gate():
    m = make().cuda()
    assert (m.capacity_factor is None) == m._kernel_ready()


def test_bad_capacity_factor_still_raises():
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="capacity_factor"):
            make(capacity_factor=bad)
