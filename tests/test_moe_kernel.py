"""
Tests for the fused Triton MoE dispatch (moe_kernel.py).

Two references are used, per review.md 6.8: `dense_forward` (every expert on every
token, no data-dependent shapes) and the eager ragged loop (same math, per-expert
GEMMs) -- the kernels must agree with both, forward and backward, including on a
deliberately collapsed router where most experts are empty and one run is ragged.
"""

import copy

import pytest
import torch

# The whole file is CUDA-only: the kernels *are* the CUDA path, and the eager loop
# they are validated against is exercised on CPU throughout test_moe.py.
if not torch.cuda.is_available():
    pytest.skip("moe_kernel needs CUDA", allow_module_level=True)

from src.model.moe import MoEBlock
from src.model.moe_kernel import ALIGN_M, KERNEL_ACTIVATIONS, moe_dispatch, moe_experts


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def make(**kw):
    """Kernel-path block on CUDA; fused=False keeps the eager GLU off inductor."""
    m = MoEBlock(**{"d_model": 32, "n_experts": 8, "d_ff": 64, "fused": False,
                    "capacity_factor": None, **kw})
    return m.cuda()


def eager_twin(m):
    """Same weights, same routing, eager dispatch: the reference implementation."""
    twin = copy.deepcopy(m)
    twin.kernel = False
    return twin


def collapse(m):
    """A zero router sends every token to the first experts: empty runs + one ragged one."""
    with torch.no_grad():
        m.router.W.zero_()
    return m


def run_both(m, x, tol):
    """Forward + backward through the kernel block and its eager twin; compare all grads."""
    twin = eager_twin(m)
    xk = x.clone().requires_grad_(True)
    xe = x.clone().requires_grad_(True)
    ok, oe = m(xk), twin(xe)
    assert m._use_kernel(xk.reshape(-1, x.shape[-1])), "kernel path was not taken"
    assert torch.allclose(ok, oe, atol=tol), f"forward diff {(ok - oe).abs().max()}"

    (ok.square().mean() + m.aux_loss).backward()
    (oe.square().mean() + twin.aux_loss).backward()
    pairs = [("x", xk.grad, xe.grad), ("W_in", m.W_in.grad, twin.W_in.grad),
             ("W_out", m.W_out.grad, twin.W_out.grad),
             ("router", m.router.W.grad, twin.router.W.grad)]
    if m.b_in is not None:
        pairs += [("b_in", m.b_in.grad, twin.b_in.grad),
                  ("b_out", m.b_out.grad, twin.b_out.grad)]
    if m.n_shared:
        pairs += [("W_sh_in", m.W_sh_in.grad, twin.W_sh_in.grad)]
    for name, a, b in pairs:
        assert torch.allclose(a, b, atol=tol), \
            f"{name} grad diff {(a - b).abs().max()}"


# --- against the dense reference ---------------------------------------------------

@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_kernel_matches_dense_reference(top_k, bias):
    """The fused dispatch must equal running every expert on every token."""
    m = make(top_k=top_k, bias=bias)
    x = torch.randn(3, 17, 32, device="cuda")
    with torch.no_grad():
        assert torch.allclose(m(x), m.dense_forward(x), atol=1e-5)


@pytest.mark.parametrize("n_shared", [1, 2])
def test_shared_experts_ride_along(n_shared):
    """Shared experts run outside the kernel; the sum must still match the dense path."""
    m = make(n_shared=n_shared, bias=True)
    x = torch.randn(5, 32, device="cuda")
    with torch.no_grad():
        assert torch.allclose(m(x), m.dense_forward(x), atol=1e-5)


# --- against the eager ragged dispatch, forward and backward -----------------------

@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_kernel_matches_eager_dispatch_with_gradients(top_k, bias):
    m = make(top_k=top_k, bias=bias)
    run_both(m, torch.randn(3, 40, 32, device="cuda"), tol=1e-5)


def test_collapsed_router_exercises_the_ragged_tail():
    """
    All tokens on one expert: most runs are empty (zero blocks), the live one is long
    and ends mid-block -- the exact shapes uniform routing never produces.
    """
    m = collapse(make(top_k=1))
    run_both(m, torch.randn(200, 32, device="cuda"), tol=1e-5)


def test_shared_experts_and_bias_with_gradients():
    m = make(n_shared=1, bias=True)
    run_both(m, torch.randn(2, 33, 32, device="cuda"), tol=1e-5)


@pytest.mark.parametrize("name", sorted(set(KERNEL_ACTIVATIONS)))
def test_every_kernel_activation_matches_eager(name):
    """Forward and backward of each in-kernel gate activation against torch's own."""
    m = make(activation=name)
    run_both(m, torch.randn(2, 21, 32, device="cuda"), tol=1e-5)


@pytest.mark.parametrize("n,top_k", [(1, 1), (1, 2), (3, 8), (ALIGN_M - 1, 2),
                                     (ALIGN_M, 2), (ALIGN_M + 1, 2)])
def test_degenerate_and_block_boundary_shapes(n, top_k):
    """Single tokens, fewer pairs than a block, and exact block-multiple row counts."""
    m = make(top_k=top_k)
    run_both(m, torch.randn(n, 32, device="cuda"), tol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_matches_eager_in_kind(dtype):
    """Same dtype both sides, so only accumulation order differs (kernel sums in fp32)."""
    m = make().to(dtype)
    run_both(m, torch.randn(2, 65, 32, device="cuda", dtype=dtype), tol=2e-2)


# --- bookkeeping the training loop reads -------------------------------------------

def test_counts_and_aux_loss_match_the_eager_path():
    """The dispatch histogram must report the same routing the sort-based plan does."""
    m = make(top_k=3)
    twin = eager_twin(m)
    x = torch.randn(4, 25, 32, device="cuda")
    m(x), twin(x)
    assert torch.equal(m.expert_counts.long(), twin.expert_counts.long())
    assert int(m.expert_counts.sum()) == 4 * 25 * 3
    assert torch.allclose(m.aux_loss, twin.aux_loss, atol=1e-7)


def test_gradients_reach_every_expert_and_the_router():
    m = make()
    x = torch.randn(4, 64, 32, device="cuda", requires_grad=True)
    (m(x).square().mean() + m.aux_loss).backward()
    assert x.grad.abs().sum() > 0
    assert m.router.W.grad.abs().sum() > 0
    per_expert = m.W_in.grad.flatten(1).abs().sum(1)
    assert (per_expert > 0).all(), f"experts without gradient: {(per_expert == 0).nonzero()}"


# --- when the kernel must step aside -----------------------------------------------

def test_fallbacks_keep_the_eager_path_correct():
    """fp64, custom callables, CPU and kernel=False all mean the loop, not a wrong kernel."""
    x64 = torch.randn(3, 17, 32, device="cuda", dtype=torch.double)
    m = make().double()
    assert not m._use_kernel(x64.reshape(-1, 32))
    with torch.no_grad():
        assert torch.allclose(m(x64), m.dense_forward(x64), atol=1e-12)

    custom = make(activation=lambda t: t * torch.sigmoid(t))
    assert custom._kernel_act is None                  # unnamed callable: no kernel
    assert make(kernel=False)._use_kernel(torch.zeros(1, 32, device="cuda")) is False
    cpu = MoEBlock(32, 8, d_ff=64, fused=False, capacity_factor=None)
    assert not cpu._use_kernel(torch.zeros(1, 32))


def test_padded_path_is_untouched():
    """capacity_factor set means the batched-GEMM path, kernel flag or not."""
    # Explicit, not defaulted: the default is now "auto", which resolves to None on a
    # CUDA module precisely so the kernel takes it.
    m = MoEBlock(32, 8, d_ff=64, fused=False, capacity_factor=1.25).cuda()
    assert not m._use_kernel(torch.zeros(1, 32, device="cuda"))


def test_auto_default_hands_cuda_blocks_to_the_kernel():
    """The point of the adaptive default: no cap, no drops, no padding where it runs."""
    m = MoEBlock(32, 8, d_ff=64, fused=False).cuda()
    assert m.capacity_factor is None
    assert m._use_kernel(torch.zeros(1, 32, device="cuda"))


# --- the ops themselves (review.md 6.7: fake kernels, autograd, compile) -----------

def test_dispatch_invariants():
    """White-box check of the counting sort: dense prefixes, sentinels, block owners."""
    topk_idx = torch.randint(0, 8, (50, 2), device="cuda")
    sids, eids, offs, counts, npp, inv = moe_dispatch(topk_idx, 8)
    m = topk_idx.numel()
    assert torch.equal(sids[inv.long()],                # inv really inverts the sort
                       torch.arange(m, device="cuda", dtype=torch.int32))
    assert torch.equal(counts.long(), torch.bincount(topk_idx.reshape(-1), minlength=8))
    assert int(npp) == int((torch.ceil(counts.float() / ALIGN_M) * ALIGN_M).sum())

    flat = topk_idx.reshape(-1)
    for e in range(8):
        run = sids[int(offs[e]): int(offs[e]) + int(counts[e])].long()
        assert (flat[run] == e).all()                  # every slot holds a pair of its expert
        pad = sids[int(offs[e]) + int(counts[e]): int(offs[e + 1])]
        assert (pad == m).all()                        # alignment tail is all sentinel
    blocks = torch.arange(int(npp) // ALIGN_M, device="cuda") * ALIGN_M
    assert (offs[eids[: len(blocks)].long()] <= blocks).all()
    assert torch.equal(torch.sort(sids[sids < m]).values,
                       torch.arange(m, device="cuda", dtype=torch.int32))


def test_dispatch_handles_no_pairs():
    sids, eids, offs, counts, npp, inv = moe_dispatch(torch.zeros(0, 2, dtype=torch.long, device="cuda"), 8)
    assert sids.numel() == 0 and int(npp) == 0 and int(counts.sum()) == 0 and inv.numel() == 0


def test_opcheck_both_ops():
    """torch.library's own audit: schema, fake tensors, autograd registration, AOT."""
    topk_idx = torch.randint(0, 4, (10, 2), device="cuda")
    torch.library.opcheck(moe_dispatch, (topk_idx, 4))

    m = make(n_experts=4)
    x = torch.randn(10, 32, device="cuda")
    with torch.no_grad():
        topk_w, topk_idx, _ = m.router(x)
    sids, eids, offs, counts, npp, inv = moe_dispatch(topk_idx, 4)
    torch.library.opcheck(moe_experts, (
        x.clone().requires_grad_(True), m.W_in.detach().clone().requires_grad_(True),
        m.W_out.detach().clone().requires_grad_(True),
        topk_w.clone().requires_grad_(True), None, None,
        sids, eids, offs, counts, npp, inv, "silu",
    ))


def test_block_compiles_fullgraph_through_forward_and_backward():
    """
    The point of the custom-op wrapping: the block that used to graph-break on
    `counts.tolist()` now compiles whole, and the compiled grads match eager.
    """
    m = make(n_experts=4)
    cm = torch.compile(m, fullgraph=True)
    x = torch.randn(2, 33, 32, device="cuda", requires_grad=True)
    out = cm(x)
    (out.square().mean() + m.aux_loss).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with torch.no_grad():
        assert torch.allclose(out, m(x.detach()), atol=1e-5)
