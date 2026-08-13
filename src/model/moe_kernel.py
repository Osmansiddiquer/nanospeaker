#   _______________________________
#  |  ___________________________  |
#  | | grouped GEMM, no host sync| |
#  | |___________________________| |
#  |_______________________________|
#     \\ tile -> expert, on device
#
# Fused Triton dispatch for MoEBlock's ragged (no-capacity) path.

"""
Triton grouped-GEMM MoE dispatch (review.md section 6, kernels A-D + backward).

The eager ragged path in `moe.py` pays three implementation costs that this module
deletes, all downstream of one design choice -- the host never learns the per-expert
token counts:

  * `counts.tolist()` was a device->host sync per MoE layer (and a guaranteed
    `torch.compile` graph break). Here every shape is a host-side worst case and the
    *actual* padded length lives in a 1-element device tensor that kernels read to
    early-exit. Nothing waits on the GPU.
  * One GEMM launch per expert becomes one launch per matmul, with each tile finding
    its expert on device from the run offsets (kernel A).
  * The `[tokens, 2*d_hidden]` GLU intermediate never round-trips to HBM: the first
    GEMM applies act(gate) * up in its epilogue (kernel B), and the second folds the
    router-weight scale and the scatter-add back to token rows into its epilogue as
    fp32 atomics (kernel C). On a 128-bit-bus card those two fusions are the point.

Dispatch (kernel D) is a counting sort: histogram the expert ids, block-align each
expert's run to ALIGN_M rows (vLLM's `moe_align_block_size` trick -- a GEMM tile then
never straddles two experts), and scatter each (token, expert) pair to its slot with
an atomic cursor. Order within a run is arbitrary, which is fine: outputs are summed,
and the fp32 atomic order was never deterministic -- exactly like the eager path's
CUDA `index_add_`.

Backward (section 6.6) reuses the same grouped structure: dH is a grouped GEMM against
W_out read transposed (an index swap), dX one against a pre-transposed copy of W_in
(its 5460 B rows sit 16B-misaligned when read the other way -- the copy measured
cheaper), dW is a per-expert `X^T @ dY` over that expert's run, and the
pre-activations are *recomputed* rather
than saved -- on a 4 GB card a transient buffer per layer beats one resident per layer
for the whole graph. Owning dW also removes the `select_backward` full-stack zero-fill
(finding 2.1) by construction. Both ops register fake kernels and autograd through
`torch.library.custom_op`, so `torch.compile` schedules around them with no break
(section 6.7).

Ampere notes: tiles are budgeted for sm_86's 100 KB shared memory, masked `tl.load`
lowers to cp.async so `num_stages` still pipelines, and the grid is data-parallel with
device-side early exit rather than persistent -- the persistent/TMA design from the
H100 writeups has nothing to lean on here, and 20 SMs are saturated by tile count
alone. Weight/output GEMMs autotune over a few configs; the atomic-accumulating
kernels use a fixed config, because an autotuner re-running them would double-add.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# The dispatch aligns each expert's run to ALIGN_M rows so a GEMM tile never
# straddles two experts. Kernels may still tile rows at any divisor of it (the
# owner of a smaller block is the owner of its enclosing aligned block), which lets
# the weight-heavy first GEMM take 128-row tiles -- halving how often each expert's
# weights are re-read from HBM, measured 3.5 -> 2.7 ms -- while the second GEMM
# keeps the 64-row tiles that measured faster for it. Not autotuned: retuning the
# alignment would mean re-running the dispatch.
ALIGN_M = 128

# Gate activations the epilogue can apply (id -> constexpr branch in _act_fwd/_act_bwd).
# A custom callable can't cross into a kernel, so moe.py falls back to eager for those.
KERNEL_ACTIVATIONS = {"silu": 0, "swish": 0, "gelu": 1, "relu": 2, "sigmoid": 3, "tanh": 4}

# tl.dot dtypes on this hardware; fp64 has no tensor-core path and stays eager.
KERNEL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def kernel_supported(t: torch.Tensor) -> bool:
    """
    Whether the fused path can take tensors like this one (device and dtype).

    The single source of truth for the hardware half of the gate: MoEBlock's dispatch
    check and its "auto" capacity resolution both call this, so they cannot drift
    apart -- a block the kernel would decline must resolve to a padded cap, never to
    the eager ragged loop, which is the slowest of the three paths.
    """
    return t.is_cuda and t.dtype in KERNEL_DTYPES


def _dot_precision(dtype: torch.dtype) -> str:
    """fp32 tl.dot defaults to tf32 on Ampere; honor torch's stricter default instead."""
    if dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return "ieee"
    return "tf32"


def _l_cap(m: int, n_experts: int) -> int:
    """
    Host-side worst case for the padded pair count, rounded to whole blocks.

    Block-aligning each run adds at most ALIGN_M - 1 pad slots per *non-empty* expert,
    and at most min(E, M) experts can be non-empty. Buffers are allocated at this cap;
    the true padded length is only ever known on device.
    """
    return triton.cdiv(m + min(n_experts, m) * (ALIGN_M - 1), ALIGN_M) * ALIGN_M


# --- kernel D: on-device dispatch (histogram, scan, tile->expert map, scatter) ------

@triton.jit
def _count_kernel(topk_ptr, counts_ptr, M, BLOCK: tl.constexpr):
    """Histogram the flat expert ids with atomics: counts[e] = #pairs routed to e."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    e = tl.load(topk_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + e, 1, mask=mask)


@triton.jit
def _scan_kernel(counts_ptr, off_ptr, npp_ptr, E, BM: tl.constexpr, E_POW2: tl.constexpr):
    """
    One program: block-align each count and exclusive-scan into run offsets.

    n_experts is small, so a single vectorized cumsum beats launching anything wider.
    Also writes the total padded length -- the device-side scalar every later kernel
    checks to early-exit, in place of the host ever knowing the counts.
    """
    offs = tl.arange(0, E_POW2)
    c = tl.load(counts_ptr + offs, mask=offs < E, other=0)
    padded = tl.cdiv(c, BM) * BM                      # empty experts take zero blocks
    cum = tl.cumsum(padded, 0)
    tl.store(off_ptr + 1 + offs, cum.to(tl.int32), mask=offs < E)
    zero = tl.zeros((1,), tl.int32)
    tl.store(off_ptr + tl.arange(0, 1), zero)
    total = tl.sum(tl.where(offs < E, padded, 0))
    tl.store(npp_ptr + tl.arange(0, 1), total.to(tl.int32) + zero)


@triton.jit
def _expert_ids_kernel(off_ptr, npp_ptr, eid_ptr, E, BM: tl.constexpr, E_POW2: tl.constexpr):
    """
    The tile->group map (the trick from review.md 6.1): expert_ids[b] = owner of block b.

    The owner is the last expert whose run starts at or before the block -- zero-width
    runs share a start and the comparison count lands on the live one. E is tiny, so a
    vectorized compare-and-sum is the whole "binary search".
    """
    b = tl.program_id(0)
    if b * BM >= tl.load(npp_ptr):
        return                                        # past the data: stays 0 from init
    offs = tl.arange(0, E_POW2)
    starts = tl.load(off_ptr + offs, mask=offs < E, other=2147483647)
    e = tl.sum((starts <= b * BM).to(tl.int32), 0) - 1
    tl.store(eid_ptr + b, e)


@triton.jit
def _scatter_kernel(topk_ptr, off_ptr, cur_ptr, sorted_ptr, M, BLOCK: tl.constexpr):
    """
    Counting-sort scatter: pair i lands at its run's offset plus an atomic cursor rank.

    Ranks are dense in [0, counts[e]), so each run's real pairs form a prefix and the
    block-alignment padding stays at the tail, already holding the sentinel from init.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    e = tl.load(topk_ptr + offs, mask=mask, other=0)
    base = tl.load(off_ptr + e, mask=mask, other=0)
    rank = tl.atomic_add(cur_ptr + e, 1, mask=mask)
    tl.store(sorted_ptr + base + rank, offs.to(tl.int32), mask=mask)


@torch.library.custom_op("autoreg::moe_dispatch", mutates_args=())
def moe_dispatch(topk_idx: torch.Tensor, n_experts: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Sort the (token, expert) pairs into block-aligned per-expert runs, on device.

    Returns int32 tensors: sorted pair ids [l_cap] (flat index into topk_idx, sentinel
    M in pad slots), aligned-block->expert map [l_cap / ALIGN_M], run offsets [E + 1],
    routing counts [E] (for the balance loss), and the true padded length [1].
    """
    flat = topk_idx.reshape(-1).contiguous()
    m = flat.numel()
    dev, i32 = topk_idx.device, torch.int32
    l_cap = _l_cap(m, n_experts)
    counts = torch.zeros(n_experts, dtype=i32, device=dev)
    cursor = torch.zeros(n_experts, dtype=i32, device=dev)
    eids = torch.zeros(l_cap // ALIGN_M, dtype=i32, device=dev)   # 0 past the data: safe
    offs = torch.zeros(n_experts + 1, dtype=i32, device=dev)
    npp = torch.zeros(1, dtype=i32, device=dev)
    sids = torch.full((l_cap,), m, dtype=i32, device=dev)         # all-sentinel start
    if m:
        ep2 = triton.next_power_of_2(n_experts)
        _count_kernel[(triton.cdiv(m, 512),)](flat, counts, m, BLOCK=512)
        _scan_kernel[(1,)](counts, offs, npp, n_experts, BM=ALIGN_M, E_POW2=ep2)
        _expert_ids_kernel[(l_cap // ALIGN_M,)](offs, npp, eids, n_experts, BM=ALIGN_M, E_POW2=ep2)
        _scatter_kernel[(triton.cdiv(m, 512),)](flat, offs, cursor, sids, m, BLOCK=512)
    return sids, eids, offs, counts, npp


@moe_dispatch.register_fake
def _(topk_idx, n_experts):
    # Shapes are pure functions of (numel, E), which is what makes this traceable.
    l_cap = _l_cap(topk_idx.numel(), n_experts)
    mk = lambda n: topk_idx.new_empty((n,), dtype=torch.int32)
    return mk(l_cap), mk(l_cap // ALIGN_M), mk(n_experts + 1), mk(n_experts), mk(1)


# --- the gate activation, in-kernel (fp32; dead branches are pruned per ACT) --------

@triton.jit
def _act_fwd(g, ACT: tl.constexpr):
    if ACT == 0:                                       # silu
        return g * tl.sigmoid(g)
    if ACT == 1:                                       # gelu (erf form, like F.gelu)
        return 0.5 * g * (1 + tl.math.erf(g * 0.7071067811865476))
    if ACT == 2:                                       # relu
        return tl.maximum(g, 0.0)
    if ACT == 3:                                       # sigmoid
        return tl.sigmoid(g)
    return 2 * tl.sigmoid(2 * g) - 1                   # tanh, via sigmoid


@triton.jit
def _act_bwd(g, ACT: tl.constexpr):
    if ACT == 0:
        s = tl.sigmoid(g)
        return s * (1 + g * (1 - s))
    if ACT == 1:                                       # Phi(g) + g * phi(g)
        return 0.5 * (1 + tl.math.erf(g * 0.7071067811865476)) \
            + g * 0.3989422804014327 * tl.exp(-0.5 * g * g)
    if ACT == 2:
        return tl.where(g > 0, 1.0, 0.0)               # grad 0 at 0, matching torch
    if ACT == 3:
        s = tl.sigmoid(g)
        return s * (1 - s)
    t = 2 * tl.sigmoid(2 * g) - 1
    return 1 - t * t


# --- kernels A+B: grouped GEMM with gather-A and the GLU fused into the epilogue ----

def _gemm1_configs():
    # Config pool measured on sm_86 (see review.md 6.1 for the starting point): the
    # 128-row tiles halve per-expert weight re-reads and win at large token counts,
    # the 64-row ones win when runs are short. Key is (D, H), so whichever wins at
    # the first call's token count sticks for that model shape. The dual accumulator
    # costs 2 * BM * BN fp32 registers, which is what keeps BN at 64.
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=_gemm1_configs(), key=["D", "H"])
@triton.jit
def _gemm1_kernel(
    x_ptr, w_ptr, b_ptr, sorted_ptr, eid_ptr, npp_ptr, h_ptr, a1_ptr,
    M, TOPK, D, H,
    ACT: tl.constexpr, PREC: tl.constexpr, HAS_BIAS: tl.constexpr,
    WRITE_H: tl.constexpr, WRITE_A1: tl.constexpr, ALIGN_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    H[slot] = act(x[tok] @ W_in[e] gate half) * (x[tok] @ W_in[e] up half).

    One A load feeds two accumulators against the two W_in halves, so the [*, 2H]
    pre-activation never exists in HBM on the forward pass. The backward recompute
    instead asks for exactly that tensor (WRITE_A1) and skips the activation.
    """
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(npp_ptr):            # past the real data: whole tile idle
        return
    # A sub-block inherits its enclosing aligned block's expert (BLOCK_M | ALIGN_M).
    e = tl.load(eid_ptr + pid_m * BLOCK_M // ALIGN_M).to(tl.int64)
    slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ptr + slots)                 # l_cap is block-aligned: in bounds
    valid = pair < M                                   # sentinel rows load as zero
    if tl.max(valid.to(tl.int32), 0) == 0:
        return                                         # alignment tail: all sentinel
    tok = tl.where(valid, pair // TOPK, 0).to(tl.int64)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ncols = rn < H
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, D, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kcols = rk < D
        a = tl.load(x_ptr + tok[:, None] * D + rk[None, :],
                    mask=valid[:, None] & kcols[None, :], other=0.0)
        bmask = kcols[:, None] & ncols[None, :]
        wp = w_ptr + e * D * 2 * H + rk[:, None] * 2 * H + rn[None, :]
        acc_g = tl.dot(a, tl.load(wp, mask=bmask, other=0.0), acc_g, input_precision=PREC)
        acc_u = tl.dot(a, tl.load(wp + H, mask=bmask, other=0.0), acc_u, input_precision=PREC)

    if HAS_BIAS:
        acc_g += tl.load(b_ptr + e * 2 * H + rn, mask=ncols, other=0.0).to(tl.float32)[None, :]
        acc_u += tl.load(b_ptr + e * 2 * H + H + rn, mask=ncols, other=0.0).to(tl.float32)[None, :]
    omask = valid[:, None] & ncols[None, :]
    srow = slots.to(tl.int64)
    if WRITE_A1:                                       # backward recompute wants raw g, u
        ty = a1_ptr.dtype.element_ty
        tl.store(a1_ptr + srow[:, None] * 2 * H + rn[None, :], acc_g.to(ty), mask=omask)
        tl.store(a1_ptr + srow[:, None] * 2 * H + H + rn[None, :], acc_u.to(ty), mask=omask)
    if WRITE_H:
        hv = _act_fwd(acc_g, ACT) * acc_u
        tl.store(h_ptr + srow[:, None] * H + rn[None, :],
                 hv.to(h_ptr.dtype.element_ty), mask=omask)


# --- kernel C (and its backward twins): grouped GEMM core with epilogue variants ----

# Fixed configs rather than autotune: in ATOMIC mode the kernel *accumulates* into
# its output, so an autotuner re-running configs would corrupt it. Chosen by an
# offline sweep on sm_86 at the review's reference shape (d_model=512, d_hidden=1365,
# 8k pairs, bf16); per-mode because the winners genuinely differ.
_K2_FWD = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=2)
_K2_G = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)
_K2_DX = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=3)


@triton.jit
def _gemm2_kernel(
    a_ptr, b_ptr, bias_ptr, tw_ptr, sorted_ptr, eid_ptr, npp_ptr, out_ptr,
    M, TOPK, K_DIM, N_DIM,
    PREC: tl.constexpr, GATHER_A: tl.constexpr, B_TRANS: tl.constexpr,
    APPLY_W: tl.constexpr, HAS_BIAS: tl.constexpr, ATOMIC: tl.constexpr,
    ALIGN_M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    One grouped-GEMM body, three uses (B_TRANS reads the [N_DIM, K_DIM] stack as its
    transpose -- an index swap, not a copy):

      forward out:  A = H by slot,  B = W_out[e],   *router weight, atomic-add by token
      backward dX:  A = dA1 by slot, B = W_in[e]^T (pre-transposed), atomic by token
      backward dH:  A = dOut by token (gather),  B = W_out[e]^T,  plain store by slot

    The transpose is spelled as two literal-index branches rather than stride
    arguments so the compiler can *see* which axis is contiguous: runtime strides
    de-vectorize the B loads, which measured as ~3x on this whole kernel.

    The atomic epilogue is kernel C: the per-expert index_add_ launches and their
    staging buffer collapse into this store. Accumulation is fp32 (the buffer, not
    just the math), so bf16 contributions don't round away (review.md 6.3).
    """
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(npp_ptr):
        return
    e = tl.load(eid_ptr + pid_m * BLOCK_M // ALIGN_M).to(tl.int64)
    slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ptr + slots)
    valid = pair < M
    if tl.max(valid.to(tl.int32), 0) == 0:
        return                                         # alignment tail: all sentinel
    tok = tl.where(valid, pair // TOPK, 0).to(tl.int64)
    arow = tok if GATHER_A else slots.to(tl.int64)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ncols = rn < N_DIM
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K_DIM, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kcols = rk < K_DIM
        a = tl.load(a_ptr + arow[:, None] * K_DIM + rk[None, :],
                    mask=valid[:, None] & kcols[None, :], other=0.0)
        if B_TRANS:                                    # physical [N_DIM, K_DIM] stack
            b = tl.load(b_ptr + e * K_DIM * N_DIM + rn[None, :] * K_DIM + rk[:, None],
                        mask=kcols[:, None] & ncols[None, :], other=0.0)
        else:                                          # physical [K_DIM, N_DIM] stack
            b = tl.load(b_ptr + e * K_DIM * N_DIM + rk[:, None] * N_DIM + rn[None, :],
                        mask=kcols[:, None] & ncols[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PREC)

    if HAS_BIAS:                                       # per-expert bias, before the scale
        acc += tl.load(bias_ptr + e * N_DIM + rn, mask=ncols, other=0.0).to(tl.float32)[None, :]
    if APPLY_W:                                        # router weight: free on registers
        w = tl.load(tw_ptr + pair, mask=valid, other=0.0).to(tl.float32)
        acc *= w[:, None]
    omask = valid[:, None] & ncols[None, :]
    if ATOMIC:                                         # top_k rows sum onto one token row
        tl.atomic_add(out_ptr + tok[:, None] * N_DIM + rn[None, :], acc, mask=omask)
    else:
        tl.store(out_ptr + slots.to(tl.int64)[:, None] * N_DIM + rn[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=omask)


# --- backward: per-expert dW = A^T @ B over that expert's run -----------------------

def _dw_configs():
    # Sweep winners on sm_86 at the reference shape: the wide-B config took dW_out
    # (its output rows are d_model-wide), the square one took dW_in.
    return [
        triton.Config({"BLOCK_A": 64, "BLOCK_B": 64, "BLOCK_R": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_A": 32, "BLOCK_B": 128, "BLOCK_R": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_A": 32, "BLOCK_B": 64, "BLOCK_R": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_A": 32, "BLOCK_B": 32, "BLOCK_R": 64}, num_warps=4, num_stages=4),
    ]


@triton.autotune(configs=_dw_configs(), key=["DA", "DB"])
@triton.jit
def _dw_kernel(
    a_ptr, b_ptr, tw_ptr, sorted_ptr, off_ptr, cnt_ptr, dw_ptr,
    M, TOPK, DA, DB,
    PREC: tl.constexpr, GATHER_A: tl.constexpr, GATHER_B: tl.constexpr, SCALE_B: tl.constexpr,
    BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """
    dW[e] = sum over expert e's rows of A_row^T (x) B_row, reduced in fp32 registers.

    Each program owns one (expert, tile) and walks the run's *real* rows -- the
    counting sort left them as a dense prefix, so the loop bound is just counts[e].
    Writing dW directly is what deletes the select_backward stack blow-up (2.1): no
    autograd ever differentiates through W[e] indexing. Empty experts store zero,
    which is also their correct gradient. No split-K: E x tile-count programs already
    saturate 20 SMs, and atomics here would cost determinism for nothing.

      dW_in:  A = x (gather by token), B = dA1 (by slot)
      dW_out: A = H (by slot),         B = dOut (gather by token, scaled by w)
    """
    e = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    na = tl.cdiv(DA, BLOCK_A)
    ra = (pid % na) * BLOCK_A + tl.arange(0, BLOCK_A)
    rb = (pid // na) * BLOCK_B + tl.arange(0, BLOCK_B)
    base = tl.load(off_ptr + e)
    cnt = tl.load(cnt_ptr + e)

    acc = tl.zeros((BLOCK_A, BLOCK_B), tl.float32)
    for r0 in range(0, cnt, BLOCK_R):
        rows = (base + r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        rmask = r0 + tl.arange(0, BLOCK_R) < cnt
        if (GATHER_A or GATHER_B) or SCALE_B:
            pair = tl.load(sorted_ptr + rows, mask=rmask, other=0)
            tok = (pair // TOPK).to(tl.int64)
        if GATHER_A:
            a = tl.load(a_ptr + tok[:, None] * DA + ra[None, :],
                        mask=rmask[:, None] & (ra[None, :] < DA), other=0.0)
        else:
            a = tl.load(a_ptr + rows[:, None] * DA + ra[None, :],
                        mask=rmask[:, None] & (ra[None, :] < DA), other=0.0)
        if GATHER_B:
            b = tl.load(b_ptr + tok[:, None] * DB + rb[None, :],
                        mask=rmask[:, None] & (rb[None, :] < DB), other=0.0)
        else:
            b = tl.load(b_ptr + rows[:, None] * DB + rb[None, :],
                        mask=rmask[:, None] & (rb[None, :] < DB), other=0.0)
        if SCALE_B:                                    # fold dY = w * dOut into the load
            w = tl.load(tw_ptr + pair, mask=rmask, other=0.0)
            b = b * w[:, None]
        acc = tl.dot(tl.trans(a), b, acc, input_precision=PREC)

    dwp = dw_ptr + e * DA * DB + ra.to(tl.int64)[:, None] * DB + rb[None, :]
    tl.store(dwp, acc.to(dw_ptr.dtype.element_ty), mask=(ra[:, None] < DA) & (rb[None, :] < DB))


@triton.jit
def _glu_bwd_kernel(
    a1_ptr, g_ptr, tw_ptr, sorted_ptr, da1_ptr, h_ptr, dtw_ptr,
    M, H,
    ACT: tl.constexpr, ZERO_INVALID: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """
    Row-wise GLU backward. Per real slot, given G = dOut @ W_out^T (unscaled):

      hidden = act(g) * u          (recomputed; also stored for the dW_out GEMM)
      dw     = <G, hidden>         (router-weight grad -- avoids recomputing y = H @ W_out)
      d_gate = w * G * u * act'(g),  d_up = w * G * act(g)

    Sentinel rows optionally zero-fill dA1 so the bias grad's segment-sum can consume
    the buffer whole. One program per row; H is walked in chunks with the dw partial
    kept as a lane-wise vector and reduced once at the end.
    """
    slot = tl.program_id(0).to(tl.int64)
    pair = tl.load(sorted_ptr + slot)
    if pair >= M:
        if ZERO_INVALID:
            zero = tl.zeros((BLOCK_H,), da1_ptr.dtype.element_ty)
            for h0 in range(0, 2 * H, BLOCK_H):
                rh = h0 + tl.arange(0, BLOCK_H)
                tl.store(da1_ptr + slot * 2 * H + rh, zero, mask=rh < 2 * H)
        return

    w = tl.load(tw_ptr + pair).to(tl.float32)
    part = tl.zeros((BLOCK_H,), tl.float32)
    for h0 in range(0, H, BLOCK_H):
        rh = h0 + tl.arange(0, BLOCK_H)
        m = rh < H
        g = tl.load(a1_ptr + slot * 2 * H + rh, mask=m, other=0.0).to(tl.float32)
        u = tl.load(a1_ptr + slot * 2 * H + H + rh, mask=m, other=0.0).to(tl.float32)
        gr = tl.load(g_ptr + slot * H + rh, mask=m, other=0.0).to(tl.float32)
        act = _act_fwd(g, ACT)
        hidden = act * u
        part += gr * hidden                            # masked lanes contribute zero
        dh = w * gr
        ty = da1_ptr.dtype.element_ty
        tl.store(da1_ptr + slot * 2 * H + rh, (dh * u * _act_bwd(g, ACT)).to(ty), mask=m)
        tl.store(da1_ptr + slot * 2 * H + H + rh, (dh * act).to(ty), mask=m)
        tl.store(h_ptr + slot * H + rh, hidden.to(h_ptr.dtype.element_ty), mask=m)
    tl.store(dtw_ptr + pair, tl.sum(part, 0).to(dtw_ptr.dtype.element_ty))


# --- the differentiable op ----------------------------------------------------------

@torch.library.custom_op("autoreg::moe_experts", mutates_args=())
def moe_experts(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    topk_w: torch.Tensor,
    b_in: Optional[torch.Tensor],
    b_out: Optional[torch.Tensor],
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    pad_offsets: torch.Tensor,
    counts: torch.Tensor,
    n_post_pad: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """
    The routed experts on pre-dispatched pairs: out[t] = sum_e w * expert_e(x[t]).

    Two launches. Inputs must share x's dtype (the wrapper casts); output matches it.
    """
    x, w_in, w_out = x.contiguous(), w_in.contiguous(), w_out.contiguous()
    n, d = x.shape
    h = w_in.shape[2] // 2
    topk = topk_w.shape[-1]
    m = n * topk
    l_cap = sorted_ids.numel()
    act, prec = KERNEL_ACTIVATIONS[activation], _dot_precision(x.dtype)
    tw = topk_w.reshape(-1).contiguous()

    hid = x.new_empty((l_cap, h))
    if l_cap:
        grid = lambda meta: (triton.cdiv(l_cap, meta["BLOCK_M"]), triton.cdiv(h, meta["BLOCK_N"]))
        _gemm1_kernel[grid](
            x, w_in, b_in if b_in is not None else w_in, sorted_ids, expert_ids,
            n_post_pad, hid, hid,                      # a1_ptr unused: any tensor works
            m, topk, d, h, ACT=act, PREC=prec, HAS_BIAS=b_in is not None,
            WRITE_H=True, WRITE_A1=False, ALIGN_M=ALIGN_M,
        )
    out32 = torch.zeros((n, d), dtype=torch.float32, device=x.device)
    if l_cap:
        _gemm2_kernel[(l_cap // _K2_FWD["BLOCK_M"], triton.cdiv(d, _K2_FWD["BLOCK_N"]))](
            hid, w_out, b_out if b_out is not None else w_out, tw, sorted_ids,
            expert_ids, n_post_pad, out32,
            m, topk, h, d,                             # B = W_out[e], untransposed
            PREC=prec, GATHER_A=False, B_TRANS=False, APPLY_W=True,
            HAS_BIAS=b_out is not None, ATOMIC=True, ALIGN_M=ALIGN_M, **_K2_FWD,
        )
    return out32.to(x.dtype)


@moe_experts.register_fake
def _(x, w_in, w_out, topk_w, b_in, b_out, sorted_ids, expert_ids, pad_offsets,
      counts, n_post_pad, activation):
    return x.new_empty(x.shape)


@torch.library.custom_op("autoreg::moe_experts_bwd", mutates_args=())
def _moe_experts_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    topk_w: torch.Tensor,
    b_in: Optional[torch.Tensor],
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    pad_offsets: torch.Tensor,
    counts: torch.Tensor,
    n_post_pad: torch.Tensor,
    activation: str,
    need_db_in: bool,
    need_db_out: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (dx, dW_in, dW_out, dtopk_w, db_in, db_out) for `moe_experts`.

    Its own custom op, not a plain function: the autograd hook below must stay
    traceable, and raw kernel launches can't run under fake tensors -- as one opaque
    op the whole backward compiles the same way the forward does. Unneeded bias grads
    come back as empty tensors (an op can't return None conditionally).
    """
    grad_out = grad_out.to(x.dtype).contiguous()
    x, w_in, w_out = x.contiguous(), w_in.contiguous(), w_out.contiguous()
    n, d = x.shape
    e_n = w_in.shape[0]
    h = w_in.shape[2] // 2
    topk = topk_w.shape[-1]
    m = n * topk
    l_cap = sorted_ids.numel()
    act, prec = KERNEL_ACTIVATIONS[activation], _dot_precision(x.dtype)
    tw = topk_w.reshape(-1).contiguous()

    # Recompute the pre-activations instead of saving them from forward: [l_cap, 2H]
    # alive for one backward step, not for every layer across the whole graph.
    a1 = x.new_empty((l_cap, 2 * h))
    grid1 = lambda meta: (triton.cdiv(l_cap, meta["BLOCK_M"]), triton.cdiv(h, meta["BLOCK_N"]))
    _gemm1_kernel[grid1](
        x, w_in, b_in if b_in is not None else w_in, sorted_ids, expert_ids,
        n_post_pad, a1, a1,                            # h_ptr unused (WRITE_H=False)
        m, topk, d, h, ACT=act, PREC=prec, HAS_BIAS=b_in is not None,
        WRITE_H=False, WRITE_A1=True, ALIGN_M=ALIGN_M,
    )

    # G = dOut @ W_out^T per expert, *unscaled* -- so the same tensor serves both the
    # GLU backward (scaled by w there) and the router-weight grad <G, hidden>.
    gbuf = x.new_empty((l_cap, h))
    _gemm2_kernel[(l_cap // _K2_G["BLOCK_M"], triton.cdiv(h, _K2_G["BLOCK_N"]))](
        grad_out, w_out, w_out, tw, sorted_ids, expert_ids, n_post_pad, gbuf,
        m, topk, d, h,                                 # B = W_out[e]^T (B_TRANS)
        PREC=prec, GATHER_A=True, B_TRANS=True, APPLY_W=False, HAS_BIAS=False,
        ATOMIC=False, ALIGN_M=ALIGN_M, **_K2_G,
    )

    # GLU backward per row: dA1, the router-weight grads, and H for the dW_out GEMM.
    da1 = x.new_empty((l_cap, 2 * h))
    hbuf = x.new_empty((l_cap, h))
    dtw = torch.empty_like(tw)
    _glu_bwd_kernel[(l_cap,)](
        a1, gbuf, tw, sorted_ids, da1, hbuf, dtw, m, h,
        ACT=act, ZERO_INVALID=need_db_in, BLOCK_H=128,
    )

    # dX: the gather's backward is a scatter -- same atomic epilogue as the forward.
    # W_in^T is materialized rather than index-swapped: W_in rows are 2H*2 = 5460 B,
    # and reading them transposed means every load sits at a 16B-misaligned offset.
    # The 22 MB copy measured ~0.5 ms against ~2.6 ms saved on this GEMM.
    w_in_t = w_in.transpose(1, 2).contiguous()
    dx32 = torch.zeros((n, d), dtype=torch.float32, device=x.device)
    _gemm2_kernel[(l_cap // _K2_DX["BLOCK_M"], triton.cdiv(d, _K2_DX["BLOCK_N"]))](
        da1, w_in_t, w_in_t, tw, sorted_ids, expert_ids, n_post_pad, dx32,
        m, topk, 2 * h, d,                             # B = W_in[e]^T, pre-transposed
        PREC=prec, GATHER_A=False, B_TRANS=False, APPLY_W=False, HAS_BIAS=False,
        ATOMIC=True, ALIGN_M=ALIGN_M, **_K2_DX,
    )

    dw_in = torch.empty_like(w_in)
    grid_a = lambda meta: (e_n, triton.cdiv(d, meta["BLOCK_A"]) * triton.cdiv(2 * h, meta["BLOCK_B"]))
    _dw_kernel[grid_a](x, da1, tw, sorted_ids, pad_offsets, counts, dw_in,
                       m, topk, d, 2 * h,
                       PREC=prec, GATHER_A=True, GATHER_B=False, SCALE_B=False)
    dw_out = torch.empty_like(w_out)
    grid_b = lambda meta: (e_n, triton.cdiv(h, meta["BLOCK_A"]) * triton.cdiv(d, meta["BLOCK_B"]))
    _dw_kernel[grid_b](hbuf, grad_out, tw, sorted_ids, pad_offsets, counts, dw_out,
                       m, topk, h, d,
                       PREC=prec, GATHER_A=False, GATHER_B=True, SCALE_B=True)

    # Bias grads are segment sums over each expert's rows. Bias is off by default, so
    # this stays plain (on-device, sync-free) torch rather than another kernel.
    db_in = x.new_empty(0)
    db_out = x.new_empty(0)
    if need_db_in or need_db_out:
        eps = expert_ids.repeat_interleave(ALIGN_M).long()     # expert of every slot
    if need_db_in:                                     # dA1 pad rows were zero-filled
        db_in = torch.zeros((e_n, 2 * h), dtype=x.dtype, device=x.device) \
            .index_add_(0, eps, da1)
    if need_db_out:
        pairs = sorted_ids.long().clamp(max=m - 1)
        wv = tw[pairs] * (sorted_ids < m).to(tw.dtype)         # sentinel rows -> weight 0
        dyw = grad_out[pairs // topk] * wv[:, None]
        db_out = torch.zeros((e_n, d), dtype=x.dtype, device=x.device) \
            .index_add_(0, eps, dyw)

    return dx32.to(x.dtype), dw_in, dw_out, dtw.view_as(topk_w), db_in, db_out


@_moe_experts_bwd.register_fake
def _(grad_out, x, w_in, w_out, topk_w, b_in, sorted_ids, expert_ids, pad_offsets,
      counts, n_post_pad, activation, need_db_in, need_db_out):
    e_n, _, h2 = w_in.shape
    return (torch.empty_like(x), torch.empty_like(w_in), torch.empty_like(w_out),
            torch.empty_like(topk_w),
            x.new_empty((e_n, h2)) if need_db_in else x.new_empty(0),
            x.new_empty((e_n, x.shape[1])) if need_db_out else x.new_empty(0))


def _moe_experts_setup(ctx, inputs, output):
    (x, w_in, w_out, topk_w, b_in, b_out, sorted_ids, expert_ids, pad_offsets,
     counts, n_post_pad, activation) = inputs
    # b_in is a real backward input (the recomputed pre-activation includes it);
    # b_out's value is never needed, only whether its gradient is.
    saved = [x, w_in, w_out, topk_w, sorted_ids, expert_ids, pad_offsets, counts, n_post_pad]
    if b_in is not None:
        saved.append(b_in)
    ctx.save_for_backward(*saved)
    ctx.has_b_in = b_in is not None
    ctx.has_b_out = b_out is not None
    ctx.activation = activation


def _moe_experts_backward(ctx, grad_out):
    """Thin traceable hook: unpack the ctx and hand everything to the backward op."""
    (x, w_in, w_out, topk_w, sorted_ids, expert_ids, pad_offsets, counts,
     n_post_pad, *rest) = ctx.saved_tensors
    b_in = rest[0] if ctx.has_b_in else None
    dx, dw_in, dw_out, dtw, db_in, db_out = _moe_experts_bwd(
        grad_out, x, w_in, w_out, topk_w, b_in, sorted_ids, expert_ids, pad_offsets,
        counts, n_post_pad, ctx.activation, ctx.has_b_in, ctx.has_b_out,
    )
    return (dx, dw_in, dw_out, dtw, db_in if ctx.has_b_in else None,
            db_out if ctx.has_b_out else None, None, None, None, None, None, None)


moe_experts.register_autograd(_moe_experts_backward, setup_context=_moe_experts_setup)


# --- entry point for MoEBlock -------------------------------------------------------

def fused_moe_forward(x, w_in, w_out, topk_w, topk_idx, b_in, b_out, activation):
    """
    Dispatch + routed experts, fused; returns (out [N, d_model], counts [n_experts]).

    counts are the routing decisions from the on-device histogram, ready for the
    balance loss -- `_plan` and its sort have nothing left to do on this path.
    """
    if activation not in KERNEL_ACTIVATIONS:
        raise ValueError(
            f"activation {activation!r} has no kernel; use one of "
            f"{sorted(set(KERNEL_ACTIVATIONS))} or fall back to the eager path."
        )
    sids, eids, offs, counts, npp = moe_dispatch(topk_idx, w_in.shape[0])
    # The GEMMs run in x's dtype, exactly as autocast would run the eager matmuls, so
    # params are cast here (a no-op off autocast; grads flow back through the cast).
    cast = lambda t: None if t is None else t.to(x.dtype)
    out = moe_experts(x, cast(w_in), cast(w_out), topk_w.to(x.dtype), cast(b_in),
                      cast(b_out), sids, eids, offs, counts, npp, activation)
    return out, counts
