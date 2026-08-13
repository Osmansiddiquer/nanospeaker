#        :\     /;               _
#       ;  \___/  ;             ; ;
#      ,:-"'   `"-:.            / ;
# _   /,---.   ,---.\   _     _; /
# _:>((  |  ) (  |  ))<:_ ,-""_,"
#     \`````   `````/""""",-""
#      '-.._ v _..-'      )
#        / ___   ____,..  \
#       / /   | |   | ( \. \
#      / /    | |    | |  \ \
#      `"     `"     `"    `"
# MoE

import math

import torch
from torch import nn

from .glu import glu_hidden_dim, make_glu
from .router import TopKRouter
from .utils import ACTIVATIONS, Activation, resolve_activation

# The fused Triton path is optional: CPU-only installs carry no triton, and the eager
# dispatch below stays as both the fallback and the reference the kernels are tested
# against (see moe_kernel.py and tests/test_moe_kernel.py).
try:
    from .moe_kernel import KERNEL_ACTIVATIONS, fused_moe_forward, kernel_supported
    _HAS_KERNEL = True
except ImportError:
    _HAS_KERNEL = False

    def kernel_supported(t: torch.Tensor) -> bool:
        """No triton, no kernel: "auto" capacity then always resolves to padding."""
        return False


# What "auto" falls back to when the fused kernel is unavailable and the eager dispatch
# has to have static shapes to stay cheap.
AUTO_CAPACITY = 1.25


class MoEBlock(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward block with GLU experts (SwiGLU by default).

    Each expert is the GLU body of `glu.GLUFeedForward`, and each token is sent to its
    top_k experts by `router.TopKRouter`; the outputs are summed, weighted by the router
    probabilities.

    `capacity_factor` caps how many tokens an expert will take, at that multiple of its
    fair share (tokens * top_k / n_experts). It is what makes the dispatch static: with a
    cap, every expert gets an equal, host-known number of rows, so the block runs as one
    batched GEMM with no device->host sync and shapes stable enough for CUDA graphs;
    without one (`capacity_factor=None`) the per-expert splits are ragged, which costs a
    sync per block and one small GEMM launch per expert. The price is that tokens
    overflowing a full expert are dropped -- they skip its FFN and continue on the
    residual -- and padding underfull experts wastes some compute. Drops fall on the
    later tokens of an overloaded run, and `self.dropped` counts them each forward.

    `n_shared` adds always-on experts (DeepSeekMoE, Dai et al. 2024) on top of the routed
    ones: every token passes through them, ungated, and their output is added to the
    routed sum. The routed experts are then free to specialise, because whatever every
    token needs no longer has to be duplicated into each of them. They are extra
    parameters and extra work per token -- the budget for them is on top of d_ff, not
    carved out of it -- so a shared expert is bought against top_k, not for free.

    Args:
        d_model: model (input/output) width.
        n_experts: number of routed experts.
        d_ff: the FFN width each expert replaces, which sets its parameter budget and
            hence the GLU hidden width (see `glu.glu_hidden_dim`). Defaults to 4 * d_model.
            Shared experts are the same width as routed ones.
        top_k: routed experts activated per token.
        n_shared: always-on experts applied to every token (0 disables).
        capacity_factor: multiple of an expert's fair share it will accept before
            dropping tokens, None for the ragged no-drop dispatch, or "auto" (the
            default) to pick per device -- None wherever the fused kernel runs, since it
            needs no cap, and 1.25 wherever the eager fallback does.
        activation: gate activation name or callable (see `utils.ACTIVATIONS`).
        bias: add biases to the expert projections.
        normalize_weights: renormalize the top-k router probabilities to sum to 1.
        aux_loss_coef: load-balancing loss weight (0 disables).
        noise_std: exploration noise on the router logits while training (0 disables);
            anneal it via `block.router.noise_std` (see `router.TopKRouter`).
        fused: compile the pointwise GLU (see `glu.make_glu`).
        kernel: on CUDA, run the ragged (capacity_factor=None) dispatch as fused
            Triton grouped GEMMs with no device->host sync (see `moe_kernel.py`);
            the eager loop below stays the fallback wherever the kernels don't apply.

    After each forward, `self.aux_loss` holds the weighted balancing loss to add to the
    training objective, `self.expert_counts` the per-expert token counts (routing
    decisions, before any capacity drop) and `self.dropped` the pairs capacity turned
    away.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff: "int | None" = None,
        top_k: int = 2,
        n_shared: int = 0,
        capacity_factor: "float | None | str" = "auto",
        activation: "str | Activation" = "silu",
        bias: bool = False,
        normalize_weights: bool = True,
        aux_loss_coef: float = 1e-2,
        noise_std: float = 0.0,
        fused: bool = True,
        kernel: bool = True,
        std: float = 0.02,
    ):
    
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.d_model, self.n_experts, self.d_ff = d_model, n_experts, d_ff
        self.d_hidden = glu_hidden_dim(d_ff)
        self.top_k = top_k
        if n_shared < 0:
            raise ValueError(f"n_shared ({n_shared}) cannot be negative.")
        self.n_shared = n_shared
        if capacity_factor == "auto":
            pass
        elif capacity_factor is not None and capacity_factor <= 0:
            raise ValueError(
                f"capacity_factor ({capacity_factor}) must be > 0, None, or \"auto\"."
            )
        self._capacity_setting = capacity_factor

        # The router validates d_model/n_experts/top_k and owns the balancing loss.
        self.router = TopKRouter(
            d_model, n_experts, top_k, normalize_weights, aux_loss_coef, noise_std, std
        )
        self.activation, act = resolve_activation(activation)
        self.glu = make_glu(act, fused)

        # The Triton epilogue implements the named activations only, and a name that
        # has been shadowed by a custom callable must not be routed to the wrong kernel,
        # so the kernel path is resolved once here rather than trusted at dispatch time.
        self.kernel = kernel
        self._kernel_act = (
            self.activation
            if _HAS_KERNEL
            and ACTIVATIONS.get(self.activation) is act
            and self.activation in KERNEL_ACTIVATIONS
            else None
        )

        # Experts are stacked along dim 0 so each one is a contiguous GEMM operand,
        # with gate and up projections fused into W_in -> one matmul per expert.
        self.W_in = nn.Parameter(torch.randn(n_experts, d_model, 2 * self.d_hidden) * std)
        self.W_out = nn.Parameter(torch.randn(n_experts, self.d_hidden, d_model) * std)

        # Biases are off by default: GLU FFNs generally drop them at no cost in quality.
        self.b_in = nn.Parameter(torch.zeros(n_experts, 2 * self.d_hidden)) if bias else None
        self.b_out = nn.Parameter(torch.zeros(n_experts, d_model)) if bias else None

        # Shared experts are stacked the same way, but never indexed by the router: they
        # are kept separate from W_in/W_out so the dispatch path stays untouched.
        self.W_sh_in = self.W_sh_out = self.b_sh_in = self.b_sh_out = None
        if n_shared:
            self.W_sh_in = nn.Parameter(
                torch.randn(n_shared, d_model, 2 * self.d_hidden) * std
            )
            self.W_sh_out = nn.Parameter(torch.randn(n_shared, self.d_hidden, d_model) * std)
            if bias:
                self.b_sh_in = nn.Parameter(torch.zeros(n_shared, 2 * self.d_hidden))
                self.b_sh_out = nn.Parameter(torch.zeros(n_shared, d_model))

        # Populated every forward for the training loop to read.
        self.aux_loss = torch.zeros(())
        self.expert_counts = torch.zeros(n_experts)
        self.dropped = torch.zeros((), dtype=torch.long)

        # Run boundaries are read against this every forward; as a buffer it follows the
        # module's device instead of being rebuilt (allocation + kernel) per layer per step.
        self.register_buffer(
            "edges", torch.arange(n_experts + 1, dtype=torch.long), persistent=False
        )

    def _kernel_ready(self) -> bool:
        """Whether the fused path would take this block's own weights, as they sit now."""
        return (
            self.kernel
            and self._kernel_act is not None
            and kernel_supported(self.W_in)
        )

    @property
    def capacity_factor(self):
        """
        Effective capacity, resolved against the device the module is currently on.

        A cap and the fused kernel are two answers to the same problem -- ragged expert
        counts giving data-dependent shapes -- and only one can apply. The kernel is the
        better one where it runs: it maps tiles to groups on device, so it needs no cap,
        drops no tokens and wastes nothing on padding. Where it does not run (CPU, fp64,
        a custom activation, triton missing) the eager fallback still wants static shapes,
        so "auto" reverts to padding.

        Resolved lazily rather than in __init__ because modules are built on CPU and
        moved afterwards; `.to("cuda")` is what flips this.
        """
        if self._capacity_setting != "auto":
            return self._capacity_setting
        return None if self._kernel_ready() else AUTO_CAPACITY

    def _plan(self, topk_idx: torch.Tensor):
        """
        Sort the N*top_k (token, expert) pairs by expert id and measure each run.

        Each expert's pairs then form one contiguous run, and order // top_k says which
        token each pair came from. The sort is stable so that a run is ordered by token
        position, which is what makes capacity drop the *later* tokens rather than an
        arbitrary subset, and makes both dispatch paths drop exactly the same ones.

        Run boundaries read straight off the sorted ids. searchsorted keeps this on
        device, where bincount would sync to size its own output.
        """
        experts, order = topk_idx.reshape(-1).sort(stable=True)
        token_idx = order.div(self.top_k, rounding_mode="floor")
        counts = torch.searchsorted(experts, self.edges).diff()
        return experts, order, token_idx, counts

    def _capacity(self, n_tokens: int) -> int:
        """Rows each expert is given: capacity_factor x its fair share, at least one."""
        fair_share = n_tokens * self.top_k / self.n_experts
        return max(1, min(math.ceil(self.capacity_factor * fair_share), n_tokens * self.top_k))

    def _slots(self, experts, counts, cap: int):
        """
        Where each sorted pair lands in the padded [n_experts, cap] buffer.

        Rank within a run is the pair's index minus where its run starts. Pairs ranked at
        or past cap are dropped, and are pointed at one extra trailing slot instead of
        being filtered out -- masking would make the shape data-dependent, which is the
        whole thing this path exists to avoid.
        """
        starts = counts.cumsum(0) - counts                       # [n_experts]
        rank = torch.arange(experts.numel(), device=experts.device) - starts[experts]
        keep = rank < cap
        return torch.where(keep, experts * cap + rank, self.n_experts * cap), keep

    def _forward_padded(self, x, experts, order, token_idx, counts, topk_w, shape):
        """
        Fixed-capacity dispatch: every shape here is known on the host.

        The ragged path splits by `counts.tolist()`, which is a device->host sync per
        block and leaves one small GEMM per expert to launch. Padding every expert to the
        same cap trades ~capacity_factor times the rows for a single batched matmul, no
        sync, and shapes stable enough for CUDA graphs -- at the cost of dropping the
        tokens that overflow, which pass through on the residual instead.
        """
        n = x.shape[0]
        cap = self._capacity(n)
        slot, keep = self._slots(experts, counts, cap)
        self.dropped = (~keep).sum().detach()

        # Gather each pair's token into its slot. The trailing row collects every dropped
        # pair; index_copy_ is undefined on duplicate indices, which is fine because that
        # row is never read back.
        xb = x.new_zeros(self.n_experts * cap + 1, self.d_model)
        xb.index_copy_(0, slot, x.index_select(0, token_idx))
        xb = xb[: self.n_experts * cap].view(self.n_experts, cap, self.d_model)

        # One batched GEMM per layer instead of one per expert.
        h = xb @ self.W_in
        if self.b_in is not None:
            h = h + self.b_in[:, None, :]
        gate, up = h.chunk(2, dim=-1)
        y = self.glu(gate, up) @ self.W_out
        if self.b_out is not None:
            y = y + self.b_out[:, None, :]

        # Zero row for the dropped pairs, so scattering back needs no mask: they add 0.
        y = torch.cat((y.reshape(-1, self.d_model), y.new_zeros(1, self.d_model)))
        w = topk_w.reshape(-1).index_select(0, order)

        out = torch.zeros_like(x)
        out.index_add_(0, token_idx, y.index_select(0, slot) * w[:, None])
        if self.n_shared:
            out = out + self._shared(x)
        return out.view(shape)

    def _expert(self, e: int, x: torch.Tensor, w: torch.Tensor, wts) -> torch.Tensor:
        """
        GLU expert `e` on its own [n, d_model] token block, scaled by router weights `w`.

        `wts` holds the stacked parameters already unbound into per-expert views.
        Indexing the stack here instead (`self.W_in[e]`) routes backward through
        `select`, whose gradient is a zero tensor the size of the *whole* stack per
        expert; `unbind` backward is one `stack` for the loop. Measured at
        n_experts=8, d_model=512, N=4096: 387 -> 352 MiB peak.
        """
        W_in, W_out, b_in, b_out = wts
        h = x @ W_in[e]
        if b_in is not None:
            h = h + b_in[e]

        # W_in packs gate and up side by side, so that single matmul fed both branches.
        gate, up = h.chunk(2, dim=-1)
        y = self.glu(gate, up) @ W_out[e]
        if b_out is not None:
            y = y + b_out[e]

        # Scale by the router weight here, on [n, d_model]: the same scale applied
        # before W_out would be algebraically identical but touch 2*d_hidden columns.
        return y * w[:, None]

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        """
        The always-on experts on the full [N, d_model] batch, summed.

        No routing, no gate: every token goes through every shared expert at weight 1.
        n_shared is 1 or 2 in practice, so this loops over dense 2D matmuls rather than
        paying for a batched one over a length-1 leading dim.
        """
        out = None
        for s in range(self.n_shared):
            h = x @ self.W_sh_in[s]
            if self.b_sh_in is not None:
                h = h + self.b_sh_in[s]

            gate, up = h.chunk(2, dim=-1)
            y = self.glu(gate, up) @ self.W_sh_out[s]
            if self.b_sh_out is not None:
                y = y + self.b_sh_out[s]
            out = y if out is None else out + y
        return out

    def _use_kernel(self, x: torch.Tensor) -> bool:
        """
        Whether this forward goes through the fused Triton dispatch: the ragged path,
        with an activation the epilogue implements, on tensors the kernel takes
        (`kernel_supported` -- the same predicate "auto" capacity resolves against, so
        the two can't disagree). The padded path keeps its own batched GEMM, and
        everything else falls back to the eager loop below -- which is also the
        reference the kernels are tested against.
        """
        return (
            self.kernel and self._kernel_act is not None
            and self.capacity_factor is None and kernel_supported(x)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {tuple(x.shape)}.")

        # Routing is per token, so the leading dims are flattened away and restored at the end.
        shape = x.shape
        x = x.reshape(-1, self.d_model)

        # No tokens: clone rather than zeros_like, so the result stays attached to the graph.
        if x.shape[0] == 0:
            return x.clone().view(shape)
        topk_w, topk_idx, probs = self.router(x)

        if self._use_kernel(x):
            # The whole ragged dispatch -- counting sort, per-expert GEMMs, weighted
            # scatter -- runs as fused Triton grouped GEMMs with no device->host sync
            # (moe_kernel.py). Its histogram hands back the routing counts, so _plan's
            # sort has nothing left to compute on this path.
            out, counts = fused_moe_forward(
                x, self.W_in, self.W_out, topk_w, topk_idx,
                self.b_in, self.b_out, self._kernel_act,
            )
            self.expert_counts = counts.detach()
            self.aux_loss = (
                self.router.balance_loss(probs, counts)
                if probs is not None
                else torch.zeros((), device=x.device, dtype=x.dtype)
            )
            if self.n_shared:
                out = out + self._shared(x)
            return out.view(shape)

        experts, order, token_idx, counts = self._plan(topk_idx)

        # Counts are the routing decisions, before any capacity drop: balance is a
        # property of the router, and penalising it for what capacity truncated would
        # blame it for its own overflow.
        self.expert_counts = counts.detach()

        # probs is None when nothing consumes it (eval, or the aux loss switched off).
        self.aux_loss = (
            self.router.balance_loss(probs, counts)
            if probs is not None
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )

        if self.capacity_factor is not None:
            return self._forward_padded(x, experts, order, token_idx, counts, topk_w, shape)

        # Gather each pair's token and router weight once, then cut both into per-expert
        # runs. sizes is the block's single device->host sync, and it buys dense GEMMs.
        sizes = counts.tolist()
        xs = x.index_select(0, token_idx).split(sizes)
        ws = topk_w.reshape(-1).index_select(0, order).split(sizes)
        idx = token_idx.split(sizes)

        wts = (
            self.W_in.unbind(0), self.W_out.unbind(0),
            self.b_in.unbind(0) if self.b_in is not None else None,
            self.b_out.unbind(0) if self.b_out is not None else None,
        )

        out = torch.zeros_like(x)
        for e, (x_e, i_e, w_e) in enumerate(zip(xs, idx, ws)):
            # Idle experts are skipped: an empty GEMM still costs kernel launches.
            if x_e.shape[0] == 0:
                continue
            # Scatter-add, so a token's top_k expert outputs land on the same row and sum.
            out.index_add_(0, i_e, self._expert(e, x_e, w_e, wts))

        # Shared experts run on the whole batch, so they sit outside the dispatch loop.
        if self.n_shared:
            out = out + self._shared(x)
        return out.view(shape)

    def dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference path: every expert on every token, masked by the gates. Costs
        O(n_experts) FLOPs instead of O(top_k), but has no data-dependent shapes and
        no host sync. Used to validate `forward`, and viable for tiny inputs where
        the dispatch loop dominates -- which is why it is differentiable and keeps
        `aux_loss`/`expert_counts` current, so it can stand in for `forward` rather
        than only check it. Wrap it in `torch.no_grad()` yourself for validation.

        Note `h = x @ self.W_in` is [n_experts, N, 2*d_hidden]: this path trades a
        data-dependent peak for a predictable but much larger one, and at big N it will
        run out of memory well before the dispatch loop does.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {tuple(x.shape)}.")
        shape = x.shape
        x = x.reshape(-1, self.d_model)
        topk_w, topk_idx, probs = self.router(x)
        experts, order, _, counts = self._plan(topk_idx)

        # Same side effects as `forward`, so a training loop can swap the two paths
        # without noticing. The sort this needs is negligible next to running every
        # expert on every token.
        self.expert_counts = counts.detach()
        self.aux_loss = (
            self.router.balance_loss(probs, counts)
            if probs is not None
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )

        # Capacity has to be enforced here too, or this stops being a reference for
        # `forward`. Zeroing an overflowing pair's weight is the same thing as dropping
        # it, since the gate is what admits an expert's output.
        if self.capacity_factor is not None and x.shape[0]:
            _, keep = self._slots(experts, counts, self._capacity(x.shape[0]))
            unsorted = torch.empty_like(keep).index_copy_(0, order, keep)
            self.dropped = (~keep).sum().detach()
            topk_w = topk_w * unsorted.view_as(topk_w)

        # Dense [N, E] gate matrix: the router weight where an expert was picked, 0 elsewhere.
        gates = torch.zeros(x.shape[0], self.n_experts, dtype=x.dtype, device=x.device)
        gates.scatter_(1, topk_idx, topk_w)

        # x broadcasts against the stacked expert weights -> [E, N, ...] for both layers.
        h = x @ self.W_in
        h = h if self.b_in is None else h + self.b_in[:, None, :]
        gate, up = h.chunk(2, dim=-1)
        y = self.glu(gate, up) @ self.W_out
        y = y if self.b_out is None else y + self.b_out[:, None, :]

        # Mix each token's expert outputs by its gate weights -> [N, d_model].
        out = torch.einsum("enf,ne->nf", y, gates)
        if self.n_shared:
            out = out + self._shared(x)
        return out.view(shape)
