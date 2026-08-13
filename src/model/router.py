"""Token-choice top-k routing for mixture-of-experts layers."""

import torch
import torch.nn.functional as F
from torch import nn


class TopKRouter(nn.Module):
    """
    Softmax router that sends each token to its top_k experts.

    Token-choice with no capacity limit: every token reaches exactly top_k experts and
    none is ever dropped, so balance has to come from `balance_loss` rather than from
    truncation.

    Args:
        d_model: model width.
        n_experts: number of experts to route over.
        top_k: experts activated per token.
        normalize_weights: renormalize the kept probabilities to sum to 1.
        aux_loss_coef: weight of the load-balancing loss (0 disables it).
        noise_std: stddev of exploration noise added to the logits while training
            (0 disables it). Anneal it by assigning to `router.noise_std`.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        normalize_weights: bool = True,
        aux_loss_coef: float = 1e-2,
        noise_std: float = 0.0,
        std: float = 0.02,
    ):
        super().__init__()
        # Reject impossible configs here rather than failing deep inside a GEMM.
        if min(d_model, n_experts, top_k) < 1:
            raise ValueError(
                f"d_model ({d_model}), n_experts ({n_experts}) and top_k ({top_k}) "
                f"must all be >= 1."
            )
        if top_k > n_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed n_experts ({n_experts}).")

        self.n_experts, self.top_k = n_experts, top_k
        self.normalize_weights = normalize_weights
        self.aux_loss_coef = aux_loss_coef
        self.noise_std = noise_std
        self.W = nn.Parameter(torch.randn(d_model, n_experts) * std)

    def wants_probs(self) -> bool:
        """Whether anything will consume the full [N, E] distribution this step."""
        return self.training and self.aux_loss_coef != 0.0

    def forward(self, x: torch.Tensor):
        """
        [N, d_model] -> top-k weights [N, k], expert ids [N, k], and full probs [N, E].

        `probs` is None whenever nothing needs it (eval, or the aux loss switched off):
        it exists only for `balance_loss`, and returning it unconditionally keeps an
        fp32 [N, E] softmax pinned in the autograd graph of every layer for the whole
        forward.
        """
        # The matmul runs in x's dtype; only the softmax and the top-k are lifted to
        # fp32, which is where low-precision noise would actually change which experts
        # win. Casting the operands instead would make the matmul fp32 too, at real cost
        # under autocast -- that is deliberately not what this does.
        logits = (x @ self.W).float()

        # Exploration noise (Shazeer et al. 2017): perturbing the logits lets experts
        # just outside the top-k occasionally win, so early training does not lock onto
        # whichever experts the init happened to favour. Training only, and worth
        # annealing to 0 once the load has spread out - later work (ST-MoE) found
        # leaving it on costs quality.
        if self.training and self.noise_std:
            logits = logits + torch.randn_like(logits) * self.noise_std

        if self.wants_probs():
            probs = F.softmax(logits, dim=-1)
            topk_w, topk_idx = probs.topk(self.top_k, dim=-1)
        else:
            # A softmax restricted to a subset equals the full softmax renormalized over
            # it, so with normalize_weights the full [N, E] softmax is redundant: take
            # the top-k logits and soften those. Identical arithmetic, E -> k wide.
            probs = None
            topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
            topk_w = (
                F.softmax(topk_logits, dim=-1)
                if self.normalize_weights
                else F.softmax(logits, dim=-1).gather(1, topk_idx)
            )

        # Renormalize so the surviving experts' weights sum to 1 (Mixtral-style);
        # without it a layer's output shrinks by whatever mass top-k discarded.
        if self.normalize_weights and self.top_k > 1 and probs is not None:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        return topk_w.to(x.dtype), topk_idx, probs

    def balance_loss(self, probs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Switch-style load-balancing loss, given the full router probs and the number
        of tokens each expert received.

        torch has no such loss to import, and top-k routing collapses onto a handful of
        experts without one. f (share of assignments) dotted with P (mean router prob)
        is minimized by a uniform split, and P is what carries the gradient, since f is
        piecewise constant in the router weights.
        """
        f = counts.float() / counts.sum().clamp(min=1)
        return self.aux_loss_coef * self.n_experts * (f * probs.mean(0)).sum()
