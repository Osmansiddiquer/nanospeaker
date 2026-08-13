"""
Muon for the hidden matrices, AdamW for everything else, and a WSD learning rate.

Muon (Jordan et al., 2024) replaces the momentum buffer's raw direction with its nearest
orthogonal matrix, computed by a Newton-Schulz iteration. One state tensor per parameter
against Adam's two, and measured here at 1.8% throughput cost over stateless SGD -- while
AdamW over the same 382M parameters would not fit on the card at all.

Two details decide whether the iteration costs 1% of a step or 26%:

  * the matrix is transposed so its *smaller* dimension leads, because Newton-Schulz costs
    2*m^2*n + m^3 in that dimension. On a 576x128 expert matrix, getting this backwards is
    13x more arithmetic.
  * experts arrive pre-stacked as [n_experts, d_model, d_expert], so `@` and `.mT` are
    batched matmuls over the whole layer -- 40 tensors for a 20-layer model rather than
    4,680 individual calls.

Embeddings, norms and the router stay on AdamW: orthogonalizing a router actively works
against what it is for, and the embedding is a lookup table, not a linear map.
"""

import math

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz: approximates the orthogonal factor of G. Batched-safe."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()

    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT                                   # min dimension first -- see docstring
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if transposed else X


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, orthogonalized. Momentum is held in bf16 to halve its cost."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.1, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    def load_state_dict(self, state_dict):
        """Restore, then put the momentum back in bf16.

        The base implementation casts state to the parameter's dtype, which would
        silently double this optimizer's memory on every resume -- the one property it
        exists for.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "momentum" in st:
                    st["momentum"] = st["momentum"].bfloat16()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)

                # Match the buffer's dtype rather than assuming bf16: load_state_dict
                # casts optimizer state to the parameter dtype, so a resumed run arrives
                # here with an fp32 buffer and a mismatched lerp_ would raise.
                buf = state["momentum"]
                grad = p.grad.to(buf.dtype)
                buf.lerp_(grad, 1 - group["momentum"])
                update = zeropower_via_newtonschulz5(
                    grad.lerp(buf, group["momentum"]), group["ns_steps"]
                )
                # Wider-than-tall matrices take larger steps under an orthogonal update,
                # so the scale compensates by the aspect ratio.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)


def split_parameters(model):
    """Hidden matrices to Muon; embedding, norms and router to AdamW."""
    muon, embed, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name:
            embed.append((name, p))
        elif "router" in name or p.ndim < 2:
            other.append((name, p))
        else:
            muon.append((name, p))
    return muon, embed, other


def muon_parameter_names(model) -> set:
    """The parameters Muon owns, by name -- for the update-norm diagnostic."""
    return {n for n, _ in split_parameters(model)[0]}


def build_optimizers(model, lr_muon=0.02, lr_adamw=2e-3, weight_decay=0.1):
    muon_p, embed_p, other_p = (
        [p for _, p in g] for g in split_parameters(model)
    )
    muon_p, embed_p, other_p = list(muon_p), list(embed_p), list(other_p)
    muon = Muon(muon_p, lr=lr_muon, momentum=0.95, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(
        [{"params": embed_p, "weight_decay": weight_decay},
         {"params": other_p, "weight_decay": 0.0}],
        lr=lr_adamw, betas=(0.9, 0.95), eps=1e-8,
        # foreach=False: the multi-tensor path allocates one temporary spanning every
        # state at once, which OOMs a 4 GB card at this size.
        foreach=False,
    )
    return {"muon": muon, "adamw": adamw}, (len(muon_p), len(embed_p), len(other_p))


def wsd_lr(step: int, total_steps: int, decay_start: float = 0.80,
           warmup: int = 0, final_frac: float = 0.0) -> float:
    """
    Warmup-Stable-Decay, as a multiplier on the base learning rate.

    Flat through the stable phase, then a cosine fall to `final_frac` across the decay.
    Warmup defaults to zero steps: Muon's update is already norm-controlled by the
    orthogonalization, so the usual reason for warming up -- unbounded early updates from
    a cold second moment -- does not apply to the parameters that hold most of the model.
    """
    if warmup and step < warmup:
        return (step + 1) / warmup

    knee = decay_start * total_steps
    if step < knee:
        return 1.0

    progress = min((step - knee) / max(total_steps - knee, 1), 1.0)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))
