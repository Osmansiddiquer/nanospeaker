"""
Per-step metrics, one flat JSON object per line.

Flat keys and one line per optimizer step, so a dashboard can tail the file and plot any
field without parsing structure. Written with an fsync-free flush every line and a real
flush every `sync_every` -- losing the last few lines to a crash is acceptable, losing
the middle of the file to a partial write is not.

What is worth logging, and why, since a metric nobody can act on is a metric nobody
should pay for:

  loss, loss_ema, perplexity   the objective, smoothed, and in units people compare
  aux_loss                     MoE balancing; if this climbs the router is collapsing
  grad_norm, clip_frac         how often the clip is binding -- if always, the LR is wrong
  update_norm_ratio            ||dW|| / ||W||. The earliest warning of divergence there
                               is, and it moves before the loss does
  expert_load_cv               spread of tokens across experts. 0 is uniform; above ~0.5
                               a few experts are doing all the work
  experts_unused               experts that saw nothing this step. Non-zero is a dead
                               capacity you paid for
  router_entropy               how decisive the router is; falling to 0 means collapse
  tok_per_s, step_time_s       throughput, and where it went
  mem_reserved_gb              the number that OOMs, not the allocated one
  mix_*                        realized source ratios, not the intended ones
"""

import json
import math
import time
from pathlib import Path

import torch


class MetricsLogger:
    def __init__(self, path, sync_every: int = 20, ema_beta: float = 0.98):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a")
        self.sync_every, self.ema_beta = sync_every, ema_beta
        self.ema, self.n = None, 0

    def update_ema(self, loss: float) -> float:
        self.ema = loss if self.ema is None else \
            self.ema_beta * self.ema + (1 - self.ema_beta) * loss
        return self.ema

    def log(self, **row) -> None:
        self.fh.write(json.dumps(row) + "\n")
        self.n += 1
        if self.n % self.sync_every == 0:
            self.fh.flush()

    def close(self) -> None:
        self.fh.flush()
        self.fh.close()


@torch.no_grad()
def moe_health(model) -> dict:
    """
    Router diagnostics, averaged over the MoE layers.

    expert_load_cv is the coefficient of variation of tokens per expert: 0 is a perfectly
    even split, and it rises without bound as load concentrates. It is scale-free, so the
    same threshold reads the same at 16 experts or 76.
    """
    cvs, unused, entropies, dropped = [], [], [], []
    for block in getattr(model, "blocks", []):
        ffn = getattr(block, "ffn", None)
        counts = getattr(ffn, "expert_counts", None)
        if counts is None or counts.numel() == 0:
            continue
        c = counts.float()
        total = c.sum().clamp(min=1)
        cvs.append((c.std() / c.mean().clamp(min=1e-9)).item())
        unused.append((c == 0).sum().item())

        p = c / total                              # empirical load distribution
        nz = p[p > 0]
        entropies.append(float(-(nz * nz.log()).sum() / math.log(max(len(c), 2))))
        dropped.append(float(getattr(ffn, "dropped", 0)) / float(total))

    if not cvs:
        return {}
    return {
        "expert_load_cv": sum(cvs) / len(cvs),
        "experts_unused": sum(unused) / len(unused),
        "router_entropy": sum(entropies) / len(entropies),
        "tokens_dropped_frac": sum(dropped) / len(dropped),
    }


@torch.no_grad()
def update_norm_ratio(model, before: dict) -> float:
    """||dW|| / ||W|| over the parameters Muon owns, against a pre-step snapshot."""
    num = den = 0.0
    for name, p in model.named_parameters():
        if name not in before:
            continue
        num += (p.detach() - before[name]).float().pow(2).sum().item()
        den += p.detach().float().pow(2).sum().item()
    return math.sqrt(num / max(den, 1e-12))


def snapshot(model, names) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in names}


class Timer:
    """Wall-clock accounting, split so a slow loader cannot hide inside the step time."""

    def __init__(self):
        self.marks, self.t0 = {}, time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = self.marks.get(name, 0.0) + (now - self.t0)
        self.t0 = now

    def reset(self) -> None:
        self.marks, self.t0 = {}, time.perf_counter()
