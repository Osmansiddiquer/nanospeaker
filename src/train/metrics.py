"""
Per-step metrics, one flat JSON object per line.

Flat keys and one line per optimizer step, so a dashboard can tail the file and plot any
field without parsing structure. Flushed every line by default: at ~14s per step a flush
costs nothing, and buffering meant anything reading the file was up to 20 steps behind --
which defeats the point of writing it at all.

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
  ts                           wall-clock of the write, so steps can be placed in real time
  tok_per_s, step_time_s       throughput, and the wall clock of the whole iteration
  t_*_s                        where that clock went: forward, backward, optimiser, loader,
                               validation, checkpointing. What they do not add up to is
                               overhead, and it is worth seeing that gap
  mem_reserved_gb              the number that OOMs, not the allocated one
  mix_*                        realized source ratios, not the intended ones
"""

import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class MetricsLogger:
    def __init__(self, path, sync_every: int = 1, ema_beta: float = 0.98):
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
        # Wall-clock, so a step can be tied back to when it actually ran -- which log line
        # was during the outage, which was after the restart. Unix seconds, UTC by
        # construction; whoever reads it picks the timezone.
        row.setdefault("ts", time.time())
        self.fh.write(json.dumps(row) + "\n")
        self.n += 1
        if self.n % self.sync_every == 0:
            self.fh.flush()
        # flush() only reaches the page cache. A power cut between there and the disk is
        # what produced a 752-byte NUL tail that broke every reader of this file; fsync
        # bounds that window to 50 steps, for a few milliseconds against an 11 s step.
        if self.n % 50 == 0:
            os.fsync(self.fh.fileno())

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


class GpuSpans:
    """
    GPU-side durations, without stalling the loop to get them.

    A wall clock wrapped around a CUDA call measures when the work was *queued*, not when it
    ran, so the CPU cannot tell forward from backward: both return immediately and the real
    time surfaces at the next synchronisation. Events are timestamps the GPU writes into its
    own stream as it passes them. Recording costs a few microseconds and no stall; they are
    read once per step, after the sync the loop already performs.

    Event pairs are recycled rather than reallocated, because at 12 micro-steps a step this
    would otherwise create a few hundred CUDA events a minute for no reason.
    """

    def __init__(self, enabled: bool = True):
        self.enabled, self.spans, self.free = enabled, [], []

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        a, b = self.free.pop() if self.free else (
            torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        a.record()
        try:
            yield
        finally:
            b.record()
            self.spans.append((name, a, b))

    def totals(self) -> dict:
        """Seconds per span name since the last call. Only valid once the work has drained."""
        out = {}
        for name, a, b in self.spans:
            out[name] = out.get(name, 0.0) + a.elapsed_time(b) / 1000.0
            self.free.append((a, b))
        self.spans.clear()
        return out


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
