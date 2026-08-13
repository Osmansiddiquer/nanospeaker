"""
Train nanoSpeaker.

Resumption is the design constraint, not an afterthought. Batches are a pure function of
(seed, step, micro_index), so the checkpoint carries no data-loader state and `--resume-step`
can start anywhere: the tokens at step 9,000 are the same tokens whether the run reached
them by training or by being told to begin there. The learning rate is likewise computed
from the step number rather than accumulated, so the schedule cannot drift out of phase
with the weights.

    python -m src.train.train --steps 12512 --accum 12
    python -m src.train.train --resume checkpoints/step_09000.pt
    python -m src.train.train --resume-step 9000     # from the newest checkpoint

Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Dropless MoE dispatch requests a
different shape every step and the default allocator fragments against it: 3955 MiB
reserved versus 2373 with expandable segments, on the same work.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from ..data.loader import MixtureLoader
from ..model.nanospeaker import NanoSpeaker, NanoSpeakerConfig
from .metrics import MetricsLogger, Timer, moe_health, snapshot, update_norm_ratio
from .optim import build_optimizers, muon_parameter_names, wsd_lr


def save_checkpoint(path, model, optims, step, tokens_seen, cfg, args):
    """Everything needed to continue. The loader needs nothing, by construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "muon": optims["muon"].state_dict(),
        "adamw": optims["adamw"].state_dict(),
        "config": vars(cfg),
        "args": vars(args),
    }, path)
    # Written last: a run that dies mid-save leaves the pointer aimed at the previous
    # checkpoint rather than at a truncated one.
    (path.parent / "latest.txt").write_text(path.name)


def load_checkpoint(path, model, optims, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optims["muon"].load_state_dict(ck["muon"])
    optims["adamw"].load_state_dict(ck["adamw"])
    return ck["step"], ck.get("tokens_seen", 0)


@torch.no_grad()
def evaluate(model, args, device, batches: int = 20):
    """Loss on each source's held-out split, drawn the same way training draws."""
    model.eval()
    out = {}
    for name in ("python", "web", "code_instruct"):
        try:
            loader = MixtureLoader(
                token_dir=args.token_dir, seq_len=args.seq_len,
                micro_batch=args.micro_batch, seed=args.seed + 1,
                total_steps=args.steps, split="valid_",
                stable_mix={name: 1.0}, decay_mix={name: 1.0},
            )
        except FileNotFoundError:
            continue
        total = 0.0
        for i in range(batches):
            x, y, _ = loader.batch(i, 0, device)
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                loss, _ = model(x, y)
            total += loss.item()
        out[f"valid_{name}"] = total / batches
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12_512)
    ap.add_argument("--accum", type=int, default=12, help="micro-steps per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-adamw", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0)       # WSD, no warmup
    ap.add_argument("--decay-start", type=float, default=0.80)
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--out", default="runs/nanospeaker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default=None, help="path to a checkpoint")
    ap.add_argument("--resume-step", type=int, default=None,
                    help="resume from the newest checkpoint, asserting this step")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = NanoSpeakerConfig(max_seq_len=args.seq_len)
    model = NanoSpeaker(cfg).to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    optims, (n_muon, n_embed, n_other) = build_optimizers(
        model, args.lr_muon, args.lr_adamw, args.weight_decay
    )
    muon_names = muon_parameter_names(model)
    base = model                       # uncompiled handle: compile hides .blocks

    step, tokens_seen = 0, 0
    ckpt_dir = out / "checkpoints"
    if args.resume or args.resume_step is not None:
        path = Path(args.resume) if args.resume else \
            ckpt_dir / (ckpt_dir / "latest.txt").read_text().strip()
        step, tokens_seen = load_checkpoint(path, model, optims, device)
        if args.resume_step is not None and step != args.resume_step:
            raise SystemExit(f"checkpoint is at step {step}, asked for {args.resume_step}")
        print(f"resumed from {path} at step {step:,} ({tokens_seen/1e9:.3f}B tokens)")

    if args.compile:
        # One graph, no breaks -- the dropless MoE path is compile-clean. `base` keeps
        # pointing at the real module so metrics and checkpoints see through the wrapper.
        model = torch.compile(model)

    loader = MixtureLoader(
        token_dir=args.token_dir, seq_len=args.seq_len, micro_batch=args.micro_batch,
        seed=args.seed, decay_start=args.decay_start, total_steps=args.steps,
    )
    tok_per_step = loader.tokens_per_step(args.accum)
    logger = MetricsLogger(out / "metrics.jsonl")

    print(f"nanoSpeaker {base.n_params():,} params | "
          f"muon {n_muon} tensors, adamw {n_embed + n_other}")
    print(f"{args.steps:,} steps x {tok_per_step:,} tokens = "
          f"{args.steps*tok_per_step/1e9:.2f}B | sources {loader.summary()}")

    timer = Timer()
    for step in range(step, args.steps):
        lr_scale = wsd_lr(step, args.steps, args.decay_start, args.warmup)
        for g in optims["muon"].param_groups:
            g["lr"] = args.lr_muon * lr_scale
        for g in optims["adamw"].param_groups:
            g["lr"] = args.lr_adamw * lr_scale

        for o in optims.values():
            o.zero_grad(set_to_none=True)

        timer.reset()
        total_loss = total_aux = 0.0
        realized = {}
        for micro in range(args.accum):
            x, y, mix = loader.batch(step, micro, device)
            timer.mark("data")
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                loss, aux = model(x, y)
                scaled = (loss + aux) / args.accum
            scaled.backward()
            timer.mark("compute")
            total_loss += loss.item() / args.accum
            total_aux += float(aux) / args.accum
            for k, v in mix.items():
                realized[k] = realized.get(k, 0.0) + v / args.accum

        before = snapshot(base, muon_names) if step % args.log_every == 0 else {}
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for o in optims.values():
            o.step()
        timer.mark("optim")

        tokens_seen += tok_per_step
        if step % args.log_every == 0:
            row = {
                "step": step,
                "tokens_seen": tokens_seen,
                "epoch_frac": tokens_seen / max(sum(loader.summary().values()), 1),
                "phase": loader.phase(step),
                "loss": total_loss,
                "loss_ema": logger.update_ema(total_loss),
                "aux_loss": total_aux,
                "perplexity": math.exp(min(total_loss, 20)),
                "lr_muon": args.lr_muon * lr_scale,
                "lr_adamw": args.lr_adamw * lr_scale,
                "grad_norm": float(grad_norm),
                "clip_frac": float(grad_norm > args.clip),
                "update_norm_ratio": update_norm_ratio(base, before) if before else None,
                "step_time_s": sum(timer.marks.values()),
                "data_wait_s": timer.marks.get("data", 0.0),
                "tok_per_s": tok_per_step / max(sum(timer.marks.values()), 1e-9),
                **{f"mix_{k}": v for k, v in realized.items()},
                **moe_health(base),
            }
            if device == "cuda":
                row["mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
                row["mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if step % args.eval_every == 0 and step > 0:
                row.update(evaluate(model, args, device))
            logger.log(**row)

            if step % 10 == 0:
                print(f"  step {step:>6,}  loss {total_loss:6.3f}  ppl "
                      f"{row['perplexity']:8.1f}  {row['tok_per_s']:6.0f} tok/s  "
                      f"cv {row.get('expert_load_cv', float('nan')):.2f}", flush=True)

        if step % args.save_every == 0 and step > 0:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", base, optims,
                            step, tokens_seen, cfg, args)

    save_checkpoint(ckpt_dir / f"step_{args.steps:06d}.pt", base, optims,
                    args.steps, tokens_seen, cfg, args)
    logger.close()
    print(f"done: {args.steps:,} steps, {tokens_seen/1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
