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
import sys
import math
import os
import time
from pathlib import Path

import torch

from ..data.loader import DECAY_MIX, STABLE_MIX, MixtureLoader
from ..data.prefetch import BatchPrefetcher
from ..model.ln import RMSNorm
from ..model.nanospeaker import NanoSpeaker, NanoSpeakerConfig
from .metrics import GpuSpans, MetricsLogger, Timer, moe_health
from .optim import build_optimizers, wsd_lr


def prune_checkpoints(ckpt_dir, keep: int) -> None:
    """Keep the newest `keep` checkpoints. Exactly that many, no exceptions."""
    for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
        old.unlink(missing_ok=True)


def save_checkpoint(path, model, optims, step, tokens_seen, cfg, args):
    """Everything needed to continue. The loader needs nothing, by construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temp name and renamed. os.replace is atomic, so a crash mid-write
    # leaves a .tmp on disk and never a truncated file wearing a real checkpoint's name
    # -- which is what a pointer file used to be for, at the cost of a second file to
    # keep straight.
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "muon": optims["muon"].state_dict(),
        "adamw": optims["adamw"].state_dict(),
        "config": vars(cfg),
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optims, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optims["muon"].load_state_dict(ck["muon"])
    optims["adamw"].load_state_dict(ck["adamw"])
    return ck["step"], ck.get("tokens_seen", 0)


@torch.no_grad()
def evaluate(model, args, device, batches: int = 1):
    """Loss on each source's held-out split, drawn the same way training draws."""
    model.eval()
    out = {}
    # Every source the run actually draws from, not a fixed list: a source added to the
    # mix but missing from here trains unmeasured, which is the one thing validation
    # exists to prevent.
    names = sorted(set(args.stable_mix or STABLE_MIX) | set(args.decay_mix or DECAY_MIX))
    for name in names:
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
    out["valid_batches"] = batches
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
    ap.add_argument("--warmup", type=int, default=200,
                    help="LR warmup steps; the router is on AdamW and needs it even "
                         "though Muon's update is already norm-bounded")
    ap.add_argument("--noise-std", type=float, default=0.5,
                    help="router exploration noise at step 0, annealed to 0")
    ap.add_argument("--noise-until", type=float, default=0.10,
                    help="fraction of the run over which noise decays to 0")
    ap.add_argument("--aux-coef", type=float, default=0.02)
    ap.add_argument("--decay-start", type=float, default=0.80)
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--out", default="runs/nanospeaker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10,
                    help="quick validation cadence, in optimizer steps")
    ap.add_argument("--eval-batches", type=int, default=1,
                    help="batches per source for the quick validation (8 sequences each)")
    ap.add_argument("--full-eval-every", type=int, default=500)
    ap.add_argument("--full-eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--keep-checkpoints", type=int, default=1,
                    help="older checkpoints are deleted; 250 of them would be 393 GB")
    ap.add_argument("--resume", default=None, help="path to a checkpoint")
    ap.add_argument("--resume-step", type=int, default=None,
                    help="resume from the newest checkpoint, asserting this step")
    ap.add_argument("--auto-resume", action="store_true",
                    help="continue from the newest checkpoint if one exists, else start "
                         "fresh -- the shutdown-safe mode: one command works whether the "
                         "run is new or coming back from a power cut")
    ap.add_argument("--force-init", action="store_true",
                    help="allow --init-from into a directory that already has "
                         "checkpoints, discarding that phase's progress")
    ap.add_argument("--init-from", default=None,
                    help="start a NEW phase from an old checkpoint: loads the weights "
                         "and optimizer state but resets step and tokens_seen to 0, so "
                         "the WSD schedule (warmup included) runs from the top. --resume "
                         "keeps the step number and is for crash recovery within a phase")
    ap.add_argument("--stable-mix", type=json.loads, default=None,
                    help='JSON source ratios for the stable phase, e.g. \'{"web":0.6}\'')
    ap.add_argument("--decay-mix", type=json.loads, default=None,
                    help="JSON source ratios for the decay phase")
    ap.add_argument("--min-doc-len", type=json.loads, default=None,
                    help='draw windows inside documents at least this long. One number, '
                         'or JSON per source: \'{"web":8192,"qa":768}\'. Sources with '
                         'fewer than 64 qualifying documents fall back to the flat stream')
    ap.add_argument("--window", type=int, default=None,
                    help="sliding-window attention span; unset means full attention")
    ap.add_argument("--global-every", type=int, default=None,
                    help="every Nth layer attends globally; with --window this is the "
                         "3:1 local/global interleave")
    ap.add_argument("--rope-base", type=float, default=10_000.0,
                    help="raise for long context; the tables are non-persistent buffers "
                         "so changing it stays checkpoint-compatible")
    ap.add_argument("--ckpt-skip", type=int, default=0,
                    help="blocks left un-checkpointed; spends spare VRAM on skipping "
                         "their recompute")
    ap.add_argument("--compile", action="store_true",
                    help="whole-model compile; OOMs this card, kept for larger ones")
    ap.add_argument("--compile-norms", action=argparse.BooleanOptionalAction, default=True,
                    help="fuse the RMSNorm chain inductor-side; worth 27%% of the step")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="batches built ahead on a worker thread; 0 builds them inline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = NanoSpeakerConfig(max_seq_len=args.seq_len, noise_std=args.noise_std,
                            aux_loss_coef=args.aux_coef, ckpt_skip=args.ckpt_skip,
                            window=args.window, global_every=args.global_every,
                            rope_base=args.rope_base)
    model = NanoSpeaker(cfg).to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    optims, (n_muon, n_embed, n_other) = build_optimizers(
        model, args.lr_muon, args.lr_adamw, args.weight_decay
    )
    base = model                       # uncompiled handle: compile hides .blocks

    step, tokens_seen = 0, 0
    ckpt_dir = out / "checkpoints"
    have_ckpt = ckpt_dir.exists() and any(ckpt_dir.glob("step_*.pt"))
    if args.init_from:
        # A new phase, not a continuation: the weights and optimizer moments carry over,
        # the schedule does not. Without this, `warmup` can never fire on a model that
        # has already passed step 200, and a fresh phase would start at whatever point
        # of the previous cosine its step number happened to land on.
        if args.resume or args.resume_step is not None or args.auto_resume:
            raise SystemExit("--init-from starts a new phase; --resume continues one")
        if have_ckpt and not args.force_init:
            # The launcher is re-run after a crash far more often than a phase is
            # started. Re-initializing here would reset the step counter to 0 and then
            # the lexicographic pruner would delete every checkpoint the restarted run
            # writes until its step number passed the stale names -- hours of training
            # producing nothing, behind two plausible-looking stale files.
            raise SystemExit(
                f"{ckpt_dir} already holds checkpoints: this phase has already started. "
                f"Resume it with --auto-resume, or pass --force-init to start over.")
        load_checkpoint(Path(args.init_from), model, optims, device)
        print(f"initialized from {args.init_from} at step 0 (new schedule)")
    elif args.resume or args.resume_step is not None or (args.auto_resume and have_ckpt):
        # Newest by step number, read straight off the filenames.
        path = Path(args.resume) if args.resume else \
            sorted(ckpt_dir.glob("step_*.pt"))[-1]
        step, tokens_seen = load_checkpoint(path, model, optims, device)
        if args.resume_step is not None and step != args.resume_step:
            raise SystemExit(f"checkpoint is at step {step}, asked for {args.resume_step}")
        print(f"resumed from {path} at step {step:,} ({tokens_seen/1e9:.3f}B tokens)")

    if args.compile_norms and device == "cuda":
        # 81 norm calls twice a step, and `F.rms_norm` is not the fused kernel its name
        # suggests -- it decomposes to a pow/mean/rsqrt chain in fp32, which profiled at
        # 4.94 s of an 11.05 s step in elementwise kernels and DtoD copies. Compiling the
        # module fuses that chain: 923 -> 672 ms per micro measured here, peak memory
        # slightly *lower*. Binding `.forward` leaves the module identity, and therefore
        # the state dict and every checkpoint, untouched.
        for mod in model.modules():
            if isinstance(mod, RMSNorm):
                mod.forward = torch.compile(mod.forward)

    if args.compile:
        # Scoped only. Compiling the whole model materializes a [32768, 576] buffer in
        # the tied-head region on the first compiled step and OOMs this card.
        model = torch.compile(model)

    # `is not None`, not truthiness: a shell-quoting accident that collapses the flag to
    # `{}` would otherwise fall back to the module defaults -- which for this phase are
    # the exact inverse of the intended mix, announced by a normal-looking startup line.
    mixes = {k: v for k, v in (("stable_mix", args.stable_mix),
                               ("decay_mix", args.decay_mix),
                               ("min_doc_len", args.min_doc_len)) if v is not None}
    loader = MixtureLoader(
        token_dir=args.token_dir, seq_len=args.seq_len, micro_batch=args.micro_batch,
        seed=args.seed, decay_start=args.decay_start, total_steps=args.steps, **mixes,
    )
    tok_per_step = loader.tokens_per_step(args.accum)
    # The worker builds batch N+1 while N trains. Same tokens, off the critical path.
    fetch = (BatchPrefetcher(loader, args.accum, device, args.prefetch, args.steps)
             if args.prefetch else None)
    next_batch = fetch.get if fetch else (lambda s, m: loader.batch(s, m, device))
    logger = MetricsLogger(out / "metrics.jsonl")

    print(f"nanoSpeaker {base.n_params():,} params | "
          f"muon {n_muon} tensors, adamw {n_embed + n_other}")
    print(f"{args.steps:,} steps x {tok_per_step:,} tokens = "
          f"{args.steps*tok_per_step/1e9:.2f}B | sources {loader.summary()}")

    # What metrics.jsonl cannot say. `window`, `global_every` and the mixes live only
    # inside checkpoints, and those get pruned to two -- so a month from now nothing on
    # disk would distinguish this from a full-attention run on the old mix. The split
    # mtimes are here because valid_web.bin was re-carved between phases: joining its
    # numbers to phase 1's would be comparing two different held-out sets.
    splits = {f"{sp}{n}.bin": Path(args.token_dir) / f"{sp}{n}.bin"
              for sp in ("", "valid_") for n in loader.summary()}
    (out / "run.json").write_text(json.dumps({
        "argv": sys.argv, "args": vars(args), "config": vars(cfg),
        "layer_windows": cfg.layer_windows(),
        "stable_mix": loader.stable_mix, "decay_mix": loader.decay_mix,
        "tokens_per_step": tok_per_step, "sources": loader.summary(),
        "splits": {k: {"bytes": v.stat().st_size, "mtime": v.stat().st_mtime}
                   for k, v in splits.items() if v.exists()},
        "started": time.time(),
    }, indent=2, default=str))

    def router_noise(s: int) -> float:
        """
        Exploration noise, decayed to zero over the opening of the run.

        Early on the router has no signal beyond its initialization, and whichever
        experts that favours win every token forever -- 24 of 76 died within four steps
        without this. Once the experts have differentiated the noise is only damage, so
        it anneals away rather than staying on (ST-MoE reports the same).
        """
        knee = max(args.noise_until * args.steps, 1)
        return args.noise_std * max(0.0, 1.0 - s / knee)

    if step == 0:
        # The baseline this phase will be judged against, measured before it moves.
        logger.log(step=0, **evaluate(model, args, device, args.full_eval_batches))

    timer, gpu = Timer(), GpuSpans(device == "cuda")
    for step in range(step, args.steps):
        t_iter = time.perf_counter()   # the whole iteration, validation and saving included
        lr_scale = wsd_lr(step, args.steps, args.decay_start, args.warmup)
        noise = router_noise(step)
        for blk in base.blocks:
            blk.ffn.router.noise_std = noise
        for g in optims["muon"].param_groups:
            g["lr"] = args.lr_muon * lr_scale
        for g in optims["adamw"].param_groups:
            g["lr"] = args.lr_adamw * lr_scale

        for o in optims.values():
            o.zero_grad(set_to_none=True)

        timer.reset()
        # Summed on device. A per-micro `.item()` blocks until the GPU has drained that
        # micro's backward -- 25 stalls a step, and because the stall lands between
        # mark("compute") and the next mark("data") it was charged to the loader, which
        # is how 0.03 s of reading came to be logged as 1.48 s of data wait.
        loss_sum = torch.zeros((), device=device)
        aux_sum = torch.zeros((), device=device)
        realized = {}
        for micro in range(args.accum):
            x, y, mix = next_batch(step, micro)
            timer.mark("data")
            with gpu.span("fwd"), torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                loss, aux = model(x, y)
                scaled = (loss + aux) / args.accum
            with gpu.span("bwd"):
                scaled.backward()
            loss_sum += loss.detach()
            aux_sum += aux.detach()
            timer.mark("compute")
            for k, v in mix.items():
                realized[k] = realized.get(k, 0.0) + v / args.accum

        total_loss = loss_sum.item() / args.accum        # the step's one sync
        total_aux = aux_sum.item() / args.accum
        timer.mark("compute")                            # the drain is compute; log it there

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for o in optims.values():
            o.step()
        if device == "cuda":
            torch.cuda.synchronize()                     # so "optim" isn't billed to "data"
        timer.mark("optim")
        spans = gpu.totals()                             # safe: the work above has drained

        tokens_seen += tok_per_step

        # Validation and checkpointing run before the metrics line, so the step that paid for
        # them is the row that reports them. Billed to the next step -- or to nothing, as
        # they were when step_time_s was the sum of the training marks only -- they show up
        # as an unexplained periodic spike in throughput, which is how a known 20-batch
        # evaluation gets mistaken for a stall.
        logging_now = step % args.log_every == 0
        valid, t_eval = {}, 0.0
        if logging_now:
            t0 = time.perf_counter()
            # Two cadences: a cheap read every few steps to see the curve move, and a
            # fuller one occasionally for a number worth quoting.
            if step % args.full_eval_every == 0 and step > 0:
                valid = evaluate(model, args, device, args.full_eval_batches)
            elif step % args.eval_every == 0 and step > 0:
                valid = evaluate(model, args, device, args.eval_batches)
            t_eval = time.perf_counter() - t0

        t0 = time.perf_counter()
        if step % args.save_every == 0 and step > 0:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", base, optims,
                            step, tokens_seen, cfg, args)
            prune_checkpoints(ckpt_dir, args.keep_checkpoints)
        t_ckpt = time.perf_counter() - t0

        if logging_now:
            step_time = time.perf_counter() - t_iter
            row = {
                "step": step,
                "tokens_seen": tokens_seen,
                "epoch_frac": tokens_seen / max(sum(loader.summary().values()), 1),
                "phase": loader.phase(step),
                "loss": total_loss,
                "loss_ema": logger.update_ema(total_loss),
                "aux_loss": total_aux,
                "perplexity": math.exp(min(total_loss, 20)),
                "noise_std": noise,
                "lr_muon": args.lr_muon * lr_scale,
                "lr_adamw": args.lr_adamw * lr_scale,
                "grad_norm": float(grad_norm),
                "clip_frac": float(grad_norm > args.clip),
                "update_norm_ratio": optims["muon"].update_norm_ratio(),
                "step_time_s": step_time,
                "data_wait_s": timer.marks.get("data", 0.0),
                # Where the step went. fwd/bwd are GPU-side and so split the compute mark
                # between them; the rest are wall clock. They will not sum to step_time_s,
                # and the remainder -- python, logging, the allocator -- is worth seeing.
                "t_data_s": timer.marks.get("data", 0.0),
                "t_compute_s": timer.marks.get("compute", 0.0),
                "t_fwd_s": spans.get("fwd", 0.0),
                "t_bwd_s": spans.get("bwd", 0.0),
                "t_optim_s": timer.marks.get("optim", 0.0),
                "t_eval_s": t_eval,
                "t_ckpt_s": t_ckpt,
                "tok_per_s": tok_per_step / max(step_time, 1e-9),
                **{f"mix_{k}": v for k, v in realized.items()},
                **moe_health(base),
                **valid,
            }
            if device == "cuda":
                # Current, then peak-since-last-read. `max_memory_reserved` alone
                # never falls, so a single transient latches every downstream threshold
                # on for the rest of the run -- which is how a memory alarm fires 615
                # times and stops being read.
                row["mem_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
                row["mem_reserved_peak_gb"] = torch.cuda.max_memory_reserved() / 2**30
                row["mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                torch.cuda.reset_peak_memory_stats()
            logger.log(**row)

            if step % 10 == 0:
                print(f"  step {step:>6,}  loss {total_loss:6.3f}  ppl "
                      f"{row['perplexity']:8.1f}  {row['tok_per_s']:6.0f} tok/s  "
                      f"cv {row.get('expert_load_cv', float('nan')):.2f}", flush=True)

    # The loop's last full eval lands 500 steps before the end; close the run on a
    # measured number rather than an extrapolated one.
    logger.log(step=args.steps, **evaluate(model, args, device, args.full_eval_batches))
    save_checkpoint(ckpt_dir / f"step_{args.steps:06d}.pt", base, optims,
                    args.steps, tokens_seen, cfg, args)
    prune_checkpoints(ckpt_dir, args.keep_checkpoints)
    if fetch:
        fetch.close()
    logger.close()
    print(f"done: {args.steps:,} steps, {tokens_seen/1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
