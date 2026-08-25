"""
SFT: response-masked fine-tuning on the packed ChatML corpus from build_sft.py.

Mechanically identical to pretraining -- same model, same fused CE, same optimizers --
with two differences: targets carry -100 over every non-assistant token (the CE path
skips ignore_index positions, so the model is only ever paid for assistant content and
its closing <|im_end|>), and the schedule is short and gentle (0.1x LR, cosine to 0).

    python -m src.train.sft --steps 5800                  # ~2 epochs at defaults
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..model.nanospeaker import NanoSpeaker, NanoSpeakerConfig
from .metrics import moe_health
from .optim import build_optimizers

IGNORE = -100


class PackedRows:
    """Deterministic row batches over a packed token+mask stream (pure in step)."""

    def __init__(self, stem: str, seq_len: int, micro_batch: int, seed: int = 0):
        self.ids = np.memmap(f"{stem}.bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(f"{stem}.mask", dtype=np.uint8, mode="r")
        self.seq_len, self.micro_batch, self.seed = seq_len, micro_batch, seed
        self.n_rows = len(self.ids) // seq_len

    def rows_per_step(self, accum: int) -> int:
        return self.micro_batch * accum

    def batch(self, step: int, micro: int, accum: int, device: str):
        L, mb = self.seq_len, self.micro_batch
        rows_per_epoch = self.n_rows // (mb * accum) * (mb * accum)
        first = (step * accum + micro) * mb
        epoch = first // rows_per_epoch
        perm = np.random.default_rng((self.seed, epoch)).permutation(self.n_rows)
        picks = perm[[(first + i) % rows_per_epoch for i in range(mb)]]
        x = np.stack([self.ids[r * L:(r + 1) * L] for r in picks]).astype(np.int64)
        m = np.stack([self.mask[r * L:(r + 1) * L] for r in picks])
        # Rows are arbitrary slices of the stream, so a row head is usually the
        # orphaned tail of a split conversation: a few answer tokens, a dangling
        # </think>, an <|im_end|> -- supervised on ~no context. That trained a
        # measurable say-nothing attractor (P(im_end)=0.17 as FIRST token on short
        # prompts). Mask everything before the row's first turn start.
        for j in range(x.shape[0]):
            starts = np.nonzero(x[j] == 2)[0]
            m[j, :starts[0] if len(starts) else L] = 0
        y = np.where(m == 1, x, IGNORE)
        return (torch.from_numpy(x).to(device, non_blocking=True),
                torch.from_numpy(y).to(device, non_blocking=True))


def sft_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model, device, seq_len, batches=4, data_dir="data/sft"):
    model.eval()
    out, used = {}, []
    for stem in sorted(Path(data_dir).glob("valid_*.bin")):
        group = stem.stem.replace("valid_", "")
        rows = PackedRows(str(stem.with_suffix("")), seq_len, 2)
        # Tiny streams (identity: ~2k tokens) can't fill one seq_len row; shrink
        # the window until at least one two-row batch exists.
        L = seq_len
        while rows.n_rows < 2 and L > 256:
            L //= 2
            rows = PackedRows(str(stem.with_suffix("")), L, 2)
        losses = []
        for b in range(min(batches, rows.n_rows // 2)):
            x, y = rows.batch(0, b, 1, device)
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                loss, _ = model(x, y)
            v = loss.item()
            if math.isfinite(v):
                losses.append(v)
        if losses:
            out[f"valid_{group}"] = round(sum(losses) / len(losses), 4)
            used.append(len(losses))
    if used:
        out["valid_batches"] = batches      # the size class: 1 = peek, 4 = solid
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-from", default="runs/nanospeaker_p2b/model.pt")
    ap.add_argument("--out", default="runs/nanospeaker_sft")
    ap.add_argument("--steps", type=int, default=5800)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--ckpt-skip", type=int, default=4)
    ap.add_argument("--lr-muon", type=float, default=2e-3)
    ap.add_argument("--lr-adamw", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200,
                    help="full 4-batch validation cadence")
    ap.add_argument("--eval-quick-every", type=int, default=20,
                    help="1-batch peek cadence between full evals")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--compile-norms", action="store_true", default=True)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--auto-resume", action="store_true",
                    help="continue from the newest sft_*.pt in --out/checkpoints")
    ap.add_argument("--data-dir", default="data/sft",
                    help="directory holding train.bin/.mask and valid_* splits")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = NanoSpeakerConfig(**{**ck["config"], "ckpt_skip": args.ckpt_skip,
                               "noise_std": 0.0})
    model = NanoSpeaker(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    model.train()
    print(f"init from {args.init_from} (pretrain step {ck.get('step')}), "
          f"window {cfg.window}/global_every {cfg.global_every}, ckpt_skip {cfg.ckpt_skip}")

    if args.compile_norms and device == "cuda":
        from ..model.ln import RMSNorm
        for mod in model.modules():
            if isinstance(mod, RMSNorm):
                mod.forward = torch.compile(mod.forward)

    optims, _ = build_optimizers(model)
    rows = PackedRows(f"{args.data_dir}/train", args.seq_len, args.micro_batch)
    tok_per_step = args.micro_batch * args.seq_len * args.accum
    steps_per_epoch = rows.n_rows // (args.micro_batch * args.accum)
    print(f"{rows.n_rows:,} rows of {args.seq_len} | {tok_per_step:,} tok/step | "
          f"{steps_per_epoch:,} steps/epoch | target {args.steps:,} steps "
          f"({args.steps / steps_per_epoch:.2f} epochs)")

    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics = open(out_dir / "metrics.jsonl", "a")
    ema, start = None, 0
    if args.auto_resume:
        # Shutdown tolerance: checkpoints carry optimizer state, and the data loader
        # is a pure function of step, so resuming replays the exact schedule.
        cks = sorted((out_dir / "checkpoints").glob("sft_*.pt"))
        if cks:
            # Load on CPU: map_location=device would park a second full copy of
            # model+optimizer state on the GPU for the life of the run (OOM'd once).
            st = torch.load(cks[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"])
            if "muon" in st:
                optims["muon"].load_state_dict(st["muon"])
                optims["adamw"].load_state_dict(st["adamw"])
            start, ema = st["step"] + 1, st.get("ema")
            del st
            print(f"resumed SFT from {cks[-1].name} at step {start}")

    for step in range(start, args.steps):
        scale = sft_lr(step, args.steps, args.warmup)
        for g in optims["muon"].param_groups:
            g["lr"] = args.lr_muon * scale
        for g in optims["adamw"].param_groups:
            g["lr"] = args.lr_adamw * scale
        for o in optims.values():
            o.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        aux_sum = torch.zeros((), device=device)
        sup_sum = torch.zeros((), device=device)
        t_data, events = 0.0, []
        for micro in range(args.accum):
            td = time.perf_counter()
            x, y = rows.batch(step, micro, args.accum, device)
            t_data += time.perf_counter() - td
            if device == "cuda":
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                e[0].record()
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                loss, aux = model(x, y)
                # A window can land entirely on masked context (tools schemas run
                # long): CE over zero targets is nan and would poison the step.
                loss = torch.nan_to_num(loss)
                if device == "cuda":
                    e[1].record()
                ((loss + aux) / args.accum).backward()
            if device == "cuda":
                e[2].record()
                events.append(e)
            loss_sum += loss.detach()
            aux_sum += aux.detach()
            sup_sum += (y != IGNORE).sum()
        t_micros = time.perf_counter() - t0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for o in optims.values():
            o.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        loss_v = loss_sum.item() / args.accum
        sup = int(sup_sum.item())
        gn = float(grad_norm)
        ema = loss_v if ema is None else 0.98 * ema + 0.02 * loss_v

        t_eval, ev_out = 0.0, {}
        if step % args.eval_every == 0 or step % args.eval_quick_every == 0:
            te = time.perf_counter()
            ev_out = evaluate(model, device, args.seq_len, data_dir=args.data_dir,
                              batches=4 if step % args.eval_every == 0 else 1)
            t_eval = time.perf_counter() - te

        t_ckpt = 0.0
        if step and step % args.save_every == 0 or step == args.steps - 1:
            tc = time.perf_counter()
            path = out_dir / "checkpoints" / f"sft_{step:06d}.pt"
            tmp = path.with_suffix(".tmp")
            torch.save(dict(step=step, model=model.state_dict(),
                            muon=optims["muon"].state_dict(),
                            adamw=optims["adamw"].state_dict(),
                            ema=ema, config=ck["config"]), tmp)
            tmp.replace(path)
            for old in sorted((out_dir / "checkpoints").glob("sft_*.pt"))[:-1]:
                old.unlink()
            t_ckpt = time.perf_counter() - tc

        row = dict(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   step=step, tokens_seen=(step + 1) * tok_per_step,
                   epoch_frac=round((step + 1) / steps_per_epoch, 4),
                   loss=round(loss_v, 4), loss_ema=round(ema, 4),
                   perplexity=round(math.exp(min(loss_v, 20)), 2),
                   aux_loss=round(aux_sum.item() / args.accum, 4),
                   sup_tokens=sup, sup_frac=round(sup / tok_per_step, 3),
                   grad_norm=round(gn, 3),
                   # <1.0 means the clip is binding -- the LR-too-high tell.
                   clip_frac=round(min(1.0, args.clip / max(gn, 1e-9)), 3),
                   lr_muon=round(args.lr_muon * scale, 6),
                   lr_adamw=round(args.lr_adamw * scale, 6),
                   tok_per_s=round(tok_per_step / dt), step_s=round(dt, 2),
                   t_data_s=round(t_data, 3), t_optim_s=round(dt - t_micros, 2),
                   t_eval_s=round(t_eval, 2), t_ckpt_s=round(t_ckpt, 2),
                   eta_h=round((args.steps - step - 1) * dt / 3600, 2))
        if events:
            row["t_fwd_s"] = round(sum(a.elapsed_time(b) for a, b, _ in events) / 1e3, 2)
            row["t_bwd_s"] = round(sum(b.elapsed_time(c) for _, b, c in events) / 1e3, 2)
        if device == "cuda":
            # Peaks are reset each step so these read per-step, not run-max.
            row["mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            row["mem_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
        row.update(moe_health(model))
        row.update(ev_out)
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if step % args.log_every == 0:
            print(row, flush=True)

    torch.save(dict(step=args.steps, model=model.state_dict(), config=ck["config"]),
               out_dir / "model_sft.pt")
    print(f"done: exported {out_dir}/model_sft.pt")


if __name__ == "__main__":
    main()
