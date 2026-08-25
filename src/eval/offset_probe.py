"""
Which offset does this checkpoint actually predict?

A next-token model is supposed to put position i's prediction on the token at i+1. An
off-by-one anywhere in the pipeline -- the loader handing over a pair that is already
shifted, the model shifting it again -- moves that peak without moving the loss curve
much, so training looks healthy while the model learns a different task. The 8,691-step
run that motivated this file scored 1.5% at +1 and 43.9% at +2: fluent, confident, and
predicting the token after next.

Loss cannot see this. Only an offset sweep can, so it is a permanent gate: run it before
trusting any checkpoint, and again on the first checkpoint of every retrain.

Healthy reading: +1 is by a wide margin the largest, with +2 and +3 down near the
unigram floor (a few percent). Any other ordering is a shift bug, not a training issue.

    python -m src.eval.offset_probe --ckpt runs/nanospeaker/model.pt
    python -m src.eval.offset_probe --ckpt runs/nanospeaker/checkpoints/step_000500.pt \
        --source code_instruct --batches 8
"""

import argparse
from pathlib import Path

import torch

from ..data.loader import MixtureLoader
from .try_it import load                 # handles both full checkpoints and exports

OFFSETS = (1, 2, 3)


def probe(model, loader, device: str, batches: int) -> dict:
    """Top-1 accuracy of position i's argmax against the token at i+k, averaged."""
    acc = dict.fromkeys(OFFSETS, 0.0)
    with torch.no_grad():
        for i in range(batches):
            x, _, _ = loader.batch(i, 0, device)     # x alone: the probe never uses y
            pred = model(x).argmax(-1)               # targets=None -> full logits
            for k in OFFSETS:
                acc[k] += (pred[:, :-k] == x[:, k:]).float().mean().item()
            print(f"  batch {i + 1}/{batches} done", flush=True)
    return {k: v / batches for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/nanospeaker/model.pt")
    ap.add_argument("--device", default="cpu", help="cpu unless you mean it")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--source", default="python",
                    choices=("python", "web", "code_instruct"))
    ap.add_argument("--token-dir", default="data/tokens")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=4)
    args = ap.parse_args()

    model, ck = load(Path(args.ckpt), args.device)
    loader = MixtureLoader(
        token_dir=args.token_dir, seq_len=args.seq_len, micro_batch=args.micro_batch,
        seed=1, split="valid_",
        stable_mix={args.source: 1.0}, decay_mix={args.source: 1.0},
    )
    print(f"step {ck['step']:,}  |  valid_{args.source}  |  {args.batches} x "
          f"{args.micro_batch} x {args.seq_len} tokens  |  {args.device}")

    acc = probe(model, loader, args.device, args.batches)
    print("\n  offset   top-1")
    for k in OFFSETS:
        print(f"  +{k}       {acc[k] * 100:5.1f}%")

    best = max(acc, key=acc.get)
    if best == 1:
        print(f"\nOK: peak at +1 ({acc[1] * 100:.1f}%) -- next-token alignment is correct.")
    else:
        print(f"\n*** SHIFT BUG: peak is at +{best} ({acc[best] * 100:.1f}%), not +1 "
              f"({acc[1] * 100:.1f}%). This model predicts {best} tokens ahead. ***")


if __name__ == "__main__":
    main()
