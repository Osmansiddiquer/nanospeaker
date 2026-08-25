"""
Strip a training checkpoint down to the model.

A training checkpoint carries Muon momentum, AdamW moments and the argv it was launched
with -- 1.57 GB, of which the weights are less than half. None of it is needed to run the
model, and all of it is a liability to ship: the optimizer state pins the checkpoint to
one optimizer implementation, while the weights only need the config.

    python -m src.eval.export                      # newest checkpoint -> model.pt
    python -m src.eval.export --ckpt path.pt --out weights.pt
"""

import argparse
from pathlib import Path

import torch


def export(ckpt_path: Path, out_path: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload = {
        "model": ck["model"],
        "config": ck["config"],
        "step": ck["step"],
        "tokens_seen": ck.get("tokens_seen", 0),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    return {
        "step": ck["step"],
        "tokens": ck.get("tokens_seen", 0),
        "before_gb": ckpt_path.stat().st_size / 2**30,
        "after_gb": out_path.stat().st_size / 2**30,
        "tensors": len(ck["model"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="default: newest in runs/nanospeaker")
    ap.add_argument("--out", default="runs/nanospeaker/model.pt")
    args = ap.parse_args()

    path = Path(args.ckpt) if args.ckpt else \
        sorted(Path("runs/nanospeaker/checkpoints").glob("step_*.pt"))[-1]
    info = export(path, Path(args.out))
    print(f"{path} -> {args.out}")
    print(f"  step {info['step']:,}  {info['tokens']/1e9:.3f}B tokens  "
          f"{info['tensors']} tensors")
    print(f"  {info['before_gb']:.2f} GB -> {info['after_gb']:.2f} GB")


if __name__ == "__main__":
    main()
