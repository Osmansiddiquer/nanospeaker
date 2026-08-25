#!/bin/bash
# Next GRPO run, fully instrumented. Usage:
#
# Config earned 2026-08-25. lr-muon 3e-4 was a NO-OP (4.2e-06 weight change/step, the
# model stayed frozen for 40 steps). 3e-3 trains it. Routers are held at 3e-5 because
# they live in AdamW, not Muon: at 3e-4 they moved 14x further, got stuck in a
# degenerate configuration, and the run died at step 16. Same trunk trajectory
# (2.235e-03 vs 2.239e-03 at step 15), opposite outcomes.
#   bash scripts/rl_run.sh <weights> <out-dir> [steps]
# Defaults to resuming the accumulation lineage for 100 steps.
set -e
cd /home/nimda/repos/transformer/autoreg-tranformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1          # parallel compile workers deadlock here

W=${1:-runs/rl_base.pt}
OUT=${2:-runs/nanospeaker_grpo_v6}
STEPS=${3:-100}

nohup python -m src.rl.grpo \
  --weights "$W" --out "$OUT" \
  --domain code --steps "$STEPS" \
  --groups-per-step 12 --k 8 --max-new 320 \
  --lr-muon 0.003 --lr-adamw 0.00003 --lr-router 0.00003 \
  --aux-coef 0.05 \
  --probe-every 10 --probe-n 24 --probe-k 4 \
  --save-every 10 --hist-every 10 \
  >> "runs/$(basename "$OUT").log" 2>&1 &
echo "launched $(basename "$OUT") pid $! from $W for $STEPS steps"
