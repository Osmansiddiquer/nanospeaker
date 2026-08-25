#!/bin/bash
# Phase 2a: web-heavy continued pretraining with 3:1 local/global attention, 1024 ctx.
#
# Phase 1 left a competent Python model with starved English -- 254M web tokens against
# 923M of Python -- and an instruct register annealed so hard (45% of decay, ending at
# perplexity 1.30) that the base distribution drifts into problem-statement scaffolding.
# This phase inverts the mix and drops instruct to a reminder rather than a diet.
#
# Flags that are load-bearing:
#
#   --init-from, NOT --resume. This is a new phase, not a continuation: weights and
#   optimizer moments carry over, the schedule restarts at step 0 so --warmup can
#   actually fire. On a --resume from step 9,400, warmup is dead code (it only applies
#   while step < warmup) and the run would begin at whatever point of the old cosine
#   step 9,400 happened to be.
#
#   --window 512 --global-every 4. Layers 0/4/8/12/16 attend globally, the other fifteen
#   over 512 tokens. Measured free at this context (11.16 s/step against 11.29 for full
#   attention) because attention is only ~0.8 s of the step -- the point is inference:
#   a 128k KV cache costs ~0.5 GiB with 3:1 instead of ~2 GiB dense.
#
#   --noise-std 0.0. Router exploration noise is for a router that has no signal yet.
#   This one has 1.34B tokens of it, entropy healthy the whole way; noise now is damage.
#
#   --out runs/nanospeaker_p2. A separate directory, so phase 1's metrics, checkpoints
#   and logs stay readable and nothing interleaves under duplicate step numbers.
#
# 7,000 x 147,456 = 1.03B tokens, ~21.9 h at the probed 13,100 tok/s. Knee at 5,600.
# Run ./backup_pre_decay.sh runs/nanospeaker_p2 5600 alongside it: phase 2b branches
# from that checkpoint, and the pruner deletes it minutes after it appears.
cd "$(dirname "$0")"
# Before the redirect, not after: bash opens >> in the forked child before exec, so
# train.py's own out.mkdir never gets the chance. Without this the launcher prints a PID,
# exits 0, and nothing runs.
mkdir -p runs/nanospeaker_p2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u -m src.train.train \
  --init-from runs/nanospeaker/checkpoints/step_009400.pt \
  --out runs/nanospeaker_p2 \
  --steps 7000 --warmup 200 --decay-start 0.80 \
  --micro-batch 12 --accum 12 --seq-len 1024 --ckpt-skip 2 \
  --window 512 --global-every 4 \
  --stable-mix '{"web":0.60,"python":0.35,"code_instruct":0.05}' \
  --decay-mix  '{"web":0.45,"python":0.30,"math":0.10,"qa":0.10,"code_instruct":0.05}' \
  --noise-std 0.0 --keep-checkpoints 1 \
  >> runs/nanospeaker_p2/train.log 2>&1 &
PID=$!
sleep 8
if kill -0 "$PID" 2>/dev/null; then
  echo "training pid $PID -- tail runs/nanospeaker_p2/train.log"
  echo "now arm:  ./watch_training.py --run runs/nanospeaker_p2"
  echo "and:      ./backup_pre_decay.sh runs/nanospeaker_p2 5600"
else
  # A launcher that reports success either way is worse than no launcher: the watchdog
  # arms, sees a run that never started, and silence is its healthy signal.
  echo "FAILED TO START -- see runs/nanospeaker_p2/train.log"; tail -20 runs/nanospeaker_p2/train.log
  exit 1
fi
