#!/bin/bash
# Resume phase 2a after a crash or a power cut. THIS is the restart command, not
# phase2a.sh -- that one is --init-from and would start the phase over from step 0.
#
# Every architecture flag below has to be repeated verbatim, because the model is rebuilt
# from the CLI on every launch (train.py builds NanoSpeakerConfig from args before it
# loads any weights) and the checkpoint does not carry them back. Omit --window and the
# 3:1 interleave silently becomes full attention: the state dict still loads, the shapes
# still match, and the loss just looks a little disappointing forever.
#
# Safe to run any number of times: --auto-resume continues from the newest checkpoint in
# the run directory, and does nothing surprising if there isn't one.
set -u
cd "$(dirname "$0")"
mkdir -p runs/nanospeaker_p2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u -m src.train.train \
  --auto-resume \
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
  echo "resumed pid $PID -- tail runs/nanospeaker_p2/train.log"
  echo "re-arm:   ./watch_training.py --run runs/nanospeaker_p2"
  # The backup is a one-shot poller and dies with the machine. If the run passes step
  # 5,700 without it, phase 2b's branch point is gone.
  echo "and:      ./backup_pre_decay.sh runs/nanospeaker_p2 5600   (unless already saved)"
else
  echo "FAILED TO START -- see runs/nanospeaker_p2/train.log"; tail -20 runs/nanospeaker_p2/train.log
  exit 1
fi
