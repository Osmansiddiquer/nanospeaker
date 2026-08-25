#!/bin/bash
# Phase 2b: extend the context to 8192, and carry the only final anneal of phase 2.
#
# 2a is stopped at its knee and never decays -- this stage's cosine is the one that
# closes phase 2, so its decay mix is the whole diet the model finishes on.
#
# --min-doc-len 8192 on web and python is the point of the stage. Naive 8k windows over
# a shuffled stream of ~450-token documents hold about nine unrelated documents, and
# nothing in them rewards attending past the nearest boundary: you train position
# mechanics, not long-range dependency (Fu et al. 2024). Measured supply at that
# threshold -- web 18,870 docs / 266.9M tokens, python 4,423 / 78.9M -- against a 105M
# draw, so no repetition.
#
# The other three sources are deliberately left on flat sampling. They have no long
# documents to upsample: OpenCodeInstruct's longest is 2,006 tokens, the Q/A set was
# capped at 6,000 characters when fetched, and FineMath's p90 is 1,590. Forcing a
# threshold on them would only trip the loader's fallback and put them back on the flat
# stream anyway. They are in the decay mix so the model does not forget them while its
# context quadruples, not to teach long range.
#
# micro-batch 1 is a hard ceiling: mb2 OOMs in the MoE backward on 16k tokens of dispatch
# buffers. skip4 probed at 10,130 tok/s and 3.14 GiB; skip6 buys 0.5% and sits 0.1 GiB
# from the cliff. 800 x 131,072 = 105M tokens, ~13 s/step, about 3 h.
#
# rope_base 10k -> 500k is safe across the load: inv_freq/cos/sin are non-persistent
# buffers and never enter the state dict, so the checkpoint does not carry the old base.
#
# Knee at 0.50 x 800 = step 400. Run alongside:
#   ./backup_pre_decay.sh runs/nanospeaker_p2b 400
# because keep-checkpoints 2 deletes it 100 steps (~22 min) later.
set -u
cd "$(dirname "$0")"
mkdir -p runs/nanospeaker_p2b
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u -m src.train.train \
  --init-from runs/nanospeaker_p2/pre_decay_step_005600.pt \
  --out runs/nanospeaker_p2b \
  --steps 800 --warmup 100 --decay-start 0.50 \
  --seq-len 8192 --micro-batch 1 --accum 16 --ckpt-skip 4 \
  --window 512 --global-every 4 --rope-base 500000 \
  --stable-mix '{"web":0.60,"python":0.30,"math":0.10}' \
  --decay-mix  '{"web":0.40,"python":0.25,"qa":0.15,"math":0.12,"code_instruct":0.08}' \
  --min-doc-len '{"web":8192,"python":8192}' \
  --noise-std 0.0 --keep-checkpoints 2 \
  >> runs/nanospeaker_p2b/train.log 2>&1 &
PID=$!
sleep 8
if kill -0 "$PID" 2>/dev/null; then
  echo "training pid $PID -- tail runs/nanospeaker_p2b/train.log"
  echo "arm:  ./watch_training.py --run runs/nanospeaker_p2b"
  echo "and:  ./backup_pre_decay.sh runs/nanospeaker_p2b 400"
else
  echo "FAILED TO START"; tail -20 runs/nanospeaker_p2b/train.log; exit 1
fi
