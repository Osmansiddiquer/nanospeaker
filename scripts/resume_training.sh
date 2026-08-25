#!/bin/bash
# Restart nanoSpeaker training. Safe to run any number of times: it continues from the
# newest checkpoint if one exists and starts fresh only if none does. Survives shutdown.
#
# The flags below are part of the run's identity, not tuning knobs -- a shutdown recovery
# that omitted --micro-batch or --steps would silently change the token budget and the
# schedule. Keep them here, not in the shell history.
#
#   micro-batch 12 x 1024 x accum 12 = 147,456 tokens/step
#   8,691 steps total: 1,050 already done at 98,304/step, so the remaining 7,641 land
#   the run on 1.2299B tokens, the budget the 12,512-step plan was built around.
#   noise-until 0.144 puts the annealing knee back at step 1,251 where it was, so the
#   router noise continues its decay instead of snapping to zero on resume.
cd "$(dirname "$0")"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u -m src.train.train \
  --steps 8691 --accum 12 --micro-batch 12 --ckpt-skip 2 --noise-until 0.144 \
  --auto-resume --out runs/nanospeaker >> runs/nanospeaker/train.log 2>&1 &
echo "training pid $! -- tail runs/nanospeaker/train.log"
