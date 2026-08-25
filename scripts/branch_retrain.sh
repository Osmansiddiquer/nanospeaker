#!/bin/bash
# Branch nanoSpeaker off the last stable-phase checkpoint and retrain on the corrected
# next-token objective (the loader no longer pre-shifts; the model still does its own
# shift). Safe to run any number of times: it always starts from the same explicit file.
# Survives shutdown.
#
# The flags below are part of this branch's identity, not tuning knobs. Two of them are
# load-bearing in ways that look like typos:
#
#   --resume runs/nanospeaker/pre_decay_step_006950.pt, NOT --auto-resume.
#   auto-resume globs the checkpoints directory and takes the newest file, which is
#   step_008691.pt -- the two-ahead weights. Naming the branch point is the whole point.
#
#   --steps 9400, NOT 8691. Every knee in the schedule is a fraction of --steps:
#     LR/mix knee = 0.80 x 9400 = 7,520. Branching at 6,950 leaves 570 full-LR steps to
#                   realign the head on the correct objective before the decay starts,
#                   then 1,880 cosine steps (the two-ahead run got 1,738).
#     Noise knee  = 0.144 x 9400 = 1,354, far behind 6,950, so router noise is 0 for the
#                   entire branch. Intended -- the experts are long since assigned.
#     Warmup 200 is an absolute step count and is long past.
#   Setting --steps 8691 here would put the decay knee at 6,952, i.e. two full-LR steps,
#   and the head would never recover.
#
#   micro-batch 12 x 1024 x accum 12 = 147,456 tokens/step
#   2,450 steps at ~12.16 s/step = about 8.3 hours, against ~25 h to retrain from zero.
#
# The token ledger is deliberately NOT 1.23B any more. This branch reruns ~361M tokens
# of schedule (2,450 x 147,456) against fresh data on the fixed objective; the 6,950
# steps underneath it were trained on the two-ahead pairs and are being repaired, not
# counted. Do not "correct" --steps back down to hit a 1.23B budget -- that number
# belonged to the old run and forcing it here moves the decay knee, not the ledger.
cd "$(dirname "$0")"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u -m src.train.train \
  --steps 9400 --accum 12 --micro-batch 12 --ckpt-skip 2 --noise-until 0.144 \
  --resume runs/nanospeaker/pre_decay_step_006950.pt \
  --out runs/nanospeaker >> runs/nanospeaker/train.log 2>&1 &
echo "training pid $! -- tail runs/nanospeaker/train.log"
