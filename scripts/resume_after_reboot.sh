#!/bin/bash
# One command after any reboot. Reports state and resumes the RL run if one was cut.
#   bash scripts/resume_after_reboot.sh
set -e
cd /home/nimda/repos/transformer/autoreg-tranformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

# Phases 2a/2b/2c, SFT and the CPT corridor are all COMPLETE (2026-08-21..23).
# The live stage is RL. Anything older is history -- do not resurrect it.

# --check reports what WOULD happen and launches nothing. This script starts real
# GPU work; running it "just to see" once collided a fresh RL run with a live bench
# on a card that fits one process. Inspect with --check, commit with no argument.
CHECK=0
[ "$1" = "--check" ] && CHECK=1

RUN=${RUN:-runs/nanospeaker_grpo_v6}
CKPT=$(ls -1 "$RUN"/checkpoints/rl_*.pt 2>/dev/null | tail -1)

# Match the python process only, and drop this script's own shell: a bare
# `pgrep -f src.rl.grpo` also matches any wrapper whose command line quotes the
# pattern, which reports a phantom run and refuses to resume a real outage.
RLPID=$(pgrep -f "^python -m src\.rl\.grpo" | grep -v "^$$\$" | head -1)
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1)
if [ -n "$RLPID" ]; then
  echo "RL already running (pid $RLPID) -- nothing to do."
elif [ -n "$BUSY" ] && [ "$CHECK" = 0 ]; then
  # 4 GB fits one CUDA process. Starting a second evicts nobody, it just OOMs.
  echo "GPU busy (pid $BUSY) -- refusing to start RL. Wait, or kill that job first."
elif [ -n "$CKPT" ]; then
  # Checkpoints carry config + Muon/AdamW state, so this resume is exact.
  if [ "$CHECK" = 1 ]; then
    echo "WOULD resume RL from $CKPT (100 steps) -- not launched (--check)"
  else
    echo "resuming RL from $CKPT"
    bash scripts/rl_run.sh "$CKPT" "$RUN" 100
  fi
else
  echo "no RL checkpoint under $RUN -- start fresh with:"
  echo "  bash scripts/rl_run.sh runs/rl_base.pt $RUN 100"
fi

echo
echo "gate any checkpoint with (--limit 100 is MANDATORY -- see nanobench memory):"
echo "  python -m src.eval.bench --weights <ckpt> --mode chat --limit 100 \\"
echo "    --tasks humaneval,arith_gen,arith_gen_hard,tools_heldout,format_compliance"
