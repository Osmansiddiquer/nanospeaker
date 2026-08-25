#!/bin/bash
# Heartbeat. Reports what is RUNNING; says so plainly when nothing is.
cd /home/nimda/repos/transformer/autoreg-tranformer
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
job=""
pgrep -f "^python -m src\.rl\.grpo"  >/dev/null && job="rl"
pgrep -f "^python -m src\.eval\.bench" >/dev/null && job="${job:+$job+}bench"
pgrep -f "^python -m src\.train\."   >/dev/null && job="${job:+$job+}train"
if [ -z "$job" ]; then
  echo "STATUS idle | gpu=$gpu | RL investigation closed, null result. No job running — awaiting direction on next phase."
else
  echo "STATUS $job | gpu=$gpu"
fi
