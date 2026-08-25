#!/bin/bash
# Waits for the lr10 RL run to exit, then runs the tests that do NOT depend on the
# held-out probe (whose noise floor is +/-0.07, wider than any effect we can expect):
#   1. weight delta vs rl_base      -- did the model move at all
#   2. frozen gate, --limit 100     -- did the movement reach behaviour
#   3. per-item failure-set diff    -- same problems failing, or different ones
set -u
cd /home/nimda/repos/transformer/autoreg-tranformer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

while pgrep -f "^python -m src\.rl\.grpo" > /dev/null; do sleep 30; done
echo "=== RL exited $(date -u +%H:%M:%S) ==="

CK=$(ls -1 runs/nanospeaker_grpo_lr10/checkpoints/rl_*.pt 2>/dev/null | tail -1)
[ -z "$CK" ] && { echo "NO CHECKPOINT -- aborting"; exit 1; }
echo "=== checkpoint: $CK ==="

python - "$CK" <<'PY'
import sys, torch
a = torch.load("runs/rl_base.pt", map_location="cpu", weights_only=False)["model"]
b = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
tn=td=rn=rd=0.0
for k in a:
    if k not in b: continue
    x,y=a[k].float(),b[k].float()
    d2=((y-x).norm().item())**2; n2=(x.norm().item())**2
    tn+=n2; td+=d2
    if "router" in k: rn+=n2; rd+=d2
print(f"WEIGHT_DELTA global {(td**.5)/(tn**.5):.3e}  router {(rd**.5)/(rn**.5):.3e}")
PY

# --limit 100 is MANDATORY: the eval files hold more items than the lineage was
# scored on, and omitting it silently makes the numbers incomparable.
python -m src.eval.bench --weights "$CK" --mode chat --limit 100 \
  --tasks humaneval,arith_gen,arith_gen_hard,tools_heldout,format_compliance \
  --dump-items runs/items_lr10 || echo "GATE FAILED"

python - <<'PY'
import json, glob, os
def fails(pat):
    fs = glob.glob(pat)
    if not fs: return None
    return {r["i"] for r in json.load(open(sorted(fs)[-1])) if not r["ok"]}
for task in ("arith_gen", "humaneval"):
    base = fails(f"runs/items/*_{task}_chat.json")
    new  = fails(f"runs/items_lr10/*_{task}_chat.json")
    if base is None or new is None:
        print(f"FAILDIFF {task}: missing dump"); continue
    inter, union = base & new, base | new
    print(f"FAILDIFF {task}: base={len(base)} new={len(new)} shared={len(inter)} "
          f"jaccard={len(inter)/max(len(union),1):.3f} "
          f"fixed={sorted(base-new)[:8]} broke={sorted(new-base)[:8]}")
PY
echo "=== post-run chain complete $(date -u +%H:%M:%S) ==="
