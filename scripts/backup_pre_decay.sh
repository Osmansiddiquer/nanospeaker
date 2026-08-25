#!/bin/bash
# Copy the last stable-phase checkpoint out of the pruner's reach, before the decay knee.
#
#   ./backup_pre_decay.sh <run_dir> <step>
#   ./backup_pre_decay.sh runs/nanospeaker_p2 5600
#
# A run keeps only its newest checkpoints, so the last stable-phase state is deleted
# minutes after it is written. That file is the only branch point a run has: everything
# after it is committed to one decay mix and one cosine, and re-reaching it costs the
# whole stable phase. Phase 1 needed exactly this file to recover from the double-shift
# bug, and phase 2b branches from the phase-2a one by design.
#
# Fires once, then exits. The copy lands in the run root, which the pruner's
# checkpoints/step_*.pt glob cannot see.
set -u
cd "$(dirname "$0")"
RUN="${1:-runs/nanospeaker_p2}"
STEP="${2:-5600}"
SRC=$(printf '%s/checkpoints/step_%06d.pt' "$RUN" "$STEP")
DST=$(printf '%s/pre_decay_step_%06d.pt' "$RUN" "$STEP")

echo "$(date '+%F %T')  waiting for $SRC"
while [ ! -f "$SRC" ]; do sleep 5; done
sleep 2                                  # os.replace has landed; let the page cache settle
cp "$SRC" "$DST.tmp" && mv "$DST.tmp" "$DST"
echo "$(date '+%F %T')  backed up $SRC -> $DST ($(du -h "$DST" | cut -f1))"
