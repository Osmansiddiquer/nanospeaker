#!/usr/bin/env python3
"""
Watchdog for a nanoSpeaker run. Prints a line only when something warrants attention.

    ./watch_training.py --run runs/nanospeaker_p2

Silence is the design: a periodic report that fires whether or not anything is wrong
trains you to ignore it. This emits on death, on divergence, on router collapse, on
memory creeping toward the spill threshold, on a run that has stopped advancing, and on
nothing else -- plus an hourly heartbeat so progress is visible without having to ask.

The run directory is an argument and not a constant. It used to be a constant, and when
phase 2 started in a new directory the watchdog went on reading phase 1's frozen file:
every threshold compared a dead row against itself, the process check passed because it
greps for *any* trainer, and it would have reported healthy for twenty-two hours.

That is also why `stalled` exists below. It is the only check that survives being pointed
at the wrong file, and the only one that catches a process alive but wedged -- a hung
prefetcher looks exactly like a healthy run to `ps`.
"""
import argparse, json, subprocess, time
from pathlib import Path

POLL, HEARTBEAT = 60, 3600
STALL_POLLS = 5            # ~5 min without a new step, while the process is alive


def alive(run: Path) -> bool:
    """A trainer writing to *this* run, not any trainer anywhere on the machine."""
    out = subprocess.run(f"ps -eo args | grep '[s]rc.train.train' | grep -c -- '--out {run}'",
                         shell=True, capture_output=True, text=True).stdout.strip()
    return out.isdigit() and int(out) > 0


def last_row(metrics: Path):
    try:
        line = None
        with metrics.open(errors="replace") as fh:
            for line in fh:
                pass
        return json.loads(line)
    except Exception:
        return None


def faults(r, prev, total):
    """What is worth waking someone for. Every read is .get: a missing key is not silence."""
    out = []
    loss = r.get("loss")
    if loss is not None and loss != loss:
        out.append("LOSS IS NaN")
    if r.get("experts_unused", 0) > 8:
        out.append(f"router: {r['experts_unused']:.0f}/76 experts idle")
    if r.get("expert_load_cv", 0) > 1.5:
        out.append(f"router: load CV {r['expert_load_cv']:.2f}")
    if 0 < r.get("router_entropy", 1) < 0.6:
        out.append(f"router: entropy {r['router_entropy']:.2f}")
    # Peak-since-last-read, not the all-time ratchet: the old field never fell, so one
    # transient latched this alarm on and it fired 615 times in a single run.
    mem = r.get("mem_reserved_peak_gb", r.get("mem_reserved_gb", 0))
    if mem > 3.75:
        out.append(f"memory {mem:.2f} GiB -- spill threshold")
    if r.get("update_norm_ratio", 0) > 0.3:
        out.append(f"update/weight {r['update_norm_ratio']:.2e}")
    if prev and r.get("loss_ema") and prev.get("loss_ema"):
        # Loss climbing over a long window, not step-to-step jitter.
        if r["loss_ema"] > prev["loss_ema"] * 1.05 and r["step"] > prev["step"] + 50:
            out.append(f"loss_ema rising {prev['loss_ema']:.3f} -> {r['loss_ema']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/nanospeaker_p2", help="the run directory to watch")
    ap.add_argument("--steps", type=int, default=None, help="total steps, for the heartbeat")
    args = ap.parse_args()

    run = Path(args.run)
    metrics, log = run / "metrics.jsonl", run / "train.log"
    total = args.steps
    if total is None:                       # read it off the run's own provenance file
        try:
            total = json.loads((run / "run.json").read_text())["args"]["steps"]
        except Exception:
            total = 0

    was_alive, last_beat, anchor = alive(run), 0.0, last_row(metrics)
    last_step, stalls = (anchor or {}).get("step"), 0
    reported = set()
    print(f"watchdog armed on {run} ({total or '?'} steps) | "
          f"training {'running' if was_alive else 'NOT running'}", flush=True)

    while True:
        time.sleep(POLL)
        r, now = last_row(metrics), time.time()

        if not alive(run):
            if was_alive:
                tail = subprocess.run(f"tail -5 {log}", shell=True, capture_output=True,
                                      text=True).stdout.strip().splitlines()
                err = next((l for l in reversed(tail) if "rror" in l), "")
                ck = sorted((run / "checkpoints").glob("step_*.pt"))
                print(f"TRAINING STOPPED at step {r['step'] if r else '?'}. {err[:120]} "
                      f"-- newest checkpoint {ck[-1].name if ck else 'NONE'} in {run}; "
                      f"resume that run with --auto-resume, never with --init-from",
                      flush=True)
                was_alive = False
            continue
        was_alive = True

        if not r:
            continue

        # Alive but not advancing. Everything else here reads a row; this reads the clock.
        if r["step"] == last_step:
            stalls += 1
            if stalls == STALL_POLLS:
                print(f"STALLED: alive but still at step {r['step']:,} after "
                      f"{STALL_POLLS * POLL // 60} min", flush=True)
        else:
            last_step, stalls = r["step"], 0

        # Report a fault when it appears, not once a minute for as long as it lasts.
        # The anchor only refreshes on the heartbeat, so a single transient would
        # otherwise emit sixty identical lines an hour -- which is how an alarm stops
        # being read. Keyed on the fault's text up to its first number, so a drifting
        # value re-reports only when the kind of fault changes, and everything still
        # re-reports once an hour when the anchor moves.
        firing = {f.split(":")[0].split(" GiB")[0][:40] for f in faults(r, anchor, total)}
        for f in faults(r, anchor, total):
            if f.split(":")[0].split(" GiB")[0][:40] not in reported:
                print(f"step {r['step']}: {f}", flush=True)
        reported = firing

        if now - last_beat > HEARTBEAT:
            print(f"ok  step {r['step']:,}/{total or '?':,}  loss {r.get('loss', 0):.3f} "
                  f"ema {r.get('loss_ema', 0):.3f}  cv {r.get('expert_load_cv', 0):.2f}  "
                  f"{r.get('experts_unused', 0):.0f} idle  "
                  f"{r.get('tok_per_s', 0):,.0f} tok/s", flush=True)
            last_beat, anchor, reported = now, r, set()


if __name__ == "__main__":
    main()
