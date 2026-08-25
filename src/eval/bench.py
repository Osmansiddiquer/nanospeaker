"""
Native benchmark runner: one process, one checkpoint, every task in data/evals.

Multiple-choice tasks are scored by length-normalized logprob of each completion --
the model is asked for a loss with everything but the choice span masked to -100, so
the number that comes back is already the per-token CE of that choice alone.
Generative tasks decode greedily (temperature 0, no repetition penalty): a benchmark
that moves with the RNG cannot be compared across checkpoints.

    python -m src.eval.bench --weights runs/nanospeaker_p2b/model.pt
    python -m src.eval.bench --weights runs/nanospeaker_p2c/model.pt --mode both
    python -m src.eval.bench --weights runs/nanospeaker/model.pt --tasks arc_easy --limit 20
    python -m src.eval.bench --report --baseline p2b
"""

import argparse
import hashlib
import json
import math
import re
import resource
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .chat import CHAT_STOP_IDS, render_chat
from .decode import load, pick_device, stream

IGNORE = -100
PPL_WINDOW = 1024          # the trained context; longer would extrapolate RoPE
CODE_TIMEOUT = 5
CHAT_ONLY = {"gen_tool", "gen_format", "gen_contains"}

FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# --- small helpers -------------------------------------------------------------------

def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def amp(device: str):
    """bf16 autocast on cuda; the cpu path stays fp32 (bf16 matmuls there are slower)."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()


def read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def gen(model, tok, device, prompt: str, n: int, stop=(), stop_ids=frozenset()):
    """Greedy completion. Also reports whether decoding ended on its own -- a stop id
    or stop string -- rather than running out of budget, which is the whole of the
    gen_format metric. stream() clamps n against the context, so recompute the budget
    the same way instead of comparing against the requested count."""
    max_ctx = model.cfg.max_seq_len
    budget = min(n, max_ctx - len(tok.encode(prompt).ids[-(max_ctx - 1):]))
    with torch.no_grad(), amp(device):
        parts = list(stream(model, tok, device, prompt, n, temperature=0.0, top_k=0,
                            top_p=1.0, rep_penalty=1.0, stop=tuple(stop),
                            stop_ids=stop_ids))
    return "".join(parts), len(parts) < budget


def choice_ce(model, device, ctx_ids, choice_ids) -> float:
    """Per-token CE of `choice_ids` continuing `ctx_ids`, in one forward.

    The model scores position i against targets[i+1], so masking every target outside
    the choice span to -100 leaves exactly the choice tokens paid for, and its mean CE
    is the length-normalized score."""
    ids = (ctx_ids + choice_ids)[-model.cfg.max_seq_len:]
    y = [IGNORE] * len(ids)
    y[-len(choice_ids):] = ids[-len(choice_ids):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), amp(device):
        loss, _ = model(x, torch.tensor([y], dtype=torch.long, device=device))
    return float(loss)


def norm_number(s: str):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def last_number(text: str):
    m = NUMBER.findall(text)
    return norm_number(m[-1]) if m else None


def run_program(program: str) -> bool:
    """Execute a candidate solution under wall/CPU/address-space limits. Anything that
    escapes those is a failing sample, not a runner bug."""
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (CODE_TIMEOUT, CODE_TIMEOUT))
    try:
        p = subprocess.run([sys.executable, "-c", program], timeout=CODE_TIMEOUT,
                           capture_output=True, preexec_fn=limits)
        return p.returncode == 0
    except Exception:      # timeout, or the child died before it could be reaped
        return False


# --- task runners --------------------------------------------------------------------

def run_mc(model, tok, device, items, spec, mode, prefix_ids):
    correct = 0
    for i, it in enumerate(items):
        if mode == "chat":
            ctx = tok.encode(render_chat([{"role": "user", "content": it["context"]}])).ids
        else:
            ctx = prefix_ids + tok.encode(it["context"]).ids
        # An empty context leaves the first choice token unpredictable (nothing precedes
        # it); id 0 is the corpus document separator, so it is the honest stand-in.
        ctx = ctx or [0]
        scores = [-choice_ce(model, device, ctx, tok.encode(c).ids) for c in it["choices"]]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == it["gold"])
        tick(i + 1, len(items))
    return correct / max(len(items), 1), "acc", {"correct": correct}


def run_lastword(model, tok, device, items, spec, mode, prefix_ids):
    correct = 0
    for i, it in enumerate(items):
        if mode == "chat":
            prompt, stop_ids = render_chat([{"role": "user", "content": it["context"]}]), CHAT_STOP_IDS
        else:
            prompt, stop_ids = spec["prefix"] + it["context"], frozenset()
        text, _ = gen(model, tok, device, prompt, 8, spec.get("stop_words", ()), stop_ids)
        words = text.strip().split()
        got = words[0].strip(".,!?;:\"'()[]") if words else ""
        correct += int(got == it["answer"].strip())
        tick(i + 1, len(items))
    return correct / max(len(items), 1), "acc", {"correct": correct}


def run_code(model, tok, device, items, spec, mode, prefix_ids):
    passed = 0
    for i, it in enumerate(items):
        if mode == "chat":
            reply, _ = gen(model, tok, device,
                           render_chat([{"role": "user", "content": it["chat_prompt"]}]),
                           512, (), CHAT_STOP_IDS)
            m = FENCE.search(reply)
            body = m.group(1) if m else reply
            # Chat replies regenerate the function without the gold header's
            # imports (typing etc.) -- prepend the prompt's preamble, as standard
            # humaneval chat harnesses do. Redefinition of the signature is
            # harmless; the extracted def simply wins.
            body = it["prompt"].split("def ")[0] + "\n" + body
        else:
            completion, _ = gen(model, tok, device, it["prompt"], 512,
                                spec.get("stop_words", ()))
            body = it["prompt"] + completion
        ok = int(run_program(body + "\n" + it["test"]))
        passed += ok
        if DUMP_ITEMS:
            ITEM_LOG.append({"i": i, "ok": ok,
                             "entry": it.get("entry_point", it.get("task_id"))})
        tick(i + 1, len(items))
    return passed / max(len(items), 1), "pass1", {"passed": passed}


def run_math(model, tok, device, items, spec, mode, prefix_ids):
    correct = 0
    for i, it in enumerate(items):
        if mode == "chat":
            prompt, stop_ids = render_chat(
                [{"role": "user", "content": it["chat_prompt"]}]), CHAT_STOP_IDS
        else:
            prompt, stop_ids = spec["prefix"] + it["question"], frozenset()
        text, _ = gen(model, tok, device, prompt, 256, spec.get("stop_words", ()), stop_ids)
        got, want = last_number(text), norm_number(str(it["answer"]))
        ok = int(got is not None and want is not None and got == want)
        correct += ok
        if DUMP_ITEMS:
            ITEM_LOG.append({"i": i, "ok": ok, "want": want, "got": got,
                             "text": text[-200:]})
        tick(i + 1, len(items))
    return correct / max(len(items), 1), "acc", {"correct": correct}


def run_tool(model, tok, device, items, spec, mode, prefix_ids):
    correct = 0
    for i, it in enumerate(items):
        reply, _ = gen(model, tok, device, render_chat(it["messages"]), 300, (), CHAT_STOP_IDS)
        m = TOOL_CALL.search(reply)
        if m:
            try:
                call = json.loads(m.group(1).strip())
                args = call.get("arguments", call.get("parameters", {}))
                if isinstance(args, str):        # some models emit the args re-encoded
                    args = json.loads(args)
                correct += int(call.get("name") == it["gold_call"]["name"]
                               and set(args) == set(it["gold_call"]["arg_keys"]))
            except (ValueError, AttributeError, TypeError):
                pass
        tick(i + 1, len(items))
    return correct / max(len(items), 1), "acc", {"correct": correct}


def run_format(model, tok, device, items, spec, mode, prefix_ids):
    correct = 0
    for i, it in enumerate(items):
        text, stopped = gen(model, tok, device, render_chat(it["messages"]), 300,
                            (), CHAT_STOP_IDS)
        correct += int(stopped and "<|im_start|>" not in text)
        tick(i + 1, len(items))
    return correct / max(len(items), 1), "acc", {"correct": correct}


def run_contains(model, tok, device, items, spec, mode, prefix_ids):
    correct = 0
    for i, it in enumerate(items):
        text, _ = gen(model, tok, device, render_chat(it["messages"]), 300, (), CHAT_STOP_IDS)
        low = text.lower()
        correct += int(all(any(s.lower() in low for s in group)
                           for group in it["must_contain"]))
        tick(i + 1, len(items))
    return correct / max(len(items), 1), "acc", {"correct": correct}


# Per-item verdicts, populated only when --dump-items is set. Aggregate scores have
# been identical (arith_gen 0.74 exactly) across four differently-trained checkpoints;
# either training does nothing or the same items fail structurally. Only the identity
# of the failures can tell those apart.
ITEM_LOG = []
DUMP_ITEMS = False

RUNNERS = {"mc": run_mc, "lastword": run_lastword, "gen_code": run_code,
           "gen_math": run_math, "gen_tool": run_tool, "gen_format": run_format,
           "gen_contains": run_contains}


def run_ppl(model, device, path: Path, meta, limit: int):
    """Mean CE over consecutive full windows from the start of the stream -- fixed
    windows, no stride, so the number is reproducible across checkpoints. Window
    size comes from the manifest when set (the ppl_long_* probes use 8192)."""
    win = meta.get("window", PPL_WINDOW)
    ids = np.memmap(path, dtype=np.uint16, mode="r")
    n_win = min(len(ids) // win, limit)
    total, counted = 0.0, 0
    for i in range(n_win):
        w = np.asarray(ids[i * win:(i + 1) * win], dtype=np.int64)
        x = torch.from_numpy(w)[None].to(device)
        with torch.no_grad(), amp(device):
            loss, _ = model(x, x)
        total += float(loss) * (win - 1)
        counted += win - 1
        tick(i + 1, n_win)
    ce = total / max(counted, 1)
    # tokens/bytes comes from the manifest (the whole file), not the sampled windows:
    # bits-per-byte is a property of the corpus encoding, not of how much we scored.
    bpb = (ce / math.log(2)) * meta["tokens"] / max(meta["bytes"], 1)
    return ce, "ce_nats+bpb", {"bpb": round(bpb, 4), "windows": n_win, "tokens": counted}


# --- driver --------------------------------------------------------------------------

def tick(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"\r    {done}/{total}", end="", flush=True)


def fewshot_prefix(spec) -> str:
    shots = spec.get("fewshot") or []
    return "\n\n".join(shots) + "\n\n" if shots else ""


def bench(args) -> None:
    evals = Path(args.evals)
    manifest = json.loads((evals / "manifest.json").read_text())
    tasks, ppl = manifest.get("tasks", {}), manifest.get("ppl", {})
    wanted = [t.strip() for t in args.tasks.split(",")] if args.tasks else \
             list(tasks) + list(ppl)
    unknown = [t for t in wanted if t not in tasks and t not in ppl]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")

    modes = ["base", "chat"] if args.mode == "both" else [args.mode]
    chat_tok = Path("tokenizer/tokenizer_chat.json")
    if "chat" in modes and not chat_tok.exists():
        raise SystemExit("chat mode needs tokenizer/tokenizer_chat.json")
    tok = Tokenizer.from_file(str(chat_tok if chat_tok.exists()
                                  else "tokenizer/tokenizer.json"))

    device = pick_device(args.device)
    weights = Path(args.weights)
    model, ck = load(weights, device)
    digest = sha16(weights)
    print(f"nanoSpeaker step {ck.get('step', 0):,} on {device} | {weights} ({digest})")

    global DUMP_ITEMS
    DUMP_ITEMS = bool(args.dump_items)
    if DUMP_ITEMS:
        Path(args.dump_items).mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefixes = {}
    for name in wanted:
        for mode in modes:
            spec = tasks.get(name)
            t0 = time.perf_counter()
            if spec is None:                       # a ppl stream, not a scored task
                if mode != modes[0]:
                    continue                       # mode-independent: measure once
                meta = ppl[name]
                print(f"  {name} [ppl]")
                limit = args.limit or args.ppl_tokens // PPL_WINDOW
                score, metric, extra = run_ppl(model, device, evals / meta["bin"],
                                               meta, limit)
                n = extra["windows"]
            else:
                if spec["type"] in CHAT_ONLY and mode == "base":
                    print(f"  {name} [{mode}] -- chat-only task, skipped")
                    continue
                if spec.get("from") == "sft" and mode == "base":
                    print(f"  {name} [{mode}] -- sft-era task, chat only, skipped")
                    continue
                items = read_jsonl(evals / spec["file"])
                if args.limit:
                    items = items[:args.limit]
                print(f"  {name} [{mode}] {len(items)} items")
                # The prefix is encoded once per task: it is identical for every item
                # and re-encoding a 5-shot preamble per choice dominates the run.
                spec = dict(spec, prefix=fewshot_prefix(spec))
                if name not in prefixes:
                    prefixes[name] = tok.encode(spec["prefix"]).ids if spec["prefix"] else []
                score, metric, extra = RUNNERS[spec["type"]](
                    model, tok, device, items, spec, mode, prefixes[name])
                n = len(items)
            dt = time.perf_counter() - t0
            extra["seconds"] = round(dt, 1)
            row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "checkpoint": str(weights), "ckpt_sha16": digest, "task": name,
                   "mode": mode, "n": n, "metric": metric,
                   "score": round(score, 6), "extra": extra}
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if DUMP_ITEMS and ITEM_LOG:
                Path(args.dump_items, f"{digest}_{name}_{mode}.json").write_text(
                    json.dumps(ITEM_LOG))
                ITEM_LOG.clear()
            print(f"\r    {name} [{mode}] {metric} {score:.4f}"
                  + (f" bpb {extra['bpb']}" if "bpb" in extra else "")
                  + f"  ({dt:.1f}s, n={n})")


def report(args) -> None:
    path = Path(args.out)
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    rows = read_jsonl(path)
    latest = {}
    for r in rows:                                  # last write per cell wins
        latest[(r["checkpoint"], r["task"], r["mode"])] = r
    ckpts = list(dict.fromkeys(r["checkpoint"] for r in rows))
    keys = list(dict.fromkeys((r["task"], r["mode"]) for r in rows))

    base = next((c for c in ckpts if args.baseline and args.baseline in c), None)
    if args.baseline and base is None:
        print(f"[baseline '{args.baseline}' matched no checkpoint]")
    labels = ["/".join(Path(c).parts[-2:]) for c in ckpts]
    head = ["task"] + labels + ([f"delta vs {labels[ckpts.index(base)]}"] if base else [])

    table = []
    for task, mode in keys:
        cells = []
        for c in ckpts:
            r = latest.get((c, task, mode))
            cells.append(f"{r['score']:.4f}" if r else "-")
        if base:
            # Delta is read against the newest non-baseline column -- the checkpoint
            # you are usually asking about when you name an older one as the baseline.
            b = latest.get((base, task, mode))
            cur = next((latest[(c, task, mode)] for c in reversed(ckpts)
                        if (c, task, mode) in latest and c != base), None)
            cells.append(f"{cur['score'] - b['score']:+.4f}" if cur and b else "-")
        table.append([f"{task} [{mode}]"] + cells)

    w = [max(len(head[i]), *(len(r[i]) for r in table)) if table else len(head[i])
         for i in range(len(head))]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(head))
    print(line)
    print("-" * len(line))
    for r in table:
        print("  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/nanospeaker/model.pt")
    ap.add_argument("--tasks", default=None, help="comma-separated; default all")
    ap.add_argument("--mode", default="base", choices=["base", "chat", "both"])
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="items per task (smoke runs)")
    ap.add_argument("--evals", default="data/evals")
    ap.add_argument("--ppl-tokens", type=int, default=500_000)
    ap.add_argument("--out", default="runs/bench.jsonl")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--baseline", default=None, help="substring of a checkpoint column")
    ap.add_argument("--dump-items", default=None,
                    help="write per-item pass/fail verdicts to this dir (one json per "
                         "task). Aggregate scores repeat exactly across differently "
                         "trained checkpoints; only the failure IDENTITY separates "
                         "'training did nothing' from 'these items are unwinnable'.")
    args = ap.parse_args()
    report(args) if args.report else bench(args)


if __name__ == "__main__":
    main()
