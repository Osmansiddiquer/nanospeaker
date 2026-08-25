"""
Offline sampler: the engine for both the pass@k probe and ReST-EM data generation.

Unlike rollout.py (one prompt x k, on-policy, keeps logprobs), this batches ACROSS
prompts with no gradients resident, so the batch can be large and the GPU stays fed.
Scores every sample with src/rl/rewards.py, reports pass@k, and -- with --emit --
writes the winners as ChatML conversations for a rejection-sampling training pass.

    python -m src.rl.sample --domain code --n 50 --k 16          # the R-probe
    python -m src.rl.sample --domain code,arith --n 400 --k 8 --emit data/rest
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..data.build_sft import render
from ..data.build_sft_t import load_facts
from ..eval.chat import CHAT_STOP_IDS, render_chat
from ..eval.decode import load
from .problems import arith_problem, load_code_bank, tool_problem
from .rewards import answer_reward, code_reward, tool_reward


@torch.no_grad()
def sample_batch(model, tok, device, prompts, k, max_new=400, temperature=1.0):
    """prompts: list of rendered strings. Returns list[list[str]] (k per prompt).

    Rows are left-padded to a common length with eos so one batch covers prompts of
    different sizes; padding sits BEFORE the content, and the trained model always
    sees eos before a conversation anyway (the cold-start lesson), so this is in
    distribution rather than a hack.
    """
    enc = [tok.encode(p).ids for p in prompts]
    width = max(len(e) for e in enc)
    rows, offs = [], []
    for e in enc:
        pad = width - len(e)
        rows.append([0] * pad + e)
        offs.append(pad)
    x = torch.tensor(rows, device=device).repeat_interleave(k, 0)
    B = x.shape[0]
    # Windowed layers ring at window + max_chunk - 1 and the capacity is min'd
    # against max_seq_len, so the prefill width must be covered explicitly.
    win = max((w for w in model.cfg.layer_windows() if w), default=0)
    total = max(width + max_new + 1, win + width)
    caches = model.caches(total, max_chunk=width)
    with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
        logits = model(x, caches=caches)[:, -1]
    out = torch.zeros(B, max_new, dtype=torch.long, device=device)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    lens = torch.zeros(B, dtype=torch.long, device=device)
    stop = torch.tensor(sorted(CHAT_STOP_IDS), device=device)
    for step in range(max_new):
        lg = logits.float() / max(temperature, 1e-6)
        if step < 4:
            lg[:, list(CHAT_STOP_IDS)] = -float("inf")
        opened = (out[:, :step] == 4).any(-1)
        lg[~opened, 1] = -float("inf")
        nxt = torch.multinomial(F.softmax(lg, -1), 1).squeeze(-1)
        hit = (nxt[:, None] == stop[None, :]).any(-1)
        rec = alive & ~hit
        out[rec, step] = nxt[rec]
        lens[rec] += 1
        alive = alive & ~hit
        if not alive.any():
            break
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            logits = model(torch.where(alive, nxt, torch.zeros_like(nxt))[:, None],
                           caches=caches)[:, -1]
    texts = [tok.decode(out[i, :int(lens[i])].tolist(), skip_special_tokens=False)
             for i in range(B)]
    return [texts[i * k:(i + 1) * k] for i in range(len(prompts))]


def taxonomy(text, prob):
    if "```" not in text and prob["kind"] == "code":
        return "no_fence"
    body = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    src = body.group(1) if body else text
    try:
        compile(src, "<s>", "exec")
    except SyntaxError:
        return "syntax_error"
    return "wrong_answer"


def score(prob, text):
    if prob["kind"] == "code":
        return code_reward(text, prob["tests"])[0]
    if prob["kind"] == "tool":
        return tool_reward(text, prob["gold_call"], prob.get("tool_result"))[0] >= 1.0
    return answer_reward(text, prob["answer"])[0]


def clean_solution(text, prob):
    """Quality filter: correctness alone breeds goofy-but-right reasoning."""
    if text.count("<|think|>") > 1 or text.count("<|/think|>") > 1:
        return None
    if text.count("<|think|>") != text.count("<|/think|>"):
        return None
    if prob["kind"] == "code" and "```" not in text:
        return None
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/rl_base.pt")
    ap.add_argument("--domain", default="code")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch-prompts", type=int, default=3,
                    help="prompts per batch. Memory scales with batch_prompts*k: "
                         "6 x k16 = 96 sequences OOMs the 4 GB card (cost 7 GPU-hours "
                         "of silent failure). 3 x k16 = 48 is the tested ceiling.")
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--emit", default=None, help="write winners as ChatML to this dir")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file("tokenizer/tokenizer_chat.json")
    model, _ = load(Path(args.weights), device)
    model.eval()
    rng = random.Random(0)
    facts = load_facts()

    probs = []
    doms = args.domain.split(",")
    if "code" in doms:
        bank = load_code_bank(limit=args.n)
        probs += [dict(kind="code", **b) for b in bank[:args.n]]
    if "arith" in doms:
        tiers = [int(t) for t in args.tiers.split(",")]
        probs += [arith_problem(rng, rng.choice(tiers)) for _ in range(args.n)]
    if "tool" in doms:
        probs += [tool_problem(rng, facts) for _ in range(args.n)]

    stats = Counter()
    passk, per_domain = [], Counter()
    winners = []
    for i in range(0, len(probs), args.batch_prompts):
        chunk = probs[i:i + args.batch_prompts]
        rendered = [render_chat(p["messages"]) if "messages" in p
                    else render_chat([{"role": "user", "content": p["prompt"]}])
                    for p in chunk]
        gens = sample_batch(model, tok, device, rendered, args.k,
                            args.max_new, args.temperature)
        for p, samples in zip(chunk, gens):
            hits = [s for s in samples if score(p, s)]
            passk.append((p["kind"], len(hits) / args.k, bool(hits)))
            per_domain[p["kind"] + "_solved"] += bool(hits)
            per_domain[p["kind"] + "_total"] += 1
            if not hits and p["kind"] == "code":     # taxonomy is code-specific
                stats[taxonomy(samples[0], p)] += 1
            kept = 0
            for s in hits:
                c = clean_solution(s, p)
                if c and kept < args.max_per_problem:
                    user = p["messages"] if "messages" in p else \
                        [{"role": "user", "content": p["prompt"]}]
                    winners.append(list(user) + [{"role": "assistant", "content": c}])
                    kept += 1
        print(f"  {min(i+args.batch_prompts, len(probs))}/{len(probs)} prompts", flush=True)

    for kind in set(k for k, _, _ in passk):
        rows = [r for r in passk if r[0] == kind]
        p1 = sum(r[1] for r in rows) / len(rows)
        pk = sum(r[2] for r in rows) / len(rows)
        print(json.dumps({"domain": kind, "n": len(rows), "pass@1": round(p1, 4),
                          f"pass@{args.k}": round(pk, 4)}))
    if stats:
        print(json.dumps({"failure_taxonomy": dict(stats)}))

    if args.emit and winners:
        out = Path(args.emit)
        out.mkdir(parents=True, exist_ok=True)
        rendered = [r for r in (render(c) for c in winners) if r]
        ROW = 4096
        rows_i, rows_m, cur_i, cur_m, n = [], [], [], [], 0
        for ci, cm in rendered:
            ci, cm = np.array(ci, np.uint16), np.array(cm, np.uint8)
            if len(ci) > ROW:
                continue
            if n + len(ci) > ROW:
                rows_i.append(np.concatenate(cur_i + [np.zeros(ROW - n, np.uint16)]))
                rows_m.append(np.concatenate(cur_m + [np.zeros(ROW - n, np.uint8)]))
                cur_i, cur_m, n = [], [], 0
            cur_i.append(ci); cur_m.append(cm); n += len(ci)
        if cur_i:
            rows_i.append(np.concatenate(cur_i + [np.zeros(ROW - n, np.uint16)]))
            rows_m.append(np.concatenate(cur_m + [np.zeros(ROW - n, np.uint8)]))
        X, M = np.stack(rows_i), np.stack(rows_m)
        X.tofile(out / "winners.bin"); M.tofile(out / "winners.mask")
        print(f"emitted {len(winners):,} winning conversations -> {len(X):,} rows")


if __name__ == "__main__":
    main()
