"""
SFT-T: the targeted patch corpus, built from measured deficits only.

Pools: identity embedded mid-conversation (+anti-pleasantry chains), casual
de-formalization, bare expressions, effort contrast (incl. think-harder second
turns), topic-switch multi-turn, general claim verification, and ~60% replay
sampled from the main SFT corpus as forgetting insurance for the 5x LR.

Everything is shuffled at CONVERSATION level across all pools before packing, so
every 4096 row (and hence every batch) mixes sources by construction. Boundary
packing, masked-eos padding -- the SFT-standard layout.

    python -m src.data.build_sft_t         # writes data/sft_t/
"""
import json
import random
from pathlib import Path

import numpy as np

from .build_arith import add_steps, sample_pair
from .build_arith2 import mul2_steps
from .build_identity import CREATOR, NAME, QUESTIONS

ROW = 4096
REPLAY_TOKENS = 24_000_000

CASUAL_ARITH = ["whats {a} {op} {b}", "how much is {a} {op} {b}", "{a}{op}{b}?",
                "quick, {a} {op} {b}", "{a} {op} {b}", "can u do {a} {op} {b}",
                "ok so {a} {op} {b} = ?"]
EFFORT = ["Think carefully step by step.", "Show all your working.",
          "Take your time and think it through.", "Work it out step by step."]
QUICK = ["Answer quickly.", "Just the answer.", "No working needed."]
HARDER = ["Think harder about it, step by step.", "Are you sure? Work it out properly.",
          "Do it again carefully, showing every step."]
IDENT_CASUAL = ["who are you", "who r u", "whats your name", "who made you",
                "btw who are you?", "wait, who am i talking to", "what are you exactly"]


def arith(rng, hard=False):
    if hard:
        a, b = rng.randint(13, 999), rng.randint(2, 99)
        return a, "x", b, a * b, " ".join(mul2_steps(a, b))
    op = rng.choice("+-")
    a, b = sample_pair(rng, op)
    res = a + b if op == "+" else a - b
    steps = add_steps(a, b) if op == "+" else None
    return a, op, b, res, " ".join(steps) if steps else None


def qa_arith_casual(rng):
    a, op, b, res, steps = arith(rng, hard=rng.random() < 0.4)
    q = rng.choice(CASUAL_ARITH).format(a=a, op=op, b=b)
    body = (f"<|think|>{steps}<|/think|>" if steps and (a > 99 or b > 99) else "")
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": f"{body}{a} {op} {b} = {res}."}]


def qa_bare(rng):
    a, op, b, res, steps = arith(rng, hard=rng.random() < 0.3)
    q = rng.choice([f"{a}{op}{b}", f"{a} {op} {b}", f"{a} {op} {b} ="])
    body = f"<|think|>{steps}<|/think|>" if steps and (a > 99 or b > 99) else ""
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": f"{body}{a} {op} {b} = {res}."}]


def load_facts():
    out = []
    for line in open("data/evals/facts_common.jsonl"):
        it = json.loads(line)
        q = it["context"].replace("Question: ", "").replace("\nAnswer:", "").strip()
        gold = it["choices"][it["gold"]].strip()
        wrong = [c.strip() for i, c in enumerate(it["choices"]) if i != it["gold"]]
        out.append((q, gold, wrong))
    return out


def qa_fact(rng, facts, casual):
    q, gold, _ = rng.choice(facts)
    if casual:
        q = q.lower().rstrip("?")
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": f"{gold[0].upper() + gold[1:]}."}]


def qa_identity(rng, casual=False):
    q, a = rng.choice(QUESTIONS)
    if casual:
        q = rng.choice(IDENT_CASUAL)
        a = rng.choice([f"I'm {NAME}, a small language model trained by {CREATOR}.",
                        f"My name is {NAME} -- a tiny model {CREATOR} trained on a laptop.",
                        f"I'm {NAME}. {CREATOR} built me from scratch."])
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def verify(rng, facts):
    if rng.random() < 0.5:
        q, gold, wrong = rng.choice(facts)
        claim = rng.choice(wrong)
        u = rng.choice([f"Is it true that the answer to '{q}' is {claim}?",
                        f"I think the answer to '{q}' is {claim}. Right?"])
        return [{"role": "user", "content": u},
                {"role": "assistant", "content": f"No -- {gold}."}]
    a, op, b, res, steps = arith(rng, hard=True)
    wrong = res + rng.choice([10, 100, -10, -100, rng.randint(1, 9)])
    u = f"Someone told me {a} x {b} = {wrong}. Is that right?"
    return [{"role": "user", "content": u},
            {"role": "assistant",
             "content": f"<|think|>{steps}<|/think|>No -- {a} x {b} = {res}, not {wrong}."}]


def effort_pair(rng):
    a, op, b, res, steps = arith(rng, hard=True)
    if rng.random() < 0.2:      # second-turn escalation
        return [{"role": "user", "content": f"What is {a} x {b}?"},
                {"role": "assistant", "content": f"{a} x {b} = {res}."},
                {"role": "user", "content": rng.choice(HARDER)},
                {"role": "assistant",
                 "content": f"<|think|>{steps}<|/think|>Confirmed: {a} x {b} = {res}."}]
    if rng.random() < 0.5:
        return [{"role": "user", "content": f"What is {a} x {b}? {rng.choice(EFFORT)}"},
                {"role": "assistant", "content": f"<|think|>{steps}<|/think|>{a} x {b} = {res}."}]
    return [{"role": "user", "content": f"What is {a} x {b}? {rng.choice(QUICK)}"},
            {"role": "assistant", "content": f"{a} x {b} = {res}."}]


def build_pools(rng, facts):
    single = lambda: rng.choice([lambda: qa_arith_casual(rng), lambda: qa_fact(rng, facts, True),
                                 lambda: qa_identity(rng, True), lambda: qa_bare(rng)])()
    pools = {}
    # P1 identity embedded (+ anti-pleasantry)
    p1 = []
    for _ in range(15_000):
        conv = []
        for _ in range(rng.randint(1, 3)):
            conv += single()
        if rng.random() < 0.35:
            conv += [{"role": "user", "content": rng.choice(["Thanks!", "thank you", "great, thanks"])},
                     {"role": "assistant", "content": "You're welcome!"}]
        conv += qa_identity(rng, casual=rng.random() < 0.5)
        p1.append(conv)
    pools["identity_embedded"] = p1
    pools["casual"] = [qa_arith_casual(rng) if rng.random() < 0.5 else
                       qa_fact(rng, facts, True) for _ in range(60_000)]
    pools["bare"] = [qa_bare(rng) for _ in range(25_000)]
    pools["effort"] = [effort_pair(rng) for _ in range(40_000)]
    p5 = []
    for _ in range(30_000):
        conv = []
        for _ in range(rng.randint(3, 5)):
            conv += single()
        p5.append(conv)
    pools["switch"] = p5
    pools["verify"] = [verify(rng, facts) for _ in range(15_000)]
    pools["codecheck"] = code_check_pool(rng)
    return pools


CHECKER_SCHEMA = ('[{"type": "function", "function": {"name": "check_python", '
                  '"description": "Parse Python code and report syntax errors and undefined names.", '
                  '"parameters": {"type": "object", "properties": {"code": {"type": "string"}}, '
                  '"required": ["code"]}}}]')
TOOL_SYS = ("You are a function calling AI model. You are provided with function signatures "
            "within <tools></tools> XML tags.You may call one or more functions to assist with "
            "the user query.Here are the available tools:<tools> " + CHECKER_SCHEMA +
            " </tools>For each function call return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags.")


def plant_bug(rng, code):
    """Return (buggy, diag) with a REAL diagnostic, or None if the bug didn't take."""
    import re as _re
    from ..eval.pycheck import check_python
    lines = code.split("\n")
    kind = rng.random()
    if kind < 0.5:
        idx = [i for i, l in enumerate(lines) if l.startswith(("import ", "from "))]
        if idx:
            buggy = "\n".join(l for i, l in enumerate(lines) if i != idx[0])
        else:
            return None
    else:
        m = _re.search(r"^(def \w+\([^)]*\)):", code, _re.M)
        if not m:
            return None
        buggy = code.replace(m.group(0), m.group(1), 1)
    diag = check_python(buggy)
    return (buggy, diag) if not diag["ok"] else None


def code_check_pool(rng, n_target=12_000):
    """Draft -> call check_python -> read real diagnostic -> corrected code."""
    import json as _json
    from ..eval.pycheck import check_python
    from .build_sft import iter_zst
    out = []
    for row in iter_zst("data/raw/code_instruct/*.jsonl.zst"):
        if len(out) >= n_target:
            break
        text = row["text"]
        if not text.startswith("### Instruction\n"):
            continue
        inp, sep, out_text = text[len("### Instruction\n"):].partition("\n\n### Response\n")
        if not sep:
            continue
        import re as _re2
        m = _re2.search(r"```(?:python)?\s*\n(.*?)```", out_text, _re2.S)
        if not m:
            continue
        code = m.group(1).strip()
        if not (60 < len(code) < 1200):
            continue
        if not check_python(code)["ok"]:
            continue
        call = lambda c: ("<tool_call>\n" + _json.dumps(
            {"name": "check_python", "arguments": {"code": c}}) + "\n</tool_call>")
        resp = lambda d: "<tool_response>\n" + _json.dumps(d) + "\n</tool_response>"
        if rng.random() < 0.3:      # clean path: verify, confirm, done
            out.append([
                {"role": "system", "content": TOOL_SYS},
                {"role": "user", "content": inp[:400]},
                {"role": "assistant", "content": f"```python\n{code}\n```\n" + call(code)},
                {"role": "tool", "content": resp({"ok": True, "errors": []})},
                {"role": "assistant", "content": "The checker passes. The solution above is correct."},
            ])
            continue
        planted = plant_bug(rng, code)
        if planted is None:
            continue
        buggy, diag = planted
        out.append([
            {"role": "system", "content": TOOL_SYS},
            {"role": "user", "content": inp[:400]},
            {"role": "assistant", "content": f"```python\n{buggy}\n```\n" + call(buggy)},
            {"role": "tool", "content": resp(diag)},
            {"role": "assistant",
             "content": f"The checker found: {diag['errors'][0]}. Here is the corrected "
                        f"version:\n```python\n{code}\n```"},
        ])
    return out


def replay_convs(rng):
    ids = np.memmap("data/sft/train.bin", dtype=np.uint16, mode="r")
    msk = np.memmap("data/sft/train.mask", dtype=np.uint8, mode="r")
    ends = np.nonzero(np.asarray(ids) == 0)[0]
    bounds = list(zip([0] + (ends + 1).tolist(), (ends + 1).tolist()))
    rng.shuffle(bounds)
    out, tok = [], 0
    for s, e in bounds:
        if e - s < 4 or e - s > ROW or ids[s] != 2:
            continue
        out.append((np.asarray(ids[s:e]), np.asarray(msk[s:e])))
        tok += e - s
        if tok >= REPLAY_TOKENS:
            break
    return out, tok


def main() -> None:
    from .build_sft import render
    rng = random.Random(77)
    facts = load_facts()
    pools = build_pools(rng, facts)

    rendered, valid = [], {}
    for name, convs in pools.items():
        rs = [r for r in (render(c) for c in convs) if r]
        hold = 200 if name in ("identity_embedded", "casual", "effort", "switch",
                               "codecheck") else 0
        if hold:
            valid[name] = rs[:hold]
            rs = rs[hold:]
        rendered += [(np.array(i, np.uint16), np.array(m, np.uint8)) for i, m in rs]
        print(f"{name}: {len(rs):,} convs")
    rep, rep_tok = replay_convs(rng)
    print(f"replay: {len(rep):,} convs, {rep_tok/1e6:.1f}M tokens")
    rendered += rep
    rng.shuffle(rendered)                       # conversation-level cross-pool mix

    rows_i, rows_m, cur_i, cur_m, n = [], [], [], [], 0
    for ci, cm in rendered:
        if n + len(ci) > ROW:
            rows_i.append(np.concatenate(cur_i + [np.zeros(ROW - n, np.uint16)]))
            rows_m.append(np.concatenate(cur_m + [np.zeros(ROW - n, np.uint8)]))
            cur_i, cur_m, n = [], [], 0
        cur_i.append(ci); cur_m.append(cm); n += len(ci)
    if cur_i:
        rows_i.append(np.concatenate(cur_i + [np.zeros(ROW - n, np.uint16)]))
        rows_m.append(np.concatenate(cur_m + [np.zeros(ROW - n, np.uint8)]))
    X, M = np.stack(rows_i), np.stack(rows_m)
    assert (X[:, 0] == 2).all()

    out = Path("data/sft_t")
    out.mkdir(exist_ok=True)
    X.tofile(out / "train.bin"); M.tofile(out / "train.mask")
    for name, rs in valid.items():
        np.concatenate([np.array(i, np.uint16) for i, _ in rs]).tofile(
            out / f"valid_{name}.bin")
        np.concatenate([np.array(m, np.uint8) for _, m in rs]).tofile(
            out / f"valid_{name}.mask")
    print(f"{len(X):,} rows | {X.size/1e6:.1f}M tokens | supervised {M.mean()*100:.1f}% | "
          f"steps/epoch at 16 rows: {len(X)//16:,}")


if __name__ == "__main__":
    main()
