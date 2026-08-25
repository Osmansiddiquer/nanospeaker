"""
Synthetic arithmetic crash course, after "Teaching Arithmetic to Small Transformers"
(Lee et al. 2023), adapted to the ChatML/<|think|> format.

Their findings, translated: the detailed SCRATCHPAD wins on sample efficiency, so the
carry-by-carry steps go inside <|think|>..<|/think|> and the final answer comes after
in normal digit order (their digit-reversal trick made answers locally computable from
carry state; the scratchpad achieves that without shipping a model that writes numbers
backwards). Sampling is BALANCED over operand-length pairs and carry counts instead of
natural frequency. Trivial single-digit problems get no think span at all -- 2+2 must
be a reflex, not a derivation. Mixed into the SFT corpus (~8-10%) rather than trained
as its own stage: the rest of the mix is the replay that prevents forgetting.

    python -m src.data.build_arith          # append to data/sft/train.bin/.mask
"""
import random
from pathlib import Path

import numpy as np

TEMPLATES = [
    "What is {a} {op} {b}?", "Compute {a} {op} {b}.", "{a} {op} {b} = ?",
    "Calculate {a} {op} {b}", "What does {a} {op} {b} equal?",
    "Solve: {a} {op} {b}", "How much is {a} {op} {b}?",
]
WORDED = [
    ("If you have {a} apples and get {b} more, how many do you have?", "+"),
    ("A box holds {a} items. You remove {b}. How many are left?", "-"),
    ("There are {a} rows with {b} seats each. How many seats in total?", "*"),
    ("{a} sweets are shared equally among {b} children. How many does each get?", "/"),
]
SYM = {"+": "+", "-": "-", "*": "x", "/": "/"}    # 'x' shown half the time for mult


def add_steps(a, b):
    da, db = str(a)[::-1], str(b)[::-1]
    steps, carry, out = [], 0, []
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        s = x + y + carry
        steps.append(f"{x}+{y}" + (f"+{carry}" if carry else "") +
                     f"={s}, write {s % 10}" + (", carry 1" if s >= 10 else ""))
        out.append(s % 10)
        carry = s // 10
    if carry:
        steps.append("carry 1 remains, write 1")
    return steps


def sub_steps(a, b):
    steps, borrow = [], 0
    da, db = str(a)[::-1], str(b)[::-1]
    for i in range(len(da)):
        x = int(da[i]) - borrow
        y = int(db[i]) if i < len(db) else 0
        if x < y:
            x += 10
            borrow = 1
            steps.append(f"{int(da[i])}-{y}: borrow, {x}-{y}={x - y}")
        else:
            borrow = 0
            steps.append(f"{x}-{y}={x - y}")
    return steps


def mul_steps(a, b):
    steps = []
    if b >= 10:
        lo, hi = b % 10, b // 10 * 10
        steps.append(f"{a}x{b} = {a}x{hi} + {a}x{lo}")
        steps.append(f"{a}x{hi} = {a * hi}")
        steps.append(f"{a}x{lo} = {a * lo}")
        steps.append(f"{a * hi} + {a * lo} = {a * b}")
    else:
        steps.append(f"{a}x{b} = {a * b}")
    return steps


def sample_pair(rng, op):
    """Balanced over digit-length pairs (and sign of carry load), not natural use."""
    if op in "+-":
        la, lb = rng.choice([(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)])
        a = rng.randint(10 ** (la - 1), 10 ** la - 1)
        b = rng.randint(10 ** (lb - 1), 10 ** lb - 1)
        if op == "-" and b > a:
            a, b = b, a
    elif op == "*":
        la, lb = rng.choice([(1, 1), (2, 1), (2, 2)])
        a = rng.randint(10 ** (la - 1), 10 ** la - 1)
        b = rng.randint(10 ** (lb - 1), 10 ** lb - 1)
    else:                                          # exact division from inverted mult
        b = rng.randint(2, 12)
        a = b * rng.randint(2, 99)
    return a, b


def conversation(rng):
    op = rng.choice("++-*/")                       # addition slightly over-weighted
    a, b = sample_pair(rng, op)
    res = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
    shown = SYM[op] if op == "*" and rng.random() < 0.5 else op
    if rng.random() < 0.15 and op in "+-*/":
        q_t, forced = rng.choice(WORDED)
        if forced == op:
            q = q_t.format(a=a, b=b)
        else:
            q = rng.choice(TEMPLATES).format(a=a, op=shown, b=b)
    else:
        q = rng.choice(TEMPLATES).format(a=a, op=shown, b=b)

    trivial = (op in "+-" and a < 10 and b < 10) or (op == "*" and a < 10 and b < 10)
    if trivial:
        ans = f"{a} {shown} {b} = {res}."
    else:
        steps = {"+": add_steps, "-": sub_steps, "*": mul_steps,
                 "/": lambda a, b: [f"{a}/{b}: {b}x{res}={a}, so {a}/{b}={res}"]}[op](a, b)
        ans = "<|think|>" + " ".join(steps) + "<|/think|>" + f"{a} {shown} {b} = {res}."
    return [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


def main() -> None:
    from .build_sft import render
    rng = random.Random(7)
    convs = [conversation(rng) for _ in range(70_000)]
    rendered = [r for r in (render(c) for c in convs) if r]
    valid, train = rendered[:400], rendered[400:]
    out = Path("data/sft")
    ids = np.concatenate([np.array(i, dtype=np.uint16) for i, _ in train])
    msk = np.concatenate([np.array(m, dtype=np.uint8) for _, m in train])
    with open(out / "train.bin", "ab") as f:
        ids.tofile(f)
    with open(out / "train.mask", "ab") as f:
        msk.tofile(f)
    np.concatenate([np.array(i, np.uint16) for i, _ in valid]).tofile(out / "valid_arith.bin")
    np.concatenate([np.array(m, np.uint8) for _, m in valid]).tofile(out / "valid_arith.mask")
    print(f"appended {len(train):,} arithmetic conversations ({len(ids):,} tokens, "
          f"mask {msk.mean():.3f}); valid_arith: {len(valid)} convs")


if __name__ == "__main__":
    main()
