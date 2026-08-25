"""
Hard arithmetic set: the crash course's second semester.

Extends build_arith past its saturated ceilings (valid_arith hit 0.32 by step 720)
and fixes its two audited warts: no single-step product ever exceeds the 12x12
times table (bigger ones decompose by place value), and every multi-digit addition
inside a multiplication gets the full carry scratchpad instead of a stated leap.

New ceilings: 5-digit +/- (cascading carries/borrows), 3x2-digit multiplication,
division with remainder, negative differences, and two-op expressions with
precedence. Same ChatML/<|think|> format, appended to the live SFT corpus.

    python -m src.data.build_arith2         # append + valid_arith_hard split
"""
import random
from pathlib import Path

import numpy as np

from .build_arith import TEMPLATES, add_steps, sub_steps

SYM = {"+": "+", "-": "-", "*": "x", "/": "/"}


def prod_steps(a: int, d: int):
    """a x d for single-digit d, decomposed by place value past the times table."""
    if a <= 12 or d == 0 or d == 1:
        return [f"{a}x{d}={a * d}"], a * d
    steps, parts = [], []
    for i, dig in enumerate(reversed(str(a))):
        dig = int(dig)
        if dig:
            v = dig * 10 ** i * d
            steps.append(f"{dig * 10 ** i}x{d}={v}")
            parts.append(v)
    tot = parts[0]
    for p in parts[1:]:
        steps.append(f"{tot}+{p}={tot + p}")
        tot += p
    return steps, tot


def mul2_steps(a: int, b: int):
    if b <= 9:
        return prod_steps(a, b)[0]
    tens, units = b // 10, b % 10
    steps = [f"{a}x{b} = {a}x{tens * 10} + {a}x{units}"]
    st, vt = prod_steps(a, tens)
    steps += st + [f"so {a}x{tens * 10}={vt * 10}"]
    su, vu = prod_steps(a, units)
    steps += su
    hi, lo = vt * 10, vu
    if lo and hi >= 1000:                      # the old wart: decompose the big add
        steps += [f"{hi}+{lo}:"] + add_steps(hi, lo)
    steps.append(f"{hi}+{lo}={a * b}")
    return steps


def conversation(rng):
    kind = rng.choice(["add", "add", "sub", "sub", "mul", "mul", "divr", "expr"])
    if kind == "add":
        la, lb = rng.choice([(3, 3), (4, 3), (4, 4), (5, 4), (5, 5)])
        a = rng.randint(10 ** (la - 1), 10 ** la - 1)
        b = rng.randint(10 ** (lb - 1), 10 ** lb - 1)
        q = rng.choice(TEMPLATES).format(a=a, op="+", b=b)
        think = " ".join(add_steps(a, b))
        ans = f"{a} + {b} = {a + b}."
    elif kind == "sub":
        la, lb = rng.choice([(3, 3), (4, 3), (4, 4), (5, 4), (5, 5)])
        a = rng.randint(10 ** (la - 1), 10 ** la - 1)
        b = rng.randint(10 ** (lb - 1), 10 ** lb - 1)
        q = rng.choice(TEMPLATES).format(a=a, op="-", b=b)
        if b > a:
            think = f"{a}<{b}, so {a}-{b} = -({b}-{a}). " + " ".join(sub_steps(b, a))
            ans = f"{a} - {b} = {a - b}."
        else:
            think = " ".join(sub_steps(a, b))
            ans = f"{a} - {b} = {a - b}."
    elif kind == "mul":
        la, lb = rng.choice([(2, 2), (3, 1), (3, 2)])
        a = rng.randint(10 ** (la - 1), 10 ** la - 1)
        b = rng.randint(10 ** (lb - 1), 10 ** lb - 1)
        shown = "x" if rng.random() < 0.5 else "*"
        q = rng.choice(TEMPLATES).format(a=a, op=shown, b=b)
        think = " ".join(mul2_steps(a, b))
        ans = f"{a} {shown} {b} = {a * b}."
    elif kind == "divr":
        b = rng.randint(3, 99)
        qt = rng.randint(3, 999)
        r = rng.randint(0, b - 1)
        a = b * qt + r
        q = rng.choice(TEMPLATES).format(a=a, op="/", b=b)
        think = (f"{b}x{qt}={b * qt}, {a}-{b * qt}={r}, "
                 f"so {a}/{b} = {qt}" + (f" remainder {r}" if r else ""))
        ans = (f"{a} / {b} = {qt}" + (f" remainder {r}" if r else "") + ".")
    else:                                       # expr: precedence teaching
        a, b, c = (rng.randint(2, 99) for _ in range(3))
        if rng.random() < 0.5:
            q = f"What is ({a} + {b}) x {c}?"
            think = f"({a}+{b})={a + b}, then {a + b}x{c}: " + " ".join(mul2_steps(a + b, c))
            ans = f"({a} + {b}) x {c} = {(a + b) * c}."
        else:
            q = f"What is {a} + {b} x {c}?"
            think = (f"multiplication first: {b}x{c}: " + " ".join(mul2_steps(b, c))
                     + f" then {a}+{b * c}={a + b * c}")
            ans = f"{a} + {b} x {c} = {a + b * c}."
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": f"<|think|>{think}<|/think|>{ans}"}]


def main() -> None:
    from .build_sft import render
    rng = random.Random(21)
    convs = [conversation(rng) for _ in range(120_000)]
    rendered = [r for r in (render(c) for c in convs) if r]
    valid, train = rendered[:400], rendered[400:]
    out = Path("data/sft")
    ids = np.concatenate([np.array(i, dtype=np.uint16) for i, _ in train])
    msk = np.concatenate([np.array(m, dtype=np.uint8) for _, m in train])
    with open(out / "train.bin", "ab") as f:
        ids.tofile(f)
    with open(out / "train.mask", "ab") as f:
        msk.tofile(f)
    np.concatenate([np.array(i, np.uint16) for i, _ in valid]).tofile(
        out / "valid_arith_hard.bin")
    np.concatenate([np.array(m, np.uint8) for _, m in valid]).tofile(
        out / "valid_arith_hard.mask")
    print(f"appended {len(train):,} hard-arithmetic conversations "
          f"({len(ids):,} tokens, mask {msk.mean():.3f}); valid_arith_hard: {len(valid)}")


if __name__ == "__main__":
    main()
