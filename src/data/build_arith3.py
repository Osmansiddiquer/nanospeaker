"""
Arithmetic set 3: effort control and error correction.

Teaches two behaviors the corpus lacks: think HARDER when asked (effort-conditioned
scratchpad depth) and RETHINK when told wrong. Correction data leans on verification
framing -- the USER brings the wrong claim -- because training the assistant to err
first teaches the error ritual (SCoRe's warning); true self-correction turns are the
smaller share. Wrong answers are realistic perturbations (dropped carry, one bad
product), never random noise. Appended as boundary-packed rows.

    python -m src.data.build_arith3        # append rows + valid_correction split
"""
import random
from pathlib import Path

import numpy as np

from .build_arith2 import mul2_steps
from .build_arith import add_steps

ROW = 4096
EFFORT = ["Think carefully step by step.", "Show all your working.",
          "Think hard about this one.", "Work it out step by step."]
QUICK = ["Answer quickly.", "Just the answer please.", "No need to explain."]
WRONG_FLAG = ["That's wrong. Try again.", "That is not correct, check your work.",
              "Wrong answer. Redo it carefully.", "No -- rethink that."]


def problem(rng):
    a = rng.randint(13, 999)
    b = rng.randint(2, 99)
    return a, b, a * b


def plausible_wrong(rng, a, b):
    res = a * b
    kind = rng.random()
    if kind < 0.4:
        return res + rng.choice([10, 100, -10, -100])     # dropped/extra carry
    if kind < 0.7:
        return res + rng.choice([-1, 1]) * rng.randint(1, 9)
    return (a + rng.choice([-1, 1])) * b                  # one bad product


def think(a, b):
    return "<|think|>" + " ".join(mul2_steps(a, b)) + "<|/think|>"


def conversation(rng):
    a, b, res = problem(rng)
    kind = rng.random()
    if kind < 0.40:      # user brings a wrong claim; model verifies and corrects
        wrong = plausible_wrong(rng, a, b)
        q = rng.choice([f"I think {a} x {b} = {wrong}. Am I right?",
                        f"Is it true that {a} x {b} equals {wrong}?",
                        f"My friend says {a} x {b} is {wrong}. Can you check?"])
        ans = (f"{think(a, b)}No -- {a} x {b} = {res}, not {wrong}.")
        return [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    if kind < 0.75:      # self-correction: brief wrong turn, flagged, full redo
        wrong = plausible_wrong(rng, a, b)
        return [
            {"role": "user", "content": f"What is {a} x {b}?"},
            {"role": "assistant", "content": f"{a} x {b} = {wrong}."},
            {"role": "user", "content": rng.choice(WRONG_FLAG)},
            {"role": "assistant",
             "content": f"{think(a, b)}You're right, I made a mistake: "
                        f"{a} x {b} = {res}."},
        ]
    # effort-conditioned: careful -> full scratchpad; quick -> bare answer
    if rng.random() < 0.6:
        q = f"What is {a} x {b}? {rng.choice(EFFORT)}"
        ans = f"{think(a, b)}{a} x {b} = {res}."
    else:
        q = f"What is {a} x {b}? {rng.choice(QUICK)}"
        ans = f"{a} x {b} = {res}."
    return [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


def pack(rendered):
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
    return np.stack(rows_i), np.stack(rows_m)


def main() -> None:
    from .build_sft import render
    rng = random.Random(31)
    rendered = [r for r in (render(conversation(rng)) for _ in range(60_000)) if r]
    valid, train = rendered[:400], rendered[400:]
    X, M = pack(train)
    assert (X[:, 0] == 2).all()
    with open("data/sft/train.bin", "ab") as f:
        X.tofile(f)
    with open("data/sft/train.mask", "ab") as f:
        M.tofile(f)
    np.concatenate([np.array(i, np.uint16) for i, _ in valid]).tofile(
        "data/sft/valid_correction.bin")
    np.concatenate([np.array(m, np.uint8) for _, m in valid]).tofile(
        "data/sft/valid_correction.mask")
    print(f"appended {len(train):,} correction/effort convs as {len(X):,} aligned rows "
          f"({X.size:,} tokens); valid_correction: {len(valid)}")


if __name__ == "__main__":
    main()
