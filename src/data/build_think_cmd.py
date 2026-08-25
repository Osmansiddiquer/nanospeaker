"""
Think-on-command pool: obey explicit thinking instructions in ANY domain.

The trigger phrase is the discriminative feature: identical questions appear with a
think-command (-> <|think|> deliberation, all domains) and without (-> direct answer).
Non-arithmetic scratchpads are genuine mini-deliberations built from the facts bank's
candidates: restate, weigh, eliminate, conclude. Second-turn escalations included.
Consumed by the reasoning-CPT blend's chat slice.

    python -m src.data.build_think_cmd     # writes data/sft_t/thinkcmd pool + valid
"""
import random
from pathlib import Path

import numpy as np

from .build_arith2 import mul2_steps
from .build_sft_t import load_facts

THINK_CMDS = ["Think about it.", "Think step by step.", "Think before you answer.",
              "Take a moment to think first.", "Reason it out before answering.",
              "Work it through carefully.", "Consider it properly first.",
              "Don't rush this one.", "Step by step, please.", "Show your working."]
AGAIN = ["Think about it and answer again.", "No, reason it through this time.",
         "Work through it step by step and reconsider."]
DISTRACTOR = ["I think the answer might be {w}. What is it actually?",
              "What do you think is the answer to: {q}",
              "I've been thinking about this: {q}"]


def fact_think(q, gold, wrong, rng):
    cands = [gold] + list(rng.sample(wrong, min(2, len(wrong))))
    rng.shuffle(cands)
    lines = [f"The question is: {q}", "Options that come to mind: " + "; ".join(cands) + "."]
    for c in cands:
        if c != gold:
            lines.append(f"{c} does not fit.")
    lines.append(f"{gold} is right.")
    return " ".join(lines)


def conversation(rng, facts):
    hard = rng.random() < 0.5
    if rng.random() < 0.45:                # arithmetic domain
        if hard:
            a, b = rng.randint(13, 999), rng.randint(2, 99)
        else:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
        q, res = f"What is {a} x {b}?", a * b
        thk = " ".join(mul2_steps(a, b))
        ans = f"{a} x {b} = {res}."
        wrongish = str(res + rng.randint(1, 9))
    else:                                  # facts domain (always "easy")
        hard = False
        fq, gold, wrong = rng.choice(facts)
        q, thk = fq, fact_think(fq, gold, wrong, rng)
        ans = f"{gold[0].upper() + gold[1:]}."
        wrongish = rng.choice(wrong) if wrong else "something else"

    r = rng.random()
    # The contingency table: the command FORCES thinking; its absence leaves the
    # model's own difficulty judgment intact. Never train direct answers on hard
    # uncommanded problems -- that would teach suppression.
    if r < 0.30:                           # commanded -> think, EVEN WHEN EASY
        u = f"{q} {rng.choice(THINK_CMDS)}" if rng.random() < 0.7 \
            else f"{rng.choice(THINK_CMDS)} {q}"
        return [{"role": "user", "content": u},
                {"role": "assistant", "content": f"<|think|>{thk}<|/think|>{ans}"}]
    if r < 0.50:                           # uncommanded: difficulty decides
        body = f"<|think|>{thk}<|/think|>" if hard else ""
        return [{"role": "user", "content": q},
                {"role": "assistant", "content": f"{body}{ans}"}]
    if r < 0.70:                           # distractor: "think" non-imperatively
        if rng.random() < 0.5 and not hard:
            u = rng.choice(DISTRACTOR[1:]).format(q=q)
            return [{"role": "user", "content": u},
                    {"role": "assistant", "content": ans}]
        u = DISTRACTOR[0].format(w=wrongish) + " " + q
        body = f"<|think|>{thk}<|/think|>" if hard else ""
        return [{"role": "user", "content": u},
                {"role": "assistant", "content": f"{body}{ans}"}]
    # second-turn escalation (works regardless of difficulty)
    body = f"<|think|>{thk}<|/think|>" if hard else ""
    first = f"{body}{ans}"
    return [{"role": "user", "content": q},
            {"role": "assistant", "content": first},
            {"role": "user", "content": rng.choice(AGAIN)},
            {"role": "assistant",
             "content": f"<|think|>{thk}<|/think|>Having worked through it: {ans}"}]


def main() -> None:
    from .build_sft import render
    rng = random.Random(41)
    facts = load_facts()
    rendered = [r for r in (render(conversation(rng, facts)) for _ in range(20_000)) if r]
    valid, train = rendered[:300], rendered[300:]
    out = Path("data/sft_t")
    ids = np.concatenate([np.array(i, np.uint16) for i, _ in train])
    msk = np.concatenate([np.array(m, np.uint8) for _, m in train])
    ids.tofile(out / "thinkcmd_pool.bin")
    msk.tofile(out / "thinkcmd_pool.mask")
    np.concatenate([np.array(i, np.uint16) for i, _ in valid]).tofile(out / "valid_thinkcmd.bin")
    np.concatenate([np.array(m, np.uint8) for _, m in valid]).tofile(out / "valid_thinkcmd.mask")
    print(f"thinkcmd: {len(train):,} convs ({len(ids):,} tokens) + 300 valid")


if __name__ == "____main__" or __name__ == "__main__":
    main()
