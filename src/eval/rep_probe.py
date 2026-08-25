"""
Layerwise semantic-abstraction probe.

20 pairs of prompts describing the SAME function two different ways. Hidden states
are captured after every block (mean-pooled over tokens, fp32), and the score per
layer is mean within-pair cosine minus mean across-pair cosine. A mid-depth bump
means the model forms phrasing-invariant task representations -- attractors that
exist and can therefore be steered.

    python -m src.eval.rep_probe [--weights ...] [--out rep_probe.png]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .decode import load, pick_device

PAIRS = [
    ("Write a function that reverses a string.",
     "Give me code that flips the characters of a string end to end."),
    ("Write a function that returns the sum of a list of numbers.",
     "Code to add up every element in a numeric list and return the total."),
    ("Write a function that checks whether a number is prime.",
     "Give me code that tests if an integer has no divisors besides 1 and itself."),
    ("Write a function to sort a list in descending order.",
     "Code that orders a list from largest value to smallest."),
    ("Write a function that counts the vowels in a string.",
     "Give me code returning how many of a, e, i, o, u appear in some text."),
    ("Write a function that returns the nth Fibonacci number.",
     "Code computing the nth term of the sequence where each term is the sum of the previous two."),
    ("Write a function to check if a word is a palindrome.",
     "Give me code that tests whether a string reads the same forwards and backwards."),
    ("Write a function that finds the largest element of a list.",
     "Code that scans a list and returns its maximum value."),
    ("Write a function computing the factorial of n.",
     "Give me code that multiplies together all integers from 1 up to n."),
    ("Write a function that removes duplicates from a list.",
     "Code that keeps only the first occurrence of each element in a list."),
    ("Write a function that capitalizes every word in a sentence.",
     "Give me code that uppercases the first letter of each word in some text."),
    ("Write a function that merges two dictionaries.",
     "Code combining the key-value pairs of two dicts into one."),
    ("Write a function that reads a file and returns its lines.",
     "Give me code that opens a text file and gives back a list of its lines."),
    ("Write a function that squares every number in a list.",
     "Code that multiplies each list element by itself and returns the results."),
    ("Write a function that keeps only the even numbers from a list.",
     "Give me code filtering a list down to elements divisible by two."),
    ("Write a function that returns the length of a string without using len.",
     "Code that counts the characters in a string manually."),
    ("Write a function that counts how often each character occurs in a string.",
     "Give me code building a frequency table of the letters in some text."),
    ("Write a function computing the greatest common divisor of two integers.",
     "Code that finds the largest number dividing both of two given integers."),
    ("Write a function converting Celsius to Fahrenheit.",
     "Give me code that turns a temperature in degrees C into degrees F."),
    ("Write a function that flattens a nested list.",
     "Code that turns a list of lists into a single flat list of items."),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/nanospeaker_sft/model_sft.pt")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--out", default="runs/rep_probe.png")
    args = ap.parse_args()

    device = args.device or "cpu"          # GPU is usually training; CPU is fine here
    model, _ = load(Path(args.weights), device)
    tok = Tokenizer.from_file("tokenizer/tokenizer_chat.json")

    acts = {}                              # layer -> list of pooled vectors
    hooks = []
    for li, block in enumerate(model.blocks):
        def hput(mod, inp, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            acts.setdefault(li, []).append(h[0].float().mean(0).detach())
        hooks.append(block.register_forward_hook(hput))

    prompts = [p for pair in PAIRS for p in pair]
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p).ids
            model(torch.tensor([ids], device=device))
    for h in hooks:
        h.remove()

    n_layers = len(model.blocks)
    scores = []
    for li in range(n_layers):
        V = torch.stack(acts[li])
        V = V / V.norm(dim=1, keepdim=True)
        S = V @ V.T
        within = torch.stack([S[2 * i, 2 * i + 1] for i in range(len(PAIRS))]).mean()
        mask = torch.ones_like(S, dtype=torch.bool)
        for i in range(len(PAIRS)):
            mask[2 * i, 2 * i + 1] = mask[2 * i + 1, 2 * i] = False
        mask.fill_diagonal_(False)
        across = S[mask].mean()
        scores.append((float(within), float(across)))
        print(f"layer {li:2d}: within {within:.4f}  across {across:.4f}  "
              f"delta {within - across:+.4f}")

    deltas = [w - a for w, a in scores]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(n_layers), deltas, marker="o")
        ax.set_xlabel("layer")
        ax.set_ylabel("within-pair minus across-pair cosine")
        ax.set_title(f"Semantic abstraction by depth — {Path(args.weights).parent.name}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=140)
        print(f"plot: {args.out}")
    except ImportError:
        print("matplotlib unavailable; deltas:", [round(d, 4) for d in deltas])


if __name__ == "__main__":
    main()
