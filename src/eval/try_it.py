"""
Ask nanoSpeaker to actually do things.

Loss curves say a model is learning something; they do not say what. This runs the model
on prompts drawn from each of its three training distributions -- raw Python, the
instruction format it saw during decay, and English prose -- plus a few it has no right
to handle, because the only way to find the ceiling is to walk into it.

    python -m src.eval.try_it                       # the whole battery
    python -m src.eval.try_it --only python --temperature 0.2
    python -m src.eval.try_it --prompt "def fib(n):" --tokens 120
"""

import argparse
import time
from pathlib import Path

from tokenizers import Tokenizer

from .decode import INSTRUCT, generate, load, pick_device  # load re-exported: offset_probe uses it

# Prompt, tokens to generate. Grouped by which training distribution they lean on.
BATTERY = {
    "python": [
        ("def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n", 90),
        ("def quicksort(arr):\n", 120),
        ("import json\n\ndef load_config(path):\n", 100),
        ("class Stack:\n    def __init__(self):\n", 110),
        ("# Read a CSV and return the column means\nimport csv\n\ndef column_means(path):\n", 130),
        ("def is_palindrome(s: str) -> bool:\n", 70),
        ("with open('data.txt') as f:\n", 60),
        ("try:\n    result = risky()\n", 60),
    ],
    "instruct": [
        (INSTRUCT.format("Write a Python function that reverses a string."), 110),
        (INSTRUCT.format("What does the `enumerate` builtin do in Python?"), 90),
        (INSTRUCT.format("Write a function to check if a number is prime."), 130),
        (INSTRUCT.format("Explain the difference between a list and a tuple."), 110),
        (INSTRUCT.format("How do I read a JSON file in Python?"), 110),
    ],
    "english": [
        ("The main advantage of using a hash table is", 60),
        ("In the early days of computing,", 60),
        ("Python is a programming language that", 60),
    ],
    "reach": [                       # no right to these; run them anyway
        (INSTRUCT.format("Write a Python decorator that caches function results."), 150),
        (INSTRUCT.format("Implement binary search over a sorted list."), 150),
        ("def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n", 150),
        (INSTRUCT.format("What is recursion?"), 100),
    ],
}


def run(model, tok, device, prompt: str, n: int, args):
    t = time.perf_counter()
    text = generate(model, tok, device, prompt, n, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep_penalty)
    dt = time.perf_counter() - t
    return text, n / max(dt, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/nanospeaker/model.pt")
    ap.add_argument("--only", default=None, choices=sorted(BATTERY))
    ap.add_argument("--prompt", default=None, help="run one prompt instead of the battery")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0, help="1.0 disables")
    ap.add_argument("--rep-penalty", type=float, default=1.05,
                    help="CTRL-style; 1.0 disables")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ck = load(Path(args.weights), device)
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    print(f"nanoSpeaker step {ck['step']:,}  {ck.get('tokens_seen',0)/1e9:.3f}B tokens  "
          f"| temperature {args.temperature}, top_k {args.top_k}\n")

    if args.prompt:
        text, tps = run(model, tok, device, args.prompt, args.tokens, args)
        print(args.prompt + text)
        print(f"\n[{tps:.1f} tok/s]")
        return

    groups = [args.only] if args.only else list(BATTERY)
    for group in groups:
        print("=" * 78)
        print(f"  {group.upper()}")
        print("=" * 78)
        for prompt, n in BATTERY[group]:
            text, tps = run(model, tok, device, prompt, n, args)
            print("-" * 78)
            print(prompt, end="")
            print(f"\033[92m{text}\033[0m")
            print(f"[{tps:.1f} tok/s]")
        print()


if __name__ == "__main__":
    main()
