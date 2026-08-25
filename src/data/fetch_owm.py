"""
Fetch a slice of OpenWebMath into the repo's .bin/.idx format.

The diversity component for the reasoning front-load phase (Akter et al. 2510.03264:
pretraining reasoning data rewards breadth over filtering). Web-scraped mathematical
text -- forums, notes, Q&A -- structurally unlike FineMath's textbook register.

    python -m src.data.fetch_owm --target-tokens 60000000
"""
import argparse

import numpy as np
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

VALID_TOKENS = 2_000_000
MIN_CHARS = 300


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=int, default=60_000_000)
    args = ap.parse_args()
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    docs, total = [], 0
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        ids = tok.encode(text).ids + [0]
        docs.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if len(docs) % 20_000 == 0:
            print(f"  {len(docs):,} docs, {total/1e6:.0f}M tokens", flush=True)
        if total >= args.target_tokens + VALID_TOKENS:
            break
    rng = np.random.default_rng(0)
    rng.shuffle(docs)
    cut, acc = len(docs), 0
    for i in range(len(docs) - 1, -1, -1):
        acc += len(docs[i])
        if acc >= VALID_TOKENS:
            cut = i
            break
    out = Path("data/tokens")
    for name, part in (("owm", docs[:cut]), ("valid_owm", docs[cut:])):
        stream = np.concatenate(part)
        stream.tofile(out / f"{name}.bin")
        offs = np.zeros(len(part) + 1, dtype=np.uint64)
        np.cumsum([len(d) for d in part], out=offs[1:])
        offs.tofile(out / f"{name}.idx")
        print(f"{name}: {len(part):,} docs, {len(stream):,} tokens")


if __name__ == "__main__":
    main()
