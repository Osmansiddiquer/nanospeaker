"""
Fetch + tokenize Simple English Wikipedia into the repo's .bin/.idx format.

The common-facts corpus for phase 2c: ~205k articles of plain-language encyclopedic
basics. Same layout as tokenize_corpus.py output -- doc-shuffled flat uint16 stream,
<|endoftext|> (id 0) after every document, uint64 cumulative offsets in .idx, valid
split carved from the shuffled tail so no document is shared.

    python -m src.data.fetch_simplewiki
"""
import numpy as np
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

VALID_TOKENS = 2_000_000
MIN_CHARS = 200                 # stub articles teach nothing


def main() -> None:
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    docs = []
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        ids = tok.encode(text).ids + [0]
        docs.append(np.array(ids, dtype=np.uint16))
    rng = np.random.default_rng(0)
    rng.shuffle(docs)
    print(f"{len(docs):,} docs, {sum(len(d) for d in docs):,} tokens")

    total, cut = 0, len(docs)
    for i in range(len(docs) - 1, -1, -1):
        total += len(docs[i])
        if total >= VALID_TOKENS:
            cut = i
            break
    out = Path("data/tokens")
    for name, part in (("simplewiki", docs[:cut]), ("valid_simplewiki", docs[cut:])):
        stream = np.concatenate(part)
        stream.tofile(out / f"{name}.bin")
        offs = np.zeros(len(part) + 1, dtype=np.uint64)
        np.cumsum([len(d) for d in part], out=offs[1:])
        offs.tofile(out / f"{name}.idx")
        print(f"{name}: {len(part):,} docs, {len(stream):,} tokens")


if __name__ == "__main__":
    main()
