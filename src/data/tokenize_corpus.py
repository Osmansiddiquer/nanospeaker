"""
Tokenize a source's shards into one flat uint16 memmap, with a document index.

The training loader wants two things: a contiguous stream of token ids it can slice
without decompressing anything, and the boundaries between documents so it never draws a
window straddling two unrelated files. So each source becomes exactly two artifacts --
`{name}.bin`, a flat uint16 array, and `{name}.idx`, the uint64 offset of every document
start plus a final offset marking the end.

uint16 holds ids up to 65,535, which a 32,768-vocab tokenizer cannot exceed. The tokenizer
is asserted against that ceiling before a byte is written rather than after.

Documents are shuffled before writing, not after: the order on disk is the order training
reads, and shuffling a 3 GB memmap afterwards means rewriting it.

The last `--holdout-tokens` of the stream are split into a separate `valid_{name}` pair
and never appear in the training file. Held out at the end, so the two files share no
document.

    python -m src.data.tokenize_corpus --shards 'data/raw/python/python-*.jsonl.zst' \\
        --name python --tokenizer tokenizer/tokenizer.json --out data/tokens
"""

import argparse
import glob
import json
import random
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .shards import read_shards


def encode_all(tok: Tokenizer, texts, eos_id: int, batch: int = 1000):
    """Encode in batches (the Rust side parallelizes within one call), appending EOS."""
    for i in range(0, len(texts), batch):
        for enc in tok.encode_batch(texts[i : i + batch]):
            yield enc.ids + [eos_id]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob for one source's .jsonl.zst")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--holdout-tokens", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab_size()
    if vocab > 65_535:
        raise SystemExit(f"vocab {vocab} exceeds uint16; widen the dtype or shrink it.")
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer has no <|endoftext|>")

    paths = [Path(p) for p in sorted(glob.glob(args.shards, recursive=True))]
    if not paths:
        raise SystemExit(f"no shards matched {args.shards!r}")

    print(f"{args.name}: reading {len(paths)} shards...", flush=True)
    texts = list(read_shards(paths))
    random.Random(args.seed).shuffle(texts)          # document-level shuffle, before write
    n_bytes = sum(len(t.encode()) for t in texts)
    print(f"  {len(texts):,} documents, {n_bytes/1e9:.2f} GB text", flush=True)

    t0 = time.time()
    docs = []                                        # each document's token ids
    total = 0
    for ids in encode_all(tok, texts, eos_id):
        docs.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if len(docs) % 100_000 == 0:
            print(f"  {len(docs):,} docs, {total/1e6:.1f}M tokens, "
                  f"{total/max(time.time()-t0,1e-9)/1e3:.0f}k tok/s", flush=True)
    del texts

    # Split from the tail so training and validation share no document.
    cut, held = len(docs), 0
    while cut > 0 and held < args.holdout_tokens:
        cut -= 1
        held += len(docs[cut])
    train_docs, valid_docs = docs[:cut], docs[cut:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, chunks) -> dict:
        n = sum(len(c) for c in chunks)
        bin_path, idx_path = out / f"{name}.bin", out / f"{name}.idx"
        arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(n,))
        offsets = np.zeros(len(chunks) + 1, dtype=np.uint64)
        at = 0
        for i, c in enumerate(chunks):
            arr[at : at + len(c)] = c
            at += len(c)
            offsets[i + 1] = at
        arr.flush()
        offsets.tofile(idx_path)
        print(f"  wrote {bin_path} ({n/1e6:.1f}M tokens) and {idx_path} "
              f"({len(chunks):,} docs)", flush=True)
        return {"tokens": int(n), "docs": len(chunks)}

    stats = {
        "source": args.name,
        "text_bytes": int(n_bytes),
        "vocab_size": vocab,
        "eos_id": eos_id,
        "train": write(args.name, train_docs),
        "valid": write(f"valid_{args.name}", valid_docs),
    }
    stats["bytes_per_token"] = n_bytes / max(total, 1)
    (out / f"{args.name}.manifest.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{args.name}: {total/1e6:.1f}M tokens from {n_bytes/1e9:.2f} GB "
          f"({stats['bytes_per_token']:.2f} bytes/token) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
