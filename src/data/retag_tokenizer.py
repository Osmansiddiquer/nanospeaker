"""
Rename four unused special tokens into the chat tags SFT needs. Ids never move.

    python -m src.data.retag_tokenizer --dry-run
    python -m src.data.retag_tokenizer --out tokenizer/tokenizer_chat.json

The vocabulary is exactly 32,768 entries and the unembedding is the embedding
transposed, so a new token cannot be added without resizing a tied matrix that carries
1.3B tokens of training. The way in is to rename tokens the model has never had a reason
to emit:

    id 1  <|pad|>          -> <|im_start|>     padding is never used; batches are dense
    id 2  <|fim_prefix|>   -> <|im_end|>       fill-in-the-middle was never trained
    id 3  <|fim_middle|>   -> <|think|>
    id 4  <|fim_suffix|>   -> <|/think|>

Their embedding rows are whatever initialization left them, which is the correct
starting point for a tag with no learned meaning -- SFT teaches it from there.

`<|endoftext|>` (id 0) is deliberately untouched: it has been the document separator for
1.34B tokens, so it is the one stopping signal the model already believes in, and SFT
should keep using it rather than train a competitor.

This edits names in two places -- `added_tokens` and `model.vocab` -- because a
tokenizer.json carries the string in both and a rename in only one silently produces a
vocabulary that cannot round-trip.
"""

import argparse
import json
from pathlib import Path

RENAMES = {
    "<|pad|>": "<|im_start|>",
    "<|fim_prefix|>": "<|im_end|>",
    "<|fim_middle|>": "<|think|>",
    "<|fim_suffix|>": "<|/think|>",
}


def retag(spec: dict, renames: dict = RENAMES) -> dict:
    """Rename in place and return the id map, raising rather than half-applying."""
    by_content = {a["content"]: a for a in spec["added_tokens"]}
    missing = [k for k in renames if k not in by_content]
    if missing:
        raise SystemExit(f"not in added_tokens: {missing}")
    clashes = [v for v in renames.values() if v in by_content]
    if clashes:
        raise SystemExit(f"target names already exist: {clashes}")

    ids = {}
    vocab = spec["model"]["vocab"]
    for old, new in renames.items():
        entry = by_content[old]
        ids[new] = entry["id"]
        entry["content"] = new
        if old in vocab:
            vocab[new] = vocab.pop(old)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="tokenizer/tokenizer_chat.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(Path(args.tokenizer).read_text())
    before = len(spec["model"]["vocab"])
    ids = retag(spec)
    assert len(spec["model"]["vocab"]) == before, "vocabulary size moved"

    for new, i in ids.items():
        print(f"  id {i:>2}  -> {new}")
    print(f"vocab {before:,} entries, unchanged")

    if args.dry_run:
        print("dry run: nothing written")
        return
    Path(args.out).write_text(json.dumps(spec, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
