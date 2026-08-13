"""
Train a byte-level BPE tokenizer, shaped for a corpus that is mostly Python.

Byte-level, so there is no UNK and no encoding can be refused: every byte maps to a
character in a 256-symbol alphabet before BPE ever runs, and the decoder inverts that
exactly. Round-tripping arbitrary bytes matters more for code than for prose -- source
files carry stray encodings, and a tokenizer that mangles them silently corrupts training
data.

The split pattern is where a code tokenizer is won or lost. GPT-2's pattern attaches a
single leading space to a word and stops, so a Python file indented four spaces spends a
token per indent level per line -- on deeply nested code that is a double-digit fraction
of the whole budget. The pattern below (the GPT-4 / cl100k shape) isolates runs of
whitespace so BPE can merge them whole, and splits numbers into groups of at most three
digits so the merge table does not waste entries memorizing individual integers.

`--check-indent` is a hard gate, not a report: if a 4-space indent does not come back as
one token the script exits nonzero, because that failure is invisible in training and
costs roughly a third of the budget.

    python -m src.data.train_tokenizer \\
        --shards 'data/raw/python/python-*.jsonl.zst:400e6' \\
                 'data/raw/web/web-*.jsonl.zst:100e6' \\
        --out tokenizer/
"""

import argparse
import glob
import json
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from .fetch_streaming import INSTRUCT_TEMPLATE
from .shards import read_shards

# cl100k-shaped: contractions, letter runs, <=3-digit numbers, punctuation runs, and --
# the part that matters here -- whitespace runs isolated instead of one-space-per-word.
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

EOS = "<|endoftext|>"
SPECIALS = [
    EOS, "<|pad|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    # Stack-Edu redacts personal data to these; reserved so BPE cannot shred them.
    "<EMAIL>", "<KEY>", "<NAME>", "<PASSWORD>",
]
# The instruction delimiters are deliberately NOT special. Special tokens are matched
# against raw text before the pre-tokenizer and re-emitted without their surrounding
# whitespace, which broke round-tripping on every OpenCodeInstruct document. They open
# 310,820 documents, so BPE learns them as ordinary merges regardless -- a string that
# common needs to be frequent, not reserved.

# Indent widths that must survive as single tokens. 4 is the one that decides the run.
INDENTS = (2, 4, 8, 16)


def build() -> Tokenizer:
    """An untrained byte-level BPE with the split pattern above."""
    tok = Tokenizer(models.BPE(unk_token=None, fuse_unk=False, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
        # use_regex=False: the split above already did the work, so ByteLevel is only
        # doing the byte->unicode mapping here, not a second (conflicting) split.
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_budgeted(spec: str):
    """`glob:bytes` -- stream texts from .jsonl.zst shards until the budget is spent."""
    pattern, _, budget = spec.partition(":")
    budget = int(float(budget)) if budget else None
    paths = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
    if not paths:
        raise SystemExit(f"no shards matched {pattern!r}")

    seen = 0
    for text in read_shards(paths):
        yield text
        seen += len(text)
        if budget and seen >= budget:
            return


def check_indent(tok: Tokenizer) -> bool:
    """
    Hard gate. Every indent width must encode to exactly one token.

    Checked mid-line (`\\n` + spaces + code) because that is how indentation actually
    occurs; a bare run of spaces is a different and easier case.
    """
    ok = True
    print("\nindent check")
    for width in INDENTS:
        ids = tok.encode("\n" + " " * width + "x").ids
        # newline, indent, then the token carrying x: 3 is the target, and the indent
        # itself must be the single middle token.
        pieces = [tok.id_to_token(i) for i in ids]
        good = len(ids) == 3
        ok &= good
        print(f"  {width:>2} spaces -> {len(ids)} tokens {pieces}  {'ok' if good else 'FAIL'}")

    real = "def f(x):\n    if x > 1:\n        return x ** 2\n    return x\n"
    ids = tok.encode(real).ids
    print(f"  real function: {len(real)} bytes -> {len(ids)} tokens "
          f"({len(real)/len(ids):.2f} bytes/token)")
    return bool(ok)


def report(tok: Tokenizer, samples, label: str) -> dict:
    """Bytes per token on held-out text -- the only number that says if this is any good."""
    n_bytes = n_tokens = 0
    for text in samples:
        n_bytes += len(text.encode())
        n_tokens += len(tok.encode(text).ids)
    ratio = n_bytes / max(n_tokens, 1)
    print(f"  {label:<12} {n_bytes/1e6:8.2f} MB -> {n_tokens/1e6:7.2f}M tokens"
          f"   {ratio:5.2f} bytes/token")
    return {"label": label, "bytes": n_bytes, "tokens": n_tokens, "bytes_per_token": ratio}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="glob:max_bytes, one per source")
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--no-gate", action="store_true",
                    help="report the indent check instead of failing on it")
    args = ap.parse_args()

    corpus, held_out = [], []
    for spec in args.shards:
        n = 0
        for i, text in enumerate(iter_budgeted(spec)):
            (held_out if i % 200 == 0 and len(held_out) < 400 else corpus).append(text)
            n += 1
        print(f"read {n:,} documents from {spec}")

    tok = build()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIALS,          # first, so they take the lowest ids
        # The full byte alphabet up front: every one of the 256 byte symbols gets an id
        # whether or not the sample happened to contain it, so nothing is unencodable.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"\ntraining on {len(corpus):,} documents, vocab {args.vocab_size}...")
    tok.train_from_iterator(corpus, trainer=trainer, length=len(corpus))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    print(f"\nvocab {tok.get_vocab_size()} | {EOS} = id {tok.token_to_id(EOS)}")
    stats = [report(tok, held_out, "held-out")] if held_out else []
    passed = check_indent(tok)

    checks = [
        "def f(x):\n    if x:\n        for i in range(10):\n            return x ** 2\n",
        "# héllo wörld — ünicode ✓\nprint('ok')\n",
        "\t\ttabs\tand   spaces\n\n\n",
        INSTRUCT_TEMPLATE.format(input="Reverse a list.", output="lst[::-1]"),
    ]
    bad = [c for c in checks if tok.decode(tok.encode(c).ids) != c]
    print(f"round-trip: {len(checks)-len(bad)}/{len(checks)} exact"
          + (f"  FAILED {bad!r}" if bad else ""))

    (out / "train_report.json").write_text(json.dumps({
        "vocab_size": tok.get_vocab_size(), "specials": SPECIALS,
        "eos_id": tok.token_to_id(EOS), "pattern": SPLIT_PATTERN,
        "indent_check_passed": passed, "stats": stats,
    }, indent=2))
    print(f"\nwrote {out/'tokenizer.json'}")

    if not passed and not args.no_gate:
        sys.exit("\nINDENT CHECK FAILED -- a third of the training budget would go to "
                 "whitespace. Fix the split pattern before tokenizing anything.")


if __name__ == "__main__":
    main()
