"""
Build the long-context ppl probes: 8192-token windows inside single long documents.

Answers "did later short-sequence stages erode the 2b context extension?" -- the
frozen ppl snapshots use 1024 windows, which cannot see it. Long docs are pulled
from the SAME frozen valid splits via their .idx offsets, so nothing here was
trained on, and each window sits entirely inside one document (no boundary noise).

    python -m src.eval.build_ppl_long
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

WINDOW = 8192
SOURCES = ("python", "web")          # the two corpora with real long docs


def main() -> None:
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    evals = Path("data/evals")
    manifest = json.loads((evals / "manifest.json").read_text())
    for name in SOURCES:
        ids = np.fromfile(f"data/tokens/valid_{name}.bin", dtype=np.uint16)
        offs = np.fromfile(f"data/tokens/valid_{name}.idx", dtype=np.uint64).astype(np.int64)
        lens = np.diff(offs)
        picks = np.nonzero(lens >= WINDOW)[0]
        # First WINDOW tokens of each long doc: fixed, reproducible, intra-document.
        wins = [ids[offs[i]:offs[i] + WINDOW] for i in picks]
        if not wins:
            print(f"{name}: no docs >= {WINDOW} tokens, skipped")
            continue
        stream = np.concatenate(wins)
        out = evals / "ppl" / f"long_{name}.bin"
        stream.tofile(out)
        nbytes = sum(len(tok.decode(w.tolist()).encode()) for w in wins)
        manifest["ppl"][f"ppl_long_{name}"] = dict(
            bin=f"ppl/long_{name}.bin", tokens=int(len(stream)), bytes=int(nbytes),
            window=WINDOW,
            note=f"first {WINDOW} tokens of the {len(wins)} valid_{name} docs >= {WINDOW}; "
                 "probes retention of the 2b context extension")
        print(f"ppl_long_{name}: {len(wins)} windows, {len(stream):,} tokens")
    (evals / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
