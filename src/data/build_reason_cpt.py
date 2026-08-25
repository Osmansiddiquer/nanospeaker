"""
Assemble the reasoning-CPT blend into data/cpt/ (main segment) and data/cpt8k/ (tail).

Two row species in one stream: RAW sources are wrapped-packed with mask=1 everywhere
(pretraining CE); CHAT sources are whole-conversation rows with their response masks.
Shares are a dict so corridor branch B just edits SHARES and reruns.

    python -m src.data.build_reason_cpt
"""
import json
import random
from pathlib import Path

import numpy as np

ROW = 4096
TOTAL = 250_000_000

# name -> (share, kind). kind: raw_bin (tokens .bin) | chat_stream (bin+mask pair)
SHARES = {
    "nemomath":   (0.35, "raw", "data/tokens/nemomath.bin"),
    "sci_mot":    (0.08, "raw", "data/tokens/sci_mot.bin"),
    "sci_mega":   (0.07, "raw", "data/tokens/sci_mega.bin"),
    "codereason": (0.15, "raw", "data/tokens/codereason.bin"),
    "owm":        (0.05, "raw", "data/tokens/owm.bin"),
    "sft_replay": (0.10, "chat_rows", "data/sft/train.bin"),   # packed SFT rows (incl math CoT)
    "chatmix":    (0.08, "chat_rows", "data/sft_t/train.bin"),  # packed SFT-T rows
    "thinkcmd":   (0.05, "chat_stream", "data/sft_t/thinkcmd_pool.bin"),
    "web":        (0.06, "raw", "data/tokens/web.bin"),
    "python":     (0.06, "raw", "data/tokens/python.bin"),
}


def raw_rows(path, budget, rng):
    ids = np.memmap(path, dtype=np.uint16, mode="r")
    n_rows_available = len(ids) // ROW
    need = min(budget // ROW, n_rows_available)
    starts = rng.choice(n_rows_available, size=need, replace=False)
    X = np.stack([np.asarray(ids[s * ROW:(s + 1) * ROW]) for s in starts])
    M = np.ones_like(X, dtype=np.uint8)
    M[X == 0] = 0                        # eos separators unsupervised, as pretraining
    return X, M


def chat_rows(stem, budget, rng, kind):
    ids = np.memmap(stem, dtype=np.uint16, mode="r")
    mask = np.memmap(stem.replace(".bin", ".mask"), dtype=np.uint8, mode="r")
    if kind == "chat_stream":            # conversation stream: pack at boundaries
        ends = np.nonzero(np.asarray(ids) == 0)[0]
        rows_i, rows_m, cur_i, cur_m, n, start = [], [], [], [], 0, 0
        for e in ends:
            ci, cm = np.asarray(ids[start:e + 1]), np.asarray(mask[start:e + 1])
            start = e + 1
            if len(ci) > ROW or (len(ci) and ci[0] != 2):
                continue
            if n + len(ci) > ROW:
                rows_i.append(np.concatenate(cur_i + [np.zeros(ROW - n, np.uint16)]))
                rows_m.append(np.concatenate(cur_m + [np.zeros(ROW - n, np.uint8)]))
                cur_i, cur_m, n = [], [], 0
            cur_i.append(ci); cur_m.append(cm); n += len(ci)
        if cur_i:
            rows_i.append(np.concatenate(cur_i + [np.zeros(ROW - n, np.uint16)]))
            rows_m.append(np.concatenate(cur_m + [np.zeros(ROW - n, np.uint8)]))
        X, M = np.stack(rows_i), np.stack(rows_m)
        need = min(budget // ROW, len(X))
        pick = rng.choice(len(X), size=need, replace=False)
        return X[pick], M[pick]
    n_rows_available = len(ids) // ROW    # chat_rows: already boundary-packed rows
    need = min(budget // ROW, n_rows_available)
    starts = rng.choice(n_rows_available, size=need, replace=False)
    X = np.stack([np.asarray(ids[s * ROW:(s + 1) * ROW]) for s in starts])
    M = np.stack([np.asarray(mask[s * ROW:(s + 1) * ROW]) for s in starts])
    return X, M


def main() -> None:
    rng = np.random.default_rng(7)
    parts_x, parts_m, report = [], [], {}
    for name, (share, kind, path) in SHARES.items():
        budget = int(TOTAL * share)
        if not Path(path).exists():
            report[name] = "MISSING (fallback shares should have replaced this)"
            continue
        X, M = raw_rows(path, budget, rng) if kind == "raw" else chat_rows(path, budget, rng, kind)
        parts_x.append(X); parts_m.append(M)
        report[name] = f"{len(X):,} rows / {X.size/1e6:.0f}M tokens"
    X = np.concatenate(parts_x); M = np.concatenate(parts_m)
    order = rng.permutation(len(X))
    X, M = X[order], M[order]
    out = Path("data/cpt"); out.mkdir(exist_ok=True)
    X.tofile(out / "train.bin"); M.astype(np.uint8).tofile(out / "train.mask")
    # valid gates: reuse existing valid splits by symlinking the relevant ones
    for v in ("valid_nemomath", "valid_owm"):
        src = Path(f"data/tokens/{v}.bin")
        if src.exists():
            ids = np.fromfile(src, dtype=np.uint16)
            ids.tofile(out / f"{v}.bin")
            np.ones(len(ids), dtype=np.uint8).tofile(out / f"{v}.mask")
    for v in ("valid_thinkcmd",):
        for ext in (".bin", ".mask"):
            p = Path(f"data/sft_t/{v}{ext}")
            if p.exists():
                (out / f"{v}{ext}").write_bytes(p.read_bytes())
    print(json.dumps(report, indent=1))
    print(f"blend: {len(X):,} rows / {X.size/1e6:.0f}M tokens / supervised {M.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
