"""
Re-pack the SFT stream into boundary-aligned rows (torchtune split_across_pack=False,
TRL "bfd"-family, axolotl multipack -- the SFT-standard packing this corpus should
have had from the start).

Reads the existing wrapped stream (conversations eos-delimited, masks parallel), so
no re-fetching or re-rendering: split at eos, greedy-fill 4096-token rows with WHOLE
conversations only, pad the gap with masked <|endoftext|>. Trailing padding sits
after every supervised position, so under causal attention it is invisible to all
predictions -- pure, harmless filler. Every row begins at <|im_start|>, which is
what finally trains the empty-window conditional at full rate instead of ~2%.

    python -m src.data.repack_sft          # rewrites data/sft/train.bin/.mask
"""
import numpy as np
from pathlib import Path

ROW = 4096


def main() -> None:
    d = Path("data/sft")
    ids = np.fromfile(d / "train.bin", dtype=np.uint16)
    msk = np.fromfile(d / "train.mask", dtype=np.uint8)
    assert len(ids) == len(msk)
    ends = np.nonzero(ids == 0)[0]
    rows_i, rows_m = [], []
    cur_i, cur_m, cur_len = [], [], 0
    start = dropped = 0
    for e in ends:
        ci, cm = ids[start:e + 1], msk[start:e + 1]     # conversation incl. its eos
        start = e + 1
        if len(ci) > ROW or ci[0] != 2:                 # oversize or malformed head
            dropped += 1
            continue
        if cur_len + len(ci) > ROW:
            pad = ROW - cur_len
            rows_i.append(np.concatenate(cur_i + [np.zeros(pad, np.uint16)]))
            rows_m.append(np.concatenate(cur_m + [np.zeros(pad, np.uint8)]))
            cur_i, cur_m, cur_len = [], [], 0
        cur_i.append(ci)
        cur_m.append(cm)
        cur_len += len(ci)
    if cur_i:
        pad = ROW - cur_len
        rows_i.append(np.concatenate(cur_i + [np.zeros(pad, np.uint16)]))
        rows_m.append(np.concatenate(cur_m + [np.zeros(pad, np.uint8)]))

    X = np.stack(rows_i)
    M = np.stack(rows_m)
    assert (X[:, 0] == 2).all(), "every row must start at a conversation"
    (d / "train_wrapped_v1.bin").write_bytes(b"")       # marker: old layout retired
    X.tofile(d / "train.bin")
    M.tofile(d / "train.mask")
    pad_frac = float((X == 0).sum() - len(ends)) / X.size
    print(f"{len(rows_i):,} rows of {ROW} | {X.size:,} tokens | "
          f"padding {pad_frac*100:.1f}% | supervised {M.mean()*100:.1f}% | "
          f"dropped {dropped} conversations")


if __name__ == "__main__":
    main()
