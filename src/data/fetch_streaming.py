"""
Fetch the two inline corpora: Ultra-FineWeb (web text) and OpenCodeInstruct.

Both ship their contents in the rows, so these are ordinary streaming reads with no
second hop. Every fetch is gated on bytes written, never on row count -- row counts in
the plan are stopping heuristics, and document lengths vary far too much to trust them.

OpenCodeInstruct is filtered hard rather than sampled: only rows whose solution passes
its own tests are kept, which is roughly 60-70% of what is read. The instruction template
below is written into the manifest verbatim, because SFT later must reuse the exact same
one -- a format change between decay and SFT wastes the adaptation.

    python -m src.data.fetch_streaming --source web  --out data/raw/web
    python -m src.data.fetch_streaming --source code --out data/raw/code_instruct
"""

import argparse
import json
import time
from pathlib import Path

from .shards import ShardWriter

# Frozen. Reuse verbatim at SFT time; the tokenizer also reserves these delimiters.
INSTRUCT_TEMPLATE = "### Instruction\n{input}\n\n### Response\n{output}"

# Ultra-FineWeb is 58,624 parquet files. `datasets` resolves that whole listing before
# yielding row one, which never completed here -- so the English shards are read directly.
# Each is ~1.3 GB holding ~566k documents, which is already more text than the target.
UFW_SHARD = "data/ultrafineweb_en/ultrafineweb-en-part-{i:04d}-of-2048.parquet"

SOURCES = {
    "web": dict(path="openbmb/Ultra-FineWeb", config=None, split="en",
                target=1.68e9, prefix="web"),
    "code": dict(path="nvidia/OpenCodeInstruct", config=None, split="train",
                 target=0.49e9, prefix="code_instruct"),
}


def passes_tests(row) -> bool:
    """Keep only solutions verified against their own tests."""
    score = row.get("average_test_score")
    if score is not None:
        try:
            return float(score) == 1.0
        except (TypeError, ValueError):
            return False

    # Fall back to the per-test statuses when the aggregate is absent.
    status = row.get("tests_execution_status")
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except json.JSONDecodeError:
            return False
    if isinstance(status, (list, tuple)) and status:
        return all(str(s).lower() in ("pass", "passed", "1", "true") for s in status)
    return False


def row_text(source: str, row) -> "str | None":
    if source == "web":
        return row.get("content") or row.get("text")
    if not passes_tests(row):
        return None
    inp, out = row.get("input"), row.get("output")
    return INSTRUCT_TEMPLATE.format(input=inp, output=out) if inp and out else None


def iter_web(target: int):
    """
    Stream Ultra-FineWeb documents straight from the parquet shards.

    Downloaded a shard at a time rather than range-read: a row group is ~130 MB and must
    arrive whole either way, and one bulk transfer beats ten trickled ones. The cache is
    the hub's own, so a re-run resumes rather than refetches.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for i in range(1, 2049):
        local = hf_hub_download("openbmb/Ultra-FineWeb", UFW_SHARD.format(i=i),
                                repo_type="dataset")
        f = pq.ParquetFile(local)
        for batch in f.iter_batches(batch_size=1024, columns=["content"]):
            for text in batch.column("content").to_pylist():
                if text:
                    yield text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-bytes", type=float, default=None)
    args = ap.parse_args()

    from datasets import load_dataset

    spec = SOURCES[args.source]
    target = int(args.target_bytes or spec["target"])
    out = Path(args.out or f"data/raw/{spec['prefix']}")

    writer = ShardWriter(out, spec["prefix"])
    print(f"{args.source}: resuming with {len(writer.seen):,} ids, target "
          f"{target/1e9:.2f} GB", flush=True)

    if args.source == "web":
        ds = ({"content": t} for t in iter_web(target))
    else:
        ds = load_dataset(spec["path"], spec["config"], split=spec["split"],
                          streaming=True)
    got, kept, seen = 0, 0, 0
    t0 = time.time()

    for row in ds:
        seen += 1
        # Ids are positional for these sources: the streams are deterministic, so an
        # index is a stable name and costs nothing to log.
        doc_id = f"{spec['prefix']}-{seen}"
        if doc_id in writer.seen:
            continue
        text = row_text(args.source, row)
        if not text:
            continue
        got += writer.write(doc_id, text)
        kept += 1
        if kept % 20_000 == 0:
            print(f"  {kept:>8,} kept / {seen:>8,} seen  {got/1e9:5.2f}/{target/1e9:.2f} GB"
                  f"  {seen/max(time.time()-t0,1e-9):6.0f} rows/s", flush=True)
        if got >= target:
            break

    writer.flush()
    writer.close()
    (out / "manifest.json").write_text(json.dumps({
        "source": spec["path"], "split": spec["split"], "config": spec["config"],
        "bytes": got, "docs": kept, "rows_read": seen,
        "template": INSTRUCT_TEMPLATE if args.source == "code" else None,
        "filter": "average_test_score == 1.0" if args.source == "code" else None,
    }, indent=2))
    print(f"done: {kept:,} docs from {seen:,} rows, {got/1e9:.2f} GB, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
