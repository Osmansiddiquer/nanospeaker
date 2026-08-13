#!/usr/bin/env python3
"""
Which Python corpus is faster to acquire on THIS machine?

Both probes measure the same quantity -- usable Python bytes per second --
and extrapolate to the 3.35 GB you need for ~1.05B tokens.

    pip install datasets boto3
    python probe_data_sources.py

Runtime: ~3 minutes. No data is kept.
"""

import gzip
import io
import time
from concurrent.futures import ThreadPoolExecutor

TARGET_BYTES = 3.35e9  # ~1.05B Python tokens at ~3.2 bytes/token


def fmt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


# --------------------------------------------------------------------------
# Probe A: Stack-Edu -- SWHIDs from HF, contents from Software Heritage S3
# --------------------------------------------------------------------------
def probe_stack_edu(n_blobs=1500, concurrencies=(64, 256)):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from datasets import get_dataset_config_names, load_dataset

    print("=" * 62)
    print("PROBE A: Stack-Edu (SWHID -> Software Heritage S3)")
    print("=" * 62)

    configs = get_dataset_config_names("HuggingFaceTB/stack-edu")
    cfg = next((c for c in configs if "python" in c.lower()), None)
    if cfg is None:
        print(f"  no python config found. available: {configs[:20]}")
        return None
    print(f"  config: {cfg}")

    t0 = time.time()
    ds = load_dataset("HuggingFaceTB/stack-edu", cfg, split="train", streaming=True)
    rows = []
    for r in ds:
        rows.append(r)
        if len(rows) >= n_blobs:
            break
    print(f"  pulled {len(rows)} SWHIDs in {time.time() - t0:.1f}s")

    # Stack v2 lineage: blob contents live at s3://softwareheritage/content/<blob_id>,
    # gzip-compressed, public/unsigned read.
    id_key = next((k for k in ("blob_id", "content_id", "swhid", "id")
                   if k in rows[0]), None)
    if id_key is None:
        print(f"  can't find a blob id field. keys: {list(rows[0])}")
        return None
    enc_key = "src_encoding" if "src_encoding" in rows[0] else None
    print(f"  id field: {id_key}")

    best = 0.0
    for workers in concurrencies:
        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED,
                          max_pool_connections=workers + 16,
                          retries={"max_attempts": 2}),
        )

        def fetch(row):
            blob = row[id_key]
            if isinstance(blob, str) and ":" in blob:      # full SWHID -> bare sha1
                blob = blob.rsplit(":", 1)[-1]
            try:
                body = s3.get_object(Bucket="softwareheritage",
                                     Key=f"content/{blob}")["Body"].read()
                raw = gzip.GzipFile(fileobj=io.BytesIO(body)).read()
                enc = (row.get(enc_key) or "utf-8") if enc_key else "utf-8"
                return len(raw.decode(enc, errors="replace").encode("utf-8"))
            except Exception:
                return 0

        sample = rows[: min(len(rows), workers * 6)]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            sizes = list(ex.map(fetch, sample))
        dt = time.time() - t0

        ok = sum(1 for s in sizes if s)
        total = sum(sizes)
        rps = ok / dt
        bps = total / dt
        best = max(best, bps)
        print(f"  {workers:4d} threads: {rps:7.1f} req/s  {bps / 1e6:6.2f} MB/s  "
              f"({ok}/{len(sample)} ok, mean {total / max(ok, 1) / 1024:.1f} KB)  "
              f"-> {fmt(TARGET_BYTES / bps) if bps else 'n/a'}")

    return best


# --------------------------------------------------------------------------
# Probe B: stack-v3-train -- stream repos, keep only Python files
# --------------------------------------------------------------------------
def probe_stack_v3(seconds=60):
    from datasets import load_dataset

    print("=" * 62)
    print("PROBE B: stack-v3-train (stream everything, keep Python)")
    print("=" * 62)

    ds = load_dataset("HuggingFaceCode/stack-v3-train",
                      split="train", streaming=True)

    kept = seen = repos = 0
    t0 = time.time()
    for repo in ds:
        repos += 1
        for f in repo["files"]:
            seen += f["size_bytes"]
            if (f["language"] == "Python"
                    and f["license_type"] == "permissive"
                    and not f["is_vendor"]
                    and not f["file_path"].endswith(".ipynb")):
                kept += len(f["content"].encode("utf-8"))
        if time.time() - t0 > seconds:
            break

    dt = time.time() - t0
    bps = kept / dt
    print(f"  {repos} repos in {dt:.0f}s")
    print(f"  streamed {seen / 1e6:.0f} MB, kept {kept / 1e6:.1f} MB Python "
          f"({100 * kept / max(seen, 1):.1f}%)")
    print(f"  {bps / 1e6:.2f} MB/s usable  -> {fmt(TARGET_BYTES / bps) if bps else 'n/a'}")
    return bps


if __name__ == "__main__":
    a = b = None
    try:
        a = probe_stack_edu()
    except Exception as e:
        print(f"  probe A failed: {e}")
    print()
    try:
        b = probe_stack_v3()
    except Exception as e:
        print(f"  probe B failed: {e}")

    print()
    print("=" * 62)
    if a and b:
        win = "Stack-Edu (SWHID)" if a > b else "stack-v3 (stream)"
        print(f"WINNER: {win}   {max(a, b) / min(a, b):.1f}x faster")
        print(f"  Stack-Edu : {fmt(TARGET_BYTES / a)}")
        print(f"  stack-v3  : {fmt(TARGET_BYTES / b)}")
    else:
        print("one probe failed -- use whichever completed")
    print("=" * 62)
    print("If probe A's req/s climbs a lot from 64 to 256 threads, it is")
    print("concurrency-bound, not server-bound: retry with 512 before deciding.")