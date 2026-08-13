"""
Fetch Stack-Edu (Python) blob contents from Software Heritage's public S3.

The dataset ships identifiers, not source. Each row names a blob; the text lives at
s3://softwareheritage/content/{sha1}, gzipped, readable without credentials. That makes
this ~1.2M small GETs at a mean 2.7 KB -- latency-bound, not bandwidth-bound, which is
why it runs 512 threads and why single-threaded it would take days rather than an hour.

Run this in its own process. A live boto3 thread pool alongside `datasets`' HTTP retry
machinery has crashed at interpreter shutdown (PyGILState_Release: no thread-state for
this thread), so the exit below is deliberately abrupt: flush, then os._exit, skipping
interpreter teardown entirely.

    python -m src.data.fetch_stack_edu --out data/raw/python --target-bytes 3.21e9
"""

import argparse
import gzip
import io
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from .shards import ShardWriter

BUCKET = "softwareheritage"


def make_client(workers: int):
    # The connection pool must exceed the thread count or threads block on each other
    # rather than on the network, which is the whole point of running 512 of them.
    return boto3.client("s3", config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=workers + 32,
        retries={"max_attempts": 3},
        connect_timeout=15,
        read_timeout=30,
    ))


def fetch_one(s3, blob_id: str, encoding: "str | None") -> "str | None":
    """One blob: GET, gunzip, decode. Returns None if it cannot be had."""
    try:
        raw = s3.get_object(Bucket=BUCKET, Key=f"content/{blob_id}")["Body"].read()
        data = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        return data.decode(encoding or "utf-8", errors="replace")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/python")
    ap.add_argument("--prefix", default="python")
    ap.add_argument("--config", default="Python")
    ap.add_argument("--target-bytes", type=float, default=3.21e9)
    ap.add_argument("--workers", type=int, default=512)
    ap.add_argument("--queue-size", type=int, default=8192)
    args = ap.parse_args()
    target = int(args.target_bytes)

    writer = ShardWriter(args.out, args.prefix)
    print(f"resuming with {len(writer.seen):,} ids already fetched, "
          f"{writer.bytes_written/1e9:.2f} GB on disk", flush=True)

    s3 = make_client(args.workers)
    rows: queue.Queue = queue.Queue(maxsize=args.queue_size)
    done = threading.Event()

    def produce():
        """Stream row metadata from Hugging Face into the queue."""
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceTB/stack-edu", args.config, split="train",
                          streaming=True)
        for row in ds:
            if done.is_set():
                break
            blob = str(row["blob_id"]).rsplit(":", 1)[-1]      # tolerate full SWHIDs
            if blob not in writer.seen:
                rows.put((blob, row.get("src_encoding")))
        rows.put(None)

    threading.Thread(target=produce, daemon=True).start()

    got = writer.bytes_written and 0      # count only this run's uncompressed bytes
    n = fails = 0
    t0 = time.time()

    def pump():
        """Yield work until the byte target is met or the producer runs dry."""
        while not done.is_set():
            item = rows.get()
            if item is None:
                return
            yield item

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for blob, enc in pump():
            futures[pool.submit(fetch_one, s3, blob, enc)] = blob
            if len(futures) < args.workers * 2:
                continue
            # Drain completed work in batches, so the queue stays full and the writer
            # (single-threaded by design) is never the bottleneck.
            for fut in list(futures):
                if not fut.done():
                    continue
                blob_id = futures.pop(fut)
                text = fut.result()
                if not text:
                    fails += 1
                    continue
                got += writer.write(blob_id, text)
                n += 1
                if n % 20_000 == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"  {n:>9,} blobs  {got/1e9:5.2f}/{target/1e9:.2f} GB  "
                          f"{rate:6.0f} req/s  {fails:,} failed", flush=True)
                if got >= target:
                    done.set()
                    break

        for fut in futures:                                    # settle what is in flight
            if done.is_set():
                fut.cancel()
            elif (text := fut.result()):
                got += writer.write(futures[fut], text)
                n += 1

    writer.flush()
    writer.close()
    elapsed = time.time() - t0
    print(f"\ndone: {n:,} blobs, {got/1e9:.2f} GB uncompressed, {fails:,} failed, "
          f"{elapsed/60:.1f} min, {n/max(elapsed,1e-9):.0f} req/s", flush=True)
    sys.stdout.flush()

    # Deliberate: skip interpreter teardown, which is where the boto3/datasets thread
    # interaction has crashed before. Everything durable is already flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
