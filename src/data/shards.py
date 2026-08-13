"""
Sharded, resumable .jsonl.zst writing.

A corpus fetch is a long errand over an unreliable network, and the one thing it must
never do is start over. Every document is appended to a shard; every id is appended to
that shard's sidecar log the moment the document is safely written. On restart the logs
are read back and those ids are skipped, so a run that dies at minute 38 of 40 resumes at
minute 38.

Ids are logged after the write, never before -- an id in the log always means the text is
on disk, and the worst case is refetching one document, not silently losing one.
"""

import json
from pathlib import Path

import zstandard as zstd

SHARD_BYTES = 256 * 1024 * 1024        # uncompressed bytes per shard


class ShardWriter:
    """
    Append documents as {"text": ...} lines across ~256 MB shards.

        with ShardWriter("out/python", "python") as w:
            if doc_id not in w.seen:
                w.write(doc_id, text)
    """

    def __init__(self, out_dir, prefix: str, shard_bytes: int = SHARD_BYTES, level: int = 3):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.shard_bytes, self.level = prefix, shard_bytes, level

        # Every id ever written here, so callers can skip work they have already done.
        self.seen: set[str] = set()
        for log in sorted(self.dir.glob(f"{prefix}-*.ids")):
            self.seen.update(log.read_text().split())

        existing = sorted(self.dir.glob(f"{prefix}-*.jsonl.zst"))
        self.index = int(existing[-1].stem.split("-")[-1].split(".")[0]) + 1 if existing else 0
        self.docs = len(self.seen)
        self.bytes_written = sum(f.stat().st_size for f in existing)
        self._open()

    def _open(self) -> None:
        self.path = self.dir / f"{self.prefix}-{self.index:05d}.jsonl.zst"
        self._fh = self.path.open("wb")
        self._zst = zstd.ZstdCompressor(level=self.level).stream_writer(self._fh)
        self._ids = (self.dir / f"{self.prefix}-{self.index:05d}.ids").open("a")
        self._shard_bytes = 0

    def write(self, doc_id: str, text: str) -> int:
        """Append one document; returns its uncompressed size in bytes."""
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        raw = line.encode()
        self._zst.write(raw)
        self._ids.write(f"{doc_id}\n")

        self.seen.add(doc_id)
        self.docs += 1
        self._shard_bytes += len(raw)
        if self._shard_bytes >= self.shard_bytes:
            self.roll()
        return len(raw)

    def roll(self) -> None:
        """Close the current shard and start the next. Flushes the id log with it."""
        self._zst.close()
        self._fh.close()
        self._ids.close()
        self.bytes_written += self.path.stat().st_size
        self.index += 1
        self._open()

    def flush(self) -> None:
        self._zst.flush(zstd.FLUSH_FRAME)
        self._fh.flush()
        self._ids.flush()

    def close(self) -> None:
        self._zst.close()
        self._fh.close()
        self._ids.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_shards(pattern):
    """
    Yield texts from .jsonl.zst shards, in filename order.

    Tolerant of a shard that is still being written: the newest one ends in a partial
    frame, whose tail decodes to a truncated line. That line is dropped and the shard
    ends there, so a tokenizer or tokenizer pass can read a corpus while it is still
    landing rather than waiting for the fetch to finish.
    """
    import io

    dctx = zstd.ZstdDecompressor()
    for path in sorted(Path().glob(pattern) if isinstance(pattern, str) else pattern):
        try:
            with path.open("rb") as fh, dctx.stream_reader(fh) as raw:
                for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)["text"]
                    except (json.JSONDecodeError, KeyError):
                        break                      # truncated tail of a live shard
        except zstd.ZstdError:
            continue                               # frame still open; nothing more here
