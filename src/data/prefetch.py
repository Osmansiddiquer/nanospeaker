"""
Background batch construction for MixtureLoader.

`MixtureLoader.batch` costs ~1.5 s of every 12 s step: eight random reads into memmaps
spread over ~3 GB of `.bin` (page faults, on WSL), a uint16 -> int64 inflate, and a
pageable host-to-device copy. All of it runs between micro-batches while the GPU idles.

None of that work depends on the model, and a batch is a pure function of
(seed, step, micro) -- so a worker thread can build batch N+1 while N is training, and
the result is bit-identical to building it inline. Resume and reproducibility are
untouched by construction, not by care.

The staging tensors are pinned, which is what finally makes `non_blocking=True` mean
anything: a pageable copy is synchronous no matter what the flag says. Torch's caching
host allocator tracks the copy events, so pinned blocks are only recycled once the
transfer that read them has completed.

The worker follows a cursor (micro 0..accum-1, then the next step). A request that
doesn't match the cursor -- a resume, an out-of-order read in a test -- resets it and
refills, so correctness never depends on the caller walking in order.
"""

import threading
from collections import deque


class BatchPrefetcher:
    """Prebuilds batches on a worker thread. `get` blocks only when the worker is behind."""

    def __init__(self, loader, accum: int, device="cpu", depth: int = 2,
                 total_steps: int | None = None):
        self.loader, self.accum, self.device = loader, accum, device
        self.depth = max(1, depth)
        self.total_steps = total_steps

        self._cv = threading.Condition()
        self._ready = deque()          # (step, micro, payload) built and waiting
        self._gen = 0                  # bumped on reset; stale builds are dropped
        self._cursor = (0, 0)          # what the worker builds next
        self._expect = (0, 0)          # what the caller is expected to ask for next
        self._stop = False

        self._thread = threading.Thread(target=self._work, name="prefetch", daemon=True)
        self._thread.start()

    # --- the cursor ----------------------------------------------------------------

    def _advance(self, step: int, micro: int):
        return (step, micro + 1) if micro + 1 < self.accum else (step + 1, 0)

    def _exhausted(self, step: int) -> bool:
        return self.total_steps is not None and step >= self.total_steps

    def _reset_locked(self, step: int, micro: int):
        self._gen += 1
        self._ready.clear()
        self._cursor = (step, micro)
        self._cv.notify_all()

    # --- the worker ----------------------------------------------------------------

    def _build(self, step: int, micro: int):
        x, y, mix = self.loader.batch(step, micro)          # cpu, pageable
        return x.pin_memory(), y.pin_memory(), mix

    def _work(self):
        while True:
            with self._cv:
                while not self._stop and (len(self._ready) >= self.depth
                                          or self._exhausted(self._cursor[0])):
                    self._cv.wait()
                if self._stop:
                    return
                gen, (step, micro) = self._gen, self._cursor
                self._cursor = self._advance(step, micro)

            try:
                payload = self._build(step, micro)
            except BaseException as e:                      # surfaced on the caller's get
                payload = e

            with self._cv:
                if gen == self._gen:
                    self._ready.append((step, micro, payload))
                self._cv.notify_all()

    # --- the caller ----------------------------------------------------------------

    def get(self, step: int, micro: int = 0):
        """The same triple `loader.batch(step, micro, device)` returns, built ahead."""
        with self._cv:
            if (step, micro) != self._expect:
                self._reset_locked(step, micro)
                self._expect = (step, micro)
            while not self._ready:
                self._cv.wait()
            s, m, payload = self._ready.popleft()
            self._expect = self._advance(step, micro)
            self._cv.notify_all()

        assert (s, m) == (step, micro), f"prefetch served ({s}, {m}), wanted ({step}, {micro})"
        if isinstance(payload, BaseException):
            raise payload
        x, y, mix = payload
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True), mix)

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
