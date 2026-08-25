"""
Mixture data loader over tokenized memmaps, with phase-dependent ratios.

Every batch is a pure function of (seed, step, micro_index). Nothing is carried between
steps -- no cursor, no shuffle buffer, no iterator state -- so resuming at step 9,000 is
exactly as correct as arriving there by training, and needs nothing in the checkpoint but
the step number. That property is worth more than any sampling sophistication it costs.

Sequences are drawn at random offsets into each source's flat token stream. Documents are
EOS-separated in that stream, so a window may span a boundary; that is deliberate and
standard, and the model learns the separator. `.idx` is loaded anyway because evaluation
and any future document-aligned work needs it.

Ratios come from the phase: stable for the first `decay_start` fraction of the run, decay
after. The realized mix is reported per batch so the log records what was actually drawn
rather than what was intended.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Instruct is held back entirely for the decay phase. It is the cleanest source by a
# wide margin (validation 3.32 against python 5.80 and web 7.37 at step 130) and there
# are only 110.7M tokens of it -- spending it during stable would leave nothing to
# anneal onto. At 45% of decay it is used exactly once, no document twice.
STABLE_MIX = {"python": 0.80, "web": 0.20}
DECAY_MIX = {"python": 0.40, "code_instruct": 0.45, "web": 0.15}


@dataclass
class MixtureLoader:
    token_dir: str = "data/tokens"
    seq_len: int = 1024
    micro_batch: int = 8
    seed: int = 0
    decay_start: float = 0.80          # fraction of the run where decay begins
    total_steps: int = 12_512
    stable_mix: dict = field(default_factory=lambda: dict(STABLE_MIX))
    decay_mix: dict = field(default_factory=lambda: dict(DECAY_MIX))
    split: str = ""                    # "" for training, "valid_" for held-out
    min_doc_len: "int | dict" = 0      # per source, or one value for all

    def __post_init__(self):
        self.data, self.index, self.long_docs = {}, {}, {}
        for name in sorted(set(self.stable_mix) | set(self.decay_mix)):
            stem = f"{self.split}{name}"
            bin_path = Path(self.token_dir) / f"{stem}.bin"
            if not bin_path.exists():
                raise FileNotFoundError(f"{bin_path} -- run tokenize_corpus first")
            self.data[name] = np.memmap(bin_path, dtype=np.uint16, mode="r")
            idx_path = Path(self.token_dir) / f"{stem}.idx"
            if idx_path.exists():
                self.index[name] = np.fromfile(idx_path, dtype=np.uint64)
        if self.min_doc_len:
            self._find_long_docs()

    def _find_long_docs(self):
        """
        Document starts whose document is long enough to fill a window on its own.

        Training a long context on the flat stream teaches position mechanics and not
        long-range dependency: the median document here is a few hundred tokens, so an
        8k window over the shuffled stream is nine unrelated documents in a trench coat
        and nothing in it rewards attending past the nearest boundary. Upsampling the
        documents that are genuinely long is what Fu et al. (2024) found actually moves
        long-context ability.

        Sources without enough long documents keep the flat draw rather than cycling a
        handful of files -- overfitting six documents would be worse than the problem.
        """
        # Per source, because the corpora are shaped differently: web and python carry
        # real long documents, while OpenCodeInstruct tops out at 2,006 tokens and the
        # Q/A set is capped at 6,000 characters by its own fetch filter. One global
        # threshold either excludes those two entirely or is too loose for the others.
        for name, idx in self.index.items():
            want = (self.min_doc_len.get(name, 0) if isinstance(self.min_doc_len, dict)
                    else self.min_doc_len)
            if not want:
                continue
            need = want + 1                       # +1 so the window has a target token
            starts, lengths = idx[:-1], np.diff(idx)
            long = lengths >= need
            # A few dozen documents cannot carry a phase; fall back to the flat stream
            # rather than cycling a handful of files until the model memorizes them.
            if long.sum() >= 64:
                self.long_docs[name] = (starts[long].astype(np.int64),
                                        lengths[long].astype(np.int64))

    # --- schedule ------------------------------------------------------------------

    def phase(self, step: int) -> str:
        return "stable" if step < self.decay_start * self.total_steps else "decay"

    def mix(self, step: int) -> dict:
        m = self.stable_mix if self.phase(step) == "stable" else self.decay_mix
        total = sum(m.values())
        return {k: v / total for k, v in m.items()}      # renormalized, so it always sums

    # --- sampling ------------------------------------------------------------------

    def _start(self, rng, name: str, n: int) -> int:
        """Where this row's window begins: uniform in the stream, or inside a long doc."""
        long = self.long_docs.get(name)
        if long is not None:
            starts, lengths = long
            d = int(rng.integers(0, len(starts)))
            # Anywhere in the document that still leaves a full window ahead of it.
            slack = int(lengths[d]) - self.seq_len - 1
            return int(starts[d]) + (int(rng.integers(0, slack + 1)) if slack > 0 else 0)
        # The +1 is vestigial -- the loader no longer shifts -- but it stays. It sets the
        # rng.integers upper bound, so dropping it would move every start offset and
        # rewrite the whole data stream, which no existing checkpoint could resume.
        return int(rng.integers(0, n - self.seq_len - 1))

    def batch(self, step: int, micro: int = 0, device="cpu"):
        """
        One [micro_batch, seq_len] window, returned twice, plus the realized source mix.

        Inputs and targets are the same unshifted tensor. The shift belongs to the model,
        which takes (idx, targets) and offsets them itself; a loader that also shifted
        would stack the two into a two-ahead objective.

        Seeded on (seed, step, micro): the same three numbers always produce the same
        tokens, which is what makes resume exact and makes a bug reproducible.
        """
        rng = np.random.default_rng((self.seed, step, micro))
        mix = self.mix(step)
        names = list(mix)
        picks = rng.choice(names, size=self.micro_batch, p=[mix[n] for n in names])

        rows = np.empty((self.micro_batch, self.seq_len + 1), dtype=np.int64)
        drawn = dict.fromkeys(names, 0)
        for i, name in enumerate(picks):
            stream = self.data[name]
            start = self._start(rng, name, len(stream))
            rows[i] = stream[start : start + self.seq_len + 1].astype(np.int64)
            drawn[name] += 1

        t = torch.from_numpy(rows).to(device, non_blocking=True)
        realized = {n: drawn[n] / self.micro_batch for n in names}
        # The same unshifted window twice: forward() does the one shift, and doing it here
        # as well is what trained a model to predict two tokens ahead.
        x = t[:, :-1].contiguous()
        return x, x, realized

    def tokens_per_step(self, accum: int) -> int:
        return self.micro_batch * self.seq_len * accum

    def summary(self) -> dict:
        return {n: int(len(a)) for n, a in self.data.items()}
