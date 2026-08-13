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

STABLE_MIX = {"python": 0.70, "web": 0.20, "code_instruct": 0.10}
DECAY_MIX = {"python": 0.60, "web": 0.40}


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

    def __post_init__(self):
        self.data, self.index = {}, {}
        for name in sorted(set(self.stable_mix) | set(self.decay_mix)):
            stem = f"{self.split}{name}"
            bin_path = Path(self.token_dir) / f"{stem}.bin"
            if not bin_path.exists():
                raise FileNotFoundError(f"{bin_path} -- run tokenize_corpus first")
            self.data[name] = np.memmap(bin_path, dtype=np.uint16, mode="r")
            idx_path = Path(self.token_dir) / f"{stem}.idx"
            if idx_path.exists():
                self.index[name] = np.fromfile(idx_path, dtype=np.uint64)

    # --- schedule ------------------------------------------------------------------

    def phase(self, step: int) -> str:
        return "stable" if step < self.decay_start * self.total_steps else "decay"

    def mix(self, step: int) -> dict:
        m = self.stable_mix if self.phase(step) == "stable" else self.decay_mix
        total = sum(m.values())
        return {k: v / total for k, v in m.items()}      # renormalized, so it always sums

    # --- sampling ------------------------------------------------------------------

    def batch(self, step: int, micro: int = 0, device="cpu"):
        """
        One [micro_batch, seq_len+1] window batch, plus the realized source mix.

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
            # +1 so the window yields both inputs and targets after the shift.
            start = int(rng.integers(0, len(stream) - self.seq_len - 1))
            rows[i] = stream[start : start + self.seq_len + 1].astype(np.int64)
            drawn[name] += 1

        t = torch.from_numpy(rows).to(device, non_blocking=True)
        realized = {n: drawn[n] / self.micro_batch for n in names}
        return t[:, :-1].contiguous(), t[:, 1:].contiguous(), realized

    def tokens_per_step(self, accum: int) -> int:
        return self.micro_batch * self.seq_len * accum

    def summary(self) -> dict:
        return {n: int(len(a)) for n, a in self.data.items()}
