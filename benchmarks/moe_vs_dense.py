"""
~200M-parameter decoder, MoE (64 experts) vs dense, iso-parameter, on one GPU.

Both models share the same skeleton built from this repo's components -- embedding,
8 x (LN -> causal MHA+RoPE -> LN -> FFN), LN, unembed -- and the same ~23M FFN
parameters per layer. The dense model spends them on one wide GLU (d_ff ~ 22.4k);
the MoE spends them on 64 small experts (d_ff 350 each) of which top_k=2 run per
token, i.e. ~3% of the dense FFN FLOPs at the same parameter count. The MoE rows
differ only in dispatch: the eager per-expert loop (64 GEMM launches + a host sync
per layer), the padded batched GEMM (drops overflow tokens), and the fused Triton
grouped GEMM from moe_kernel.py (no sync, no drops).

Measures a training step (forward + backward, cross-entropy on random targets, no
optimizer state -- Adam moments alone would not fit 200M on a 4 GB card) in bf16.
Run:  python benchmarks/moe_vs_dense.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.model.glu import GLUFeedForward, glu_hidden_dim
from src.model.ln import LayerNormalization
from src.model.mha import OptimizedMultiHeadAttention
from src.model.moe import MoEBlock
from src.model.rope import RotaryPositionalEmbedding

VOCAB, D_MODEL, N_LAYERS, N_HEADS, D_HEAD = 8192, 512, 8, 8, 64
SEQ = 512
FFN_BUDGET = 3 * D_MODEL * glu_hidden_dim(22_398)      # dense GLU: ~22.9M params/layer
N_EXPERTS, TOP_K, D_FF_EXPERT = 64, 2, 350             # 64 * 3*512*233 ~ same budget


class Block(nn.Module):
    """Pre-norm decoder block; `ffn` is the only thing the two models disagree on."""

    def __init__(self, ffn, rope):
        super().__init__()
        self.ln1 = LayerNormalization(D_MODEL)
        self.attn = OptimizedMultiHeadAttention(
            D_MODEL, N_HEADS, D_HEAD, D_HEAD, mask=True, rope=rope
        )
        self.ln2 = LayerNormalization(D_MODEL)
        self.ffn = ffn

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class Decoder(nn.Module):
    def __init__(self, make_ffn):
        super().__init__()
        rope = RotaryPositionalEmbedding(D_HEAD, max_seq_len=SEQ)
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block(make_ffn(), rope) for _ in range(N_LAYERS))
        self.ln_f = LayerNormalization(D_MODEL)
        self.unembed = nn.Parameter(torch.randn(D_MODEL, VOCAB) * 0.02)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        # Logits in fp32: a bf16 softmax over 8k classes is where precision actually goes.
        return (self.ln_f(x) @ self.unembed).float()

    def aux(self):
        return sum(b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoEBlock))


def build(kind):
    torch.manual_seed(0)
    if kind == "dense":
        make = lambda: GLUFeedForward(D_MODEL, d_ff=22_398)
    else:
        cap = 1.25 if kind == "moe-padded" else None
        make = lambda: MoEBlock(
            D_MODEL, N_EXPERTS, d_ff=D_FF_EXPERT, top_k=TOP_K,
            capacity_factor=cap, kernel=(kind == "moe-kernel"),
        )
    m = Decoder(make).cuda().bfloat16()
    return m


def step_fn(m, batch):
    ids = torch.randint(0, VOCAB, (batch, SEQ), device="cuda")
    tgt = torch.randint(0, VOCAB, (batch, SEQ), device="cuda")

    def step():
        loss = nn.functional.cross_entropy(
            m(ids).reshape(-1, VOCAB), tgt.reshape(-1)
        ) + m.aux()
        loss.backward()
        m.zero_grad(set_to_none=True)

    return step


def bench(kind, batch, iters=20):
    m = build(kind)
    n_params = sum(p.numel() for p in m.parameters())
    step = step_fn(m, batch)
    try:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"{kind:12s} batch {batch}: {dt * 1e3:8.1f} ms/step  "
              f"{batch * SEQ / dt:9.0f} tok/s  peak {peak:5.0f} MiB  "
              f"({n_params / 1e6:.0f}M params)")
        return dt
    except torch.OutOfMemoryError:
        print(f"{kind:12s} batch {batch}:      OOM on this card")
        return None
    finally:
        del m, step
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.manual_seed(0)
    name = torch.cuda.get_device_name(0)
    print(f"{name}, bf16, seq {SEQ}, vocab {VOCAB}, {N_LAYERS} layers, "
          f"{N_EXPERTS} experts top-{TOP_K} vs dense d_ff 22398\n")
    for batch in (2, 8):
        for kind in ("dense", "moe-eager", "moe-padded", "moe-kernel"):
            bench(kind, batch)
        print()
