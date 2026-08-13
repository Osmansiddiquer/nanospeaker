"""
nanoSpeaker: the decoder this repo was built toward.

A ~382M-parameter sparse decoder that trains inside 4 GB. Every choice below was settled
by measurement on that card rather than by taste, and the notes say which measurement.

    embedding (tied)
    20 x [ RMSNorm -> GQA + QK-norm + RoPE -> +residual
           RMSNorm -> MoE (76 routed, top-6, 2 shared) -> +residual ]
    RMSNorm
    tied unembed -> fused linear cross-entropy

Shape, and why:

    d_model 576, 20 layers      depth over width: the parameter budget buys more from
                                layers than from a wider residual stream at this scale.
    9 query / 3 KV heads        grouped-query, so the KV cache is a third the size at
                                inference, which is what bounds batch at long context.
    d_head 64                   9 * 64 = 576, so the heads tile the stream exactly.
    QK-norm                     bounds attention logits in bf16 (see mha.qk_norm).
    RoPE base 10000             relative position, no learned table (see rope.py).
    76 routed experts, top-6    granularity G = d_ff_dense / d_expert ~ 16, which is the
    2 shared, d_expert 128      band the fine-grained MoE scaling laws put a model this
                                size in (Ludziejewski et al., 2024). Shared experts carry
                                what every token needs so the routed ones can specialize.
    capacity_factor None        dropless. No token skips its FFN, and on CUDA this is the
                                path the fused Triton kernel takes (moe_kernel.py).
    vocab 32768, tied           the unembed is the embedding transposed.

Roughly 382M parameters, ~72M active per token (19%).

Training notes, all of them measured on the 4 GB card:

    gradient checkpointing      `use_reentrant=True`. The non-reentrant implementation
                                holds every block's recompute until the whole backward
                                finishes -- 3816 MiB peak against 1393 for reentrant.
    fused linear cross-entropy  never materializes the [tokens, 32768] logits, which
                                alone is 2.8 GiB at 8192 tokens. Falls back to a plain
                                head when liger-kernel is absent.
    expandable_segments         set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
                                Dropless dispatch requests a different shape every step,
                                so the default allocator fragments: 3955 MiB reserved
                                against 2373 with expandable segments.
    optimizer                   Muon on the 2D hidden weights, AdamW on embedding, norms
                                and router. Muon cost 1.8% throughput over stateless SGD
                                (6848 vs 6977 tok/s); AdamW would not fit at all.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .kv_cache import KVCache, make_kv_caches
from .ln import RMSNorm
from .mha import OptimizedMultiHeadAttention
from .moe import MoEBlock
from .rope import RotaryPositionalEmbedding

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    _HAS_LIGER = True
except ImportError:                                    # falls back to a plain head
    _HAS_LIGER = False


@dataclass
class NanoSpeakerConfig:
    """Every number here is a measured choice; see the module docstring for which."""

    vocab_size: int = 32768
    d_model: int = 576
    n_layers: int = 20
    n_heads: int = 9
    n_kv_heads: int = 3
    d_head: int = 64
    max_seq_len: int = 1024

    n_experts: int = 76
    n_shared: int = 2
    top_k: int = 6
    d_expert: int = 128
    capacity_factor: "float | None | str" = None       # dropless; the kernel's path
    aux_loss_coef: float = 1e-2
    noise_std: float = 0.0                             # raise early to spread the load

    rope_base: float = 10_000.0
    rope_p: float = 1.0                                # 1.0 = plain RoPE, < 1 truncates
    window: "int | None" = None                        # sliding window, or None for full
    std: float = 0.02
    checkpoint: bool = True

    @property
    def d_ff_expert(self) -> int:
        """MoEBlock budgets in d_ff and takes 2/3 of it, so invert that for d_expert."""
        return 3 * self.d_expert // 2


class Block(nn.Module):
    """Pre-norm: attention and FFN each read a normalized copy and write to the residual."""

    def __init__(self, cfg: NanoSpeakerConfig, rope: RotaryPositionalEmbedding):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = OptimizedMultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_head,
            mask=True, rope=rope, n_kv_heads=cfg.n_kv_heads, window=cfg.window,
            qk_norm=True, std=cfg.std, n_layers=cfg.n_layers,
        )
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = MoEBlock(
            cfg.d_model, cfg.n_experts, cfg.d_ff_expert, top_k=cfg.top_k,
            n_shared=cfg.n_shared, capacity_factor=cfg.capacity_factor,
            normalize_weights=False,                   # gate = the selected affinity
            aux_loss_coef=cfg.aux_loss_coef, noise_std=cfg.noise_std, std=cfg.std,
        )

    def forward(self, x: torch.Tensor, cache: "KVCache | None" = None):
        x = x + self.attn(self.norm_attn(x), cache=cache)
        # The balancing loss rides out with the activations: reading it off the module
        # afterwards would return whatever the recompute left there, not this pass.
        return x + self.ffn(self.norm_ffn(x)), self.ffn.aux_loss


class NanoSpeaker(nn.Module):
    """
    Args:
        cfg: see `NanoSpeakerConfig`.

    `forward(idx)` returns logits; `forward(idx, targets)` returns (loss, aux_loss) and
    never materializes the logits, which is what keeps a 32k vocab affordable. Add
    `aux_loss` to the loss you optimize -- with no capacity limit it is the only thing
    keeping the router balanced.
    """

    def __init__(self, cfg: NanoSpeakerConfig = NanoSpeakerConfig()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.std)

        # One RoPE for the whole stack: it holds tables, not parameters.
        rope = RotaryPositionalEmbedding(
            cfg.d_head, base=cfg.rope_base, p=cfg.rope_p, max_seq_len=cfg.max_seq_len
        )
        self.blocks = nn.ModuleList(Block(cfg, rope) for _ in range(cfg.n_layers))
        self.norm_out = RMSNorm(cfg.d_model)
        self.fused_ce = LigerFusedLinearCrossEntropyLoss() if _HAS_LIGER else None

    # --- parameter accounting ------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_active_params(self) -> int:
        """What one token actually touches: only top_k + n_shared experts of each layer."""
        cfg = self.cfg
        per_expert = 3 * cfg.d_model * cfg.d_expert
        idle = cfg.n_layers * per_expert * (cfg.n_experts - cfg.top_k)
        return self.n_params() - idle

    def caches(self, max_seq_len: "int | None" = None, max_chunk: int = 1):
        """One KV cache per layer, window-sized when the model attends over a window."""
        return make_kv_caches(
            self.cfg.n_layers, max_seq_len or self.cfg.max_seq_len,
            self.cfg.window, max_chunk,
        )

    # --- forward -------------------------------------------------------------------

    def trunk(self, idx: torch.Tensor, caches=None):
        x, aux = self.embed(idx), idx.new_zeros((), dtype=torch.float32)
        caches = caches or [None] * len(self.blocks)
        for block, cache in zip(self.blocks, caches):
            if self.cfg.checkpoint and self.training and cache is None:
                # Reentrant: the non-reentrant path never frees a block's recompute.
                x, a = checkpoint(block, x, use_reentrant=True)
            else:
                x, a = block(x, cache)
            aux = aux + a
        return self.norm_out(x), aux

    def forward(self, idx: torch.Tensor, targets: "torch.Tensor | None" = None,
                caches=None):
        x, aux = self.trunk(idx, caches)
        if targets is None:
            return x @ self.embed.weight.T             # tied unembed

        # Next-token objective: position t predicts t+1, so the last position has no
        # target and the first no prediction.
        h = x[:, :-1].reshape(-1, self.cfg.d_model)
        y = targets[:, 1:].reshape(-1)

        # The fused path is a Triton kernel: CUDA only, and not in fp64. Everywhere else
        # falls back to materializing the logits, which is fine at the sizes that run
        # there and would be 2.8 GiB at the size that doesn't.
        if (self.fused_ce is not None and h.is_cuda
                and h.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            return self.fused_ce(self.embed.weight, h, y), aux
        return F.cross_entropy((h @ self.embed.weight.T).float(), y), aux

    # --- inference -----------------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n_tokens: int, temperature: float = 1.0,
                 top_k: "int | None" = None):
        """Greedy/sampled decode through per-layer KV caches, one token per step."""
        caches = self.caches(idx.shape[1] + n_tokens + 1)
        logits = self(idx, caches=caches)[:, -1]
        out = []
        for _ in range(n_tokens):
            logits = logits.float() / max(temperature, 1e-6)
            if top_k:
                logits[logits < logits.topk(top_k, dim=-1).values[:, -1:]] = -float("inf")
            nxt = (torch.multinomial(logits.softmax(-1), 1) if temperature > 0
                   else logits.argmax(-1, keepdim=True))
            out.append(nxt)
            logits = self(nxt, caches=caches)[:, -1]
        return torch.cat([idx] + out, dim=1)
