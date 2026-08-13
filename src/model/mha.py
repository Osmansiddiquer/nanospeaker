import warnings

import torch
import torch.nn as nn
from torch.nn.attention.bias import causal_lower_right

from .kv_cache import KVCache
from .ln import RMSNorm
from .rope import RotaryPositionalEmbedding
from .utils import initialize_normal_torch_weights

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:                                    # older torch: windowed falls back
    create_block_mask = flex_attention = None

# One compiled flex kernel and one set of mask tables for the whole process: every layer
# runs the same shapes, so 32 blocks share a single compile and a single mask.
#
# Bounded, and FIFO-evicted. Chunked prefill walks the offset (0, 256, 512, ...) and
# decode walks S one token at a time, so both keys are unbounded in principle -- and a
# BlockMask holds index tensors resident on the GPU, so an uncapped cache is a leak that
# grows for as long as generation runs.
_FLEX_COMPILED = None
_CACHE_SIZE = 32
_BLOCK_MASKS: dict = {}
_MASKS: dict = {}


def _cached(cache: dict, key, build):
    hit = cache.get(key)
    if hit is None:
        if len(cache) >= _CACHE_SIZE:
            cache.pop(next(iter(cache)))            # oldest out; insertion order is FIFO
        hit = cache[key] = build()
    return hit


def _device_key(device):
    """`cuda` and `cuda:0` are the same device, and must not mint two cache entries."""
    device = torch.device(device)
    return device.type, torch.cuda.current_device() if device.index is None and device.type == "cuda" else device.index

# create_block_mask works in 128-wide blocks, and eager flex_attention is ~100x slower
# than the compiled one, so short queries are better served by a materialized mask.
_FLEX_MIN_QUERIES = 128


def causal_mask(T: int, S: int, offset: int, device, window: "int | None" = None):
    """
    [T, S] bool mask, True where a query may attend, for queries at offset..offset+T.

    Aligned bottom-right rather than top-left: with a KV cache the S keys are the whole
    prefix while the T queries are only its tail, so query i sits at absolute position
    offset + i and may see every key up to it. `window` additionally drops keys further
    than `window` tokens back, counting the query's own position as the first of them.
    """
    def build():
        q = torch.arange(offset, offset + T, device=device)[:, None]
        k = torch.arange(S, device=device)[None, :]
        allowed = k <= q
        return allowed if window is None else allowed & (q - k < window)

    # Every layer asks for the identical mask each step, so this is built once per shape
    # rather than 32 times: two aranges, a compare and a [T, S] bool allocation each.
    return _cached(_MASKS, (T, S, offset, window, _device_key(device)), build)


def ring_mask(T: int, offset: int, key_pos: torch.Tensor, window: int) -> torch.Tensor:
    """
    [T, len(key_pos)] bool mask for keys held out of order in a rolling cache.

    The ring stores slot j at absolute position key_pos[j], so the band is expressed
    against those positions rather than against the slot index.
    """
    q = torch.arange(offset, offset + T, device=key_pos.device)[:, None]
    k = key_pos[None, :]
    return (k <= q) & (q - k < window)


def _block_mask(T: int, S: int, offset: int, window: int, device):
    """Block-sparse BlockMask for the same band, cached: building one costs ~5ms."""
    def build():
        def mask_mod(b, h, q_idx, kv_idx):
            q = q_idx + offset
            return (kv_idx <= q) & (q - kv_idx < window)

        return create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=S, device=device
        )

    return _cached(_BLOCK_MASKS, (T, S, offset, window, _device_key(device)), build)


def _flex():
    """The compiled flex kernel. Eager flex_attention is not worth using."""
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_COMPILED


class OptimizedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional RoPE, KV cache, grouped-query heads and a
    sliding window.

    `n_kv_heads` sets how many key/value heads the query heads share, which is the one
    knob between MHA and MQA: n_kv_heads = n_heads is plain multi-head, 1 is multi-query,
    and anything dividing n_heads in between is grouped-query. Fewer KV heads shrink both
    the K/V projections and, decisively for generation, the KV cache -- which is what
    bounds batch size at long context -- at some cost in quality.

    `window` restricts each query to the `window` most recent tokens (itself included),
    i.e. Mistral-style sliding-window attention, turning the cost in sequence length from
    quadratic to linear. On CUDA this dispatches to `flex_attention`, whose block mask
    lets it skip whole tiles that fall outside the band rather than computing and
    discarding them; measured on 2048 tokens with a 256 window that is ~7x a masked SDPA
    and ~2x an unmasked causal one. Everywhere else (CPU, float64, short query blocks,
    flex unavailable) it falls back to SDPA with a materialized mask, which is correct
    and, on CPU, is actually the faster of the two by ~25x.

    Args:
        d_model: model (input/output) width.
        n_heads: query heads.
        d_k: per-head query/key width.
        d_v: per-head value width.
        mask: apply causal masking (decoder self-attention).
        d_out: output width; defaults to d_model.
        rope: a `rope.RotaryPositionalEmbedding` with `d_head == d_k`, applied to Q and K.
            None leaves the block position-agnostic (NoPE). Share one instance across
            layers -- it holds tables, not parameters.
        n_kv_heads: key/value heads; must divide n_heads. Defaults to n_heads (no sharing).
        window: sliding-window span, or None for full causal attention. Requires `mask`.
        flex: allow the block-sparse flex kernel for windowed attention (CUDA only).
        qk_norm: RMS-normalize each head's Q and K before RoPE, which bounds the
            attention logits in low precision.

    Pair a windowed block with `KVCache(max_seq_len, window=window)` at inference: the
    cache then rolls a window-sized ring instead of holding the whole context, and a
    decode step attends over the window rather than over a full-length prefix it would
    only mask away. That also drops the mask entirely at T = 1, which matters because
    SDPA's flash backend refuses an explicit `attn_mask` and falls back to the
    memory-efficient one -- every masked path here pays that.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        mask: bool = False,
        d_out=None,
        rope: "RotaryPositionalEmbedding | None" = None,
        n_kv_heads: "int | None" = None,
        window: "int | None" = None,
        flex: bool = True,
        qk_norm: bool = False,
        std: float = 0.02,
        n_layers: "int | None" = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.use_mask = mask

        # Only a problem when d_k was defaulted from d_model; with d_k given explicitly
        # the head widths need not tile d_model at all, and warning would be noise.
        if n_heads * d_k != d_model and d_model % n_heads != 0:
            warnings.warn(
                f"d_model ({d_model}) is not divisible by n_heads ({n_heads}), and "
                f"n_heads * d_k ({n_heads * d_k}) does not cover it either."
            )
        if rope is not None and rope.d_head != d_k:
            raise ValueError(
                f"rope is built for d_head {rope.d_head}, but this block has d_k {d_k}."
            )
        self.rope = rope

        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if self.n_kv_heads < 1 or n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be >= 1 and divide n_heads "
                f"({n_heads}) so every query head belongs to exactly one group."
            )
        self.n_groups = n_heads // self.n_kv_heads          # query heads per KV head

        if window is not None:
            if window < 1:
                raise ValueError(f"window ({window}) must be >= 1.")
            if not mask:
                raise ValueError("window requires mask=True: it only narrows a causal band.")
        self.window = window
        self.flex = flex

        # QK-norm (Henry et al. 2020; used by ViT-22B, Gemma 2, OLMo 2): normalize each
        # head's query and key before the rotation. Attention logits are a dot product of
        # two unbounded vectors, and in low precision they are the first thing to blow up;
        # normalizing the operands bounds them without touching what the head attends to.
        # Applied before RoPE, so it scales the vector and the rotation then turns it.
        self.q_norm = RMSNorm(d_k) if qk_norm else None
        self.k_norm = RMSNorm(d_k) if qk_norm else None

        self.d_out = d_out if d_out is not None else d_model

        # Combine all heads into single projection weights for speed. K and V are only
        # n_kv_heads wide, which is where grouped-query attention gets its savings.
        self.W_q = initialize_normal_torch_weights(d_model, n_heads * d_k, std)
        self.W_k = initialize_normal_torch_weights(d_model, self.n_kv_heads * d_k, std)
        self.W_v = initialize_normal_torch_weights(d_model, self.n_kv_heads * d_v, std)

        # Every block adds its output into one residual stream, so without damping the
        # stream's variance grows with depth. GPT-2's fix: scale the projection that
        # writes into it by 1/sqrt(2 * n_layers). Opt-in, since it needs the depth.
        out_std = std / (2 * n_layers) ** 0.5 if n_layers else std
        self.W_O = initialize_normal_torch_weights(n_heads * d_v, self.d_out, out_std)

    def _use_flex(self, Q: torch.Tensor, T: int) -> bool:
        """Whether the block-sparse path beats a materialized mask for this call."""
        return (
            self.window is not None
            and self.flex
            and flex_attention is not None
            and Q.is_cuda                                   # on CPU flex is far slower
            and Q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and T >= _FLEX_MIN_QUERIES
        )

    def _grouped(self, Q, K, V):
        """
        Reshape for grouped attention without copying K/V.

        `repeat_interleave` would physically duplicate both n_groups times, which on the
        debug path is exactly when memory is tightest. Splitting Q's head dim into
        (n_kv_heads, n_groups) and giving K/V a length-1 group axis gets the same pairing
        by broadcast.
        """
        B, _, T, _ = Q.shape
        Q = Q.view(B, self.n_kv_heads, self.n_groups, T, self.d_k)
        return Q, K.unsqueeze(2), V.unsqueeze(2)

    def _attend(self, Q, K, V, T: int, S: int, offset: int, key_pos=None):
        """Scaled dot-product attention under whichever mask this block calls for."""
        # A rolling cache holds exactly the window and nothing else, so at decode every
        # stored key is attendable and no mask is needed -- which is also what keeps this
        # path on the flash backend, since flash refuses an explicit attn_mask.
        if key_pos is not None:
            gqa = {"enable_gqa": True} if self.n_groups > 1 else {}
            if T == 1:
                return torch.nn.functional.scaled_dot_product_attention(Q, K, V, **gqa)
            return torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=ring_mask(T, offset, key_pos, self.window), **gqa
            )

        if self._use_flex(Q, T):
            return _flex()(
                Q, K, V,
                block_mask=_block_mask(T, S, offset, self.window, Q.device),
                enable_gqa=self.n_groups > 1,
            )

        gqa = {"enable_gqa": True} if self.n_groups > 1 else {}
        if not self.use_mask:
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V, **gqa)

        # No window: the causal band has a mask-free spelling in every case. is_causal
        # aligns top-left, which is only right when the queries are the whole sequence;
        # past that causal_lower_right expresses the same band without materializing it.
        if self.window is None:
            if S == T:
                return torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V, is_causal=True, **gqa
                )
            if T == 1:                                      # decode: all of the prefix
                return torch.nn.functional.scaled_dot_product_attention(Q, K, V, **gqa)
            return torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=causal_lower_right(T, S), **gqa
            )

        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=causal_mask(T, S, offset, Q.device, self.window),
            **gqa,
        )

    def forward(
        self,
        Z: torch.Tensor,
        cache: "KVCache | None" = None,
        get_attention_list: bool = False,
    ):
        """
        Z: [B, T, d_model] -> [B, T, d_out].

        With a `cache`, Z holds only the new tokens: their keys and values are appended
        and attention runs against the whole stored prefix. Positions continue from
        `cache.pos`, so RoPE stays consistent across steps. Inference only -- the cache
        buffers are written in place and carry no gradient.
        """
        B, T, _ = Z.shape

        # 1. Linear projections: [B, T, n_(kv_)heads * d]
        # 2. Reshape & Transpose to batch heads: [B, n_(kv_)heads, T, d]
        Q = (Z @ self.W_q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = (Z @ self.W_k).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = (Z @ self.W_v).view(B, T, self.n_kv_heads, self.d_v).transpose(1, 2)

        if self.q_norm is not None:
            Q, K = self.q_norm(Q), self.k_norm(K)

        # Position enters here, on Q and K only: RoPE is a rotation of the query/key
        # planes, and V carries no position of its own.
        offset = cache.pos if cache is not None else 0
        if self.rope is not None:
            Q = self.rope(Q, offset=offset)
            K = self.rope(K, offset=offset)

        # Cache the rotated keys, so a token's key is rotated once, at its own position.
        # Only n_kv_heads of them, which is most of what GQA buys at inference time.
        key_pos = None
        if cache is not None:
            K, V = cache.update(K, V)
            if cache.window is not None:
                if cache.window != self.window:
                    raise ValueError(
                        f"cache rolls a {cache.window}-token window but this block "
                        f"attends over {self.window}."
                    )
                key_pos = cache.positions()
        S = K.shape[-2]

        if get_attention_list:
            # Fallback path if explicit attention maps are requested (Slower, heavy memory)
            Qg, Kg, Vg = self._grouped(Q, K, V)
            E = (Qg @ Kg.transpose(-2, -1)) / (self.d_k ** 0.5)
            if self.use_mask:
                allowed = (ring_mask(T, offset, key_pos, self.window) if key_pos is not None
                           else causal_mask(T, S, offset, Z.device, self.window))
                E = E.masked_fill(~allowed, float("-inf"))
            A = torch.nn.functional.softmax(E, dim=-1)
            out = (A @ Vg).reshape(B, self.n_heads, T, self.d_v)
            A = A.reshape(B, self.n_heads, T, S)

            # Reshape back to [B, T, n_heads * d_v]
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_v)
            out = out @ self.W_O

            # Split attention maps into a list of heads to preserve your API
            A_list = [A[:, i, :, :] for i in range(self.n_heads)]
            return A_list, out

        # Fast path: FlashAttention / memory-efficient SDPA, or the block-sparse flex
        # kernel when a window makes most of the matrix skippable.
        out = self._attend(Q, K, V, T, S, offset, key_pos)   # [B, n_heads, T, d_v]

        # Reshape back to [B, T, n_heads * d_v] and project output
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_v)
        return out @ self.W_O

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"d_k={self.d_k}, d_v={self.d_v}, causal={self.use_mask}, window={self.window}"
        )
