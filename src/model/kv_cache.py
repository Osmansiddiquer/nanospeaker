"""Key/value cache for incremental (autoregressive) decoding."""

import torch


class KVCache:
    """
    Per-layer key/value cache: one of these per attention block, held by the caller.

    Generation re-runs the whole prefix for every new token, and with causal attention
    the keys and values of a token never change once computed. Storing them turns each
    decode step from O(T^2) into O(T): the block projects K and V for the new token only,
    `update` appends them, and attention runs against the full stored prefix.

    The buffers are allocated once, at `max_seq_len`, on the first `update` -- which is
    also where the batch size, head count, widths, dtype and device come from, so
    construction needs nothing but the length. Each step then writes into a slice instead
    of allocating and concatenating, keeping the address of the cache fixed (which is
    what CUDA graph capture and torch.compile want) at the cost of holding the full
    context up front.

    This is a plain object, not an nn.Module: it is decoding state, not model state, and
    has no business in a state_dict.

        cache = KVCache(max_seq_len=1024)
        logits = model(prompt, cache=cache)          # prefill, T tokens
        logits = model(next_token, cache=cache)      # decode, T = 1

    Attributes:
        pos: tokens currently stored, i.e. the position of the next token to be written.
    """

    def __init__(
        self, max_seq_len: int, window: "int | None" = None, max_chunk: int = 1
    ):
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be >= 1.")
        if window is not None and window < 1:
            raise ValueError(f"window ({window}) must be >= 1.")
        if max_chunk < 1:
            raise ValueError(f"max_chunk ({max_chunk}) must be >= 1.")

        self.max_seq_len = max_seq_len
        self.window = window
        self.max_chunk = max_chunk

        # With a window, a token older than `window` steps can never be attended again,
        # so there is no reason to hold it: the buffer is sized to the window and reused
        # as a ring. That turns decode from O(context) memory and attention into
        # O(window), which is the whole point of a sliding window at inference.
        # A T-token step needs the oldest query's whole window still resident, hence
        # window + max_chunk - 1 slots. max_chunk stays 1 unless asked: silently losing
        # keys on a multi-token step would be far worse than refusing to serve it.
        self.capacity = (
            min(max_seq_len, window + max_chunk - 1) if window else max_seq_len
        )
        self.pos = 0
        self.k: "torch.Tensor | None" = None
        self.v: "torch.Tensor | None" = None

        # Which absolute position sits in which slot is pure arithmetic (see
        # `positions`), so the ring keeps no side table: at decode every stored key is
        # attendable and nothing needs it, and paying a few extra kernels per step to
        # maintain one is not free when a launch costs more than the attention does.

    def _allocate(self, k: torch.Tensor, v: torch.Tensor):
        """Empty [B, n_heads, capacity, d] buffers shaped after the first step's k/v."""
        B, H, _, d_k = k.shape
        d_v = v.shape[-1]
        return (
            torch.empty(B, H, self.capacity, d_k, dtype=k.dtype, device=k.device),
            torch.empty(B, H, self.capacity, d_v, dtype=v.dtype, device=v.device),
        )

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Append this step's [B, n_heads, T, d] keys/values and return the whole prefix.

        The returned tensors are views of the buffers, valid until the next update.
        RoPE must already have been applied to `k`: what is cached is the final key.
        """
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"expected [B, n_heads, T, d] keys and values, got {tuple(k.shape)} "
                f"and {tuple(v.shape)}."
            )
        if k.shape[:3] != v.shape[:3]:
            raise ValueError(
                f"keys {tuple(k.shape)} and values {tuple(v.shape)} disagree on "
                f"batch/heads/length."
            )

        T = k.shape[-2]
        if self.window is None and self.pos + T > self.max_seq_len:
            raise ValueError(
                f"cache holds {self.max_seq_len} tokens; {self.pos} stored and {T} more "
                f"would overflow it."
            )
        if self.window is not None and T > 1 and self.capacity < self.window + T - 1:
            raise ValueError(
                f"a {T}-token step against a {self.window}-token window needs a ring of "
                f"at least {self.window + T - 1}, but this cache holds {self.capacity}. "
                f"Build it with max_chunk >= {T}, or prefill without a cache and decode "
                f"one token at a time."
            )
        # The two buffers are allocated and replaced together, so they are read out into
        # locals: one None check then covers both, for the reader and the type checker.
        if self.k is None or self.v is None:
            self.k, self.v = self._allocate(k, v)
        elif k.shape[:2] + k.shape[3:] != self.k.shape[:2] + self.k.shape[3:]:
            raise ValueError(
                f"cache was allocated for {tuple(self.k.shape)}-shaped keys, got "
                f"{tuple(k.shape)}. Use a fresh cache per sequence batch."
            )
        k_buf, v_buf = self.k, self.v

        if self.window is not None:
            if T == 1:
                # The decode path, kept to two copies and no index tensors: at this size
                # a kernel launch costs more than the attention it feeds.
                slot = self.pos % self.capacity
                k_buf[:, :, slot : slot + 1].copy_(k)
                v_buf[:, :, slot : slot + 1].copy_(v)
            else:
                # Only the last `capacity` incoming tokens can survive, so a step longer
                # than the ring is trimmed to its tail before scattering.
                positions = torch.arange(self.pos, self.pos + T, device=k.device)
                if T > self.capacity:
                    k, v = k[:, :, -self.capacity :], v[:, :, -self.capacity :]
                    positions = positions[-self.capacity :]
                slots = positions % self.capacity
                k_buf.index_copy_(2, slots, k)
                v_buf.index_copy_(2, slots, v)
            self.pos += T

            # Once wrapped every slot is live; before that only the written prefix is.
            filled = min(self.pos, self.capacity)
            return k_buf[:, :, :filled], v_buf[:, :, :filled]

        # copy_ rather than slice assignment: same write, but explicit that the buffer
        # owns the memory and nothing here is autograd-tracked bookkeeping.
        k_buf[:, :, self.pos : self.pos + T].copy_(k)
        v_buf[:, :, self.pos : self.pos + T].copy_(v)
        self.pos += T
        return k_buf[:, :, : self.pos], v_buf[:, :, : self.pos]

    def reset(self) -> None:
        """Forget the sequence, keeping the buffers for the next one."""
        self.pos = 0

    def positions(self) -> "torch.Tensor | None":
        """
        Absolute position of each live slot, in buffer order (a ring stores out of it).

        Derived, not recorded: the live positions are [pos - filled, pos), and position p
        always lives in slot p % capacity, so slot j holds the one such p congruent to j.
        Only a multi-token step needs this -- decode attends to every slot unmasked.
        """
        if self.k is None:
            return None
        filled = min(self.pos, self.capacity)
        start = self.pos - filled
        j = torch.arange(filled, device=self.k.device)
        return start + (j - start) % self.capacity

    def reorder(self, index: torch.Tensor) -> None:
        """
        Permute/gather the batch dim by `index`, for beam search and similar.

        Beams are pruned and duplicated between steps, so each surviving beam has to be
        paired with the cache of whichever beam it grew from.
        """
        if self.k is None or self.v is None:
            return
        index = index.to(self.k.device)
        self.k = self.k.index_select(0, index).contiguous()
        self.v = self.v.index_select(0, index).contiguous()

    def __len__(self) -> int:
        return self.pos

    def __repr__(self) -> str:
        shape = tuple(self.k.shape) if self.k is not None else "unallocated"
        return f"KVCache(pos={self.pos}/{self.max_seq_len}, keys={shape})"


def make_kv_caches(
    n_layers: int, max_seq_len: int, window: "int | None" = None, max_chunk: int = 1
) -> "list[KVCache]":
    """One cache per layer, to thread through a stack of attention blocks."""
    return [KVCache(max_seq_len, window, max_chunk) for _ in range(n_layers)]
