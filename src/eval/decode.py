"""
Shared inference harness: checkpoint loading, device pick, and the sampling loop.

Per-token controls, in the order applied: repetition penalty (CTRL-style, on
generated ids only), temperature, top-k, top-p. Stop strings end the generation
the moment one appears in the decoded text.
"""

import subprocess
from pathlib import Path

import torch

from ..model.nanospeaker import NanoSpeaker, NanoSpeakerConfig

INSTRUCT = "### Instruction\n{}\n\n### Response\n"
NEED_MIB = 1536      # 0.73 GB bf16 weights + prefill activations + KV (~16 MB) + slack


def pick_device(requested: "str | None" = None) -> str:
    """Use the GPU only when it has room. Asked via nvidia-smi, not torch.cuda --
    initializing a CUDA context just to probe would itself take memory from a
    live training run on this 4 GB card."""
    if requested:
        return requested
    try:
        free = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.split()[0])
    except Exception:
        return "cpu"
    if free < NEED_MIB:
        print(f"gpu has {free} MiB free, need ~{NEED_MIB} (training run?) -- using cpu")
        return "cpu"
    return "cuda"


def load(weights: Path, device: str):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = NanoSpeakerConfig(**ck["config"])
    model = NanoSpeaker(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    if device == "cuda":
        model = model.to(torch.bfloat16)
    return model, ck


def _filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k:
        logits[logits < logits.topk(top_k, dim=-1).values[:, -1:]] = -float("inf")
    if 0.0 < top_p < 1.0:
        srt, idx = logits.sort(-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum > top_p
        drop[:, 1:] = drop[:, :-1].clone()   # keep the token that crosses the line
        drop[:, 0] = False
        srt[drop] = -float("inf")
        logits = torch.full_like(logits, -float("inf")).scatter_(-1, idx, srt)
    return logits


def stream(model, tok, device, prompt: str, n: int, temperature: float = 0.3,
           top_k: int = 40, top_p: float = 1.0, rep_penalty: float = 1.0,
           stop: tuple = (), stop_ids: frozenset = frozenset(),
           min_tokens: int = 0):
    """Yield one decoded text delta per sampled token (possibly empty mid-rune)."""
    # Clamp to the checkpoint's trained length -- RoPE grows past it but quality doesn't.
    max_ctx = model.cfg.max_seq_len
    ids = tok.encode(prompt).ids[-(max_ctx - 1):]
    n = min(n, max_ctx - len(ids))
    # Windowed layers ring at window + max_chunk - 1 slots (kv_cache.py); the prompt
    # prefills as ONE chunk, so max_chunk must cover it and the capacity floor
    # (min'd against max_seq_len) must reach window + prompt - 1.
    total = len(ids) + n + 1
    max_w = max((w for w in model.cfg.layer_windows() if w), default=0)
    if max_w:
        total = max(total, max_w + len(ids))
    caches = model.caches(total, max_chunk=max(len(ids), 1))
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device), caches=caches)[:, -1]
        out, text = [], ""
        for _ in range(n):
            logits = logits.float()
            if rep_penalty != 1.0 and out:
                # Generated ids only -- penalizing the prompt would tax the very
                # identifiers a completion is supposed to reuse.
                seen = torch.tensor(sorted(set(out)), device=logits.device)
                picked = logits[0, seen]
                logits[0, seen] = torch.where(picked > 0, picked / rep_penalty,
                                              picked * rep_penalty)
            logits = _filter(logits / max(temperature, 1e-6), top_k, top_p)
            # Serving guard against the trained early-stop attractor: no stop ids
            # before min_tokens, and never close a think span that never opened.
            if len(out) < min_tokens:
                for s in stop_ids:
                    logits[0, s] = -float("inf")
            if 4 not in out:
                logits[0, 1] = -float("inf")
            nxt = (torch.multinomial(logits.softmax(-1), 1) if temperature > 0
                   else logits.argmax(-1, keepdim=True))
            if int(nxt) in stop_ids:     # e.g. <|im_end|>/<|endoftext|> -- never emitted
                return
            out.append(int(nxt))
            # Decode the whole tail each step and emit the delta: decoding tokens
            # one at a time splits multi-token BPE byte sequences. Hold back while
            # the tail ends in an unresolved partial rune. Think spans render
            # visibly -- a hidden scratchpad is undebuggable.
            full = (tok.decode(out, skip_special_tokens=False)
                    .replace("<|think|>", "〔think: ").replace("<|/think|>", "〕 ")
                    .replace("<|im_end|>", "").replace("<|endoftext|>", ""))
            cut = min((i for i in (full.find(s) for s in stop) if i >= 0), default=-1)
            if cut >= 0:
                if cut > len(text):
                    yield full[len(text):cut]
                return
            if full.startswith(text):
                yield full[len(text):]
                text = full
            else:
                yield ""
            logits = model(nxt, caches=caches)[:, -1]


def generate(model, tok, device, prompt: str, n: int, **kw) -> str:
    """Non-streaming convenience over stream()."""
    return "".join(stream(model, tok, device, prompt, n, **kw))
