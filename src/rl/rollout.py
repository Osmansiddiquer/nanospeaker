"""
Batched group rollouts for GRPO.

A GRPO group is k samples of ONE prompt, so the group is the batch: prefill the
prompt once as [k, T] identical rows, then all k sequences decode in lockstep with
temperature sampling. Per-token logprobs of the sampled tokens are collected during
generation (the old-policy side of the ratio). Sequences that hit a stop id are
frozen in place (their further tokens ignored) until the whole group finishes.

Chat template comes from render_chat -- the eos-prefix and turn structure are load
bearing (cold-start lesson); think spans decode as their special ids.
"""
import torch
import torch.nn.functional as F

from ..eval.chat import CHAT_STOP_IDS, render_chat


@torch.no_grad()
def rollout_group(model, tok, device, messages, k=8, max_new=300, temperature=0.9,
                  min_tokens=4):
    prompt = render_chat(messages if isinstance(messages, list)
                         else [{"role": "user", "content": messages}])
    ids = tok.encode(prompt).ids
    # NO left-padding here: render_chat already leads with <|endoftext|>, and extra
    # eos padding shifts every RoPE position, putting the prompt out of distribution.
    # Measured cost of trying: rollout pass@1 0.25 -> 0.08. Recompiles are cheaper.
    T = len(ids)
    total = max(T + max_new + 1, 512 + T)
    caches = model.caches(total, max_chunk=T)
    x = torch.tensor([ids], device=device).expand(k, T).contiguous()
    with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
        logits = model(x, caches=caches)[:, -1]

    out = torch.zeros(k, max_new, dtype=torch.long, device=device)
    logps = torch.zeros(k, max_new, device=device)
    alive = torch.ones(k, dtype=torch.bool, device=device)
    lens = torch.zeros(k, dtype=torch.long, device=device)
    stop_ids = torch.tensor(sorted(CHAT_STOP_IDS), device=device)

    for step in range(max_new):
        lg = logits.float() / max(temperature, 1e-6)
        if step < min_tokens:
            lg[:, list(CHAT_STOP_IDS)] = -float("inf")
        # never close an unopened think span
        opened = (out[:, :step] == 4).any(-1)
        lg[~opened, 1] = -float("inf")
        probs = F.softmax(lg, -1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)              # [k]
        logp = torch.log(probs.gather(-1, nxt[:, None]).squeeze(-1) + 1e-12)
        hit_stop = (nxt[:, None] == stop_ids[None, :]).any(-1)
        record = alive & ~hit_stop
        out[record, step] = nxt[record]
        logps[record, step] = logp[record]
        lens[record] += 1
        alive = alive & ~hit_stop
        if not alive.any():
            break
        # frozen sequences keep stepping with a harmless token; masked from records
        nxt = torch.where(alive, nxt, torch.zeros_like(nxt))
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            logits = model(nxt[:, None], caches=caches)[:, -1]

    seqs = []
    for i in range(k):
        n = int(lens[i])
        toks = out[i, :n].tolist()
        seqs.append({
            "prompt_ids": ids,
            "out_ids": toks,
            "logps": logps[i, :n].detach().cpu(),
            "stopped": bool(n < max_new),
            "text": tok.decode(toks, skip_special_tokens=False),
        })
    return seqs
