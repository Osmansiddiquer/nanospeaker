"""
GRPO for nanoSpeaker: group-relative advantages, PPO clip, no value net, no KL (v1).

The guards stand in for the KL term on this 4 GB card: an entropy floor, the
format tripwire (rollout format-reward collapse aborts), and 50-step checkpoints
for revert. Router health (moe_health) logs every step; the full 76-expert load
histogram dumps every --hist-every steps -- drift under RL is silent until it isn't.

    python -m src.rl.grpo --dry-run          # rollouts + rewards only, no updates
    python -m src.rl.grpo --steps 500        # real training (user-gated)
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.build_sft_t import load_facts
from ..eval.chat import render_chat
from ..eval.decode import load
from ..train.metrics import moe_health
from ..train.optim import build_optimizers
from .problems import arith_problem, load_code_bank, tool_problem
from .rewards import answer_reward, code_reward, shaped, tool_reward
from .rollout import rollout_group
from .sample import sample_batch


def current_logps_batch(model, device, seqs):
    """Per-token logprobs of out_ids under the CURRENT policy, plus the MoE aux loss.

    One left-padded forward for the whole group (k separate forwards was the second
    largest cost after generation). Returns (logps, aux): the aux load-balancing term
    MUST be added to the RL objective -- without it, policy gradient funnels tokens
    through whichever experts earned reward and the router collapses (measured twice:
    entropy 0.95 -> 0.63, ~56 of 76 experts unused, policy dead by step 25-31).
    """
    lens = [len(x["prompt_ids"]) + len(x["out_ids"]) for x in seqs]
    # Bucket the width to a multiple of 128: flex_attention is torch.compiled and
    # re-traces on every unseen shape. Variable rollout lengths were recompiling the
    # kernel each step -- the dominant cost, and invisible in GPU utilisation.
    width = ((max(lens) + 127) // 128) * 128
    rows = [[0] * (width - L) + x["prompt_ids"] + x["out_ids"]
            for x, L in zip(seqs, lens)]
    xt = torch.tensor(rows, device=device)
    with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
        # targets=xt makes the trunk return its aux term; the CE it also returns is
        # discarded -- only the router's balancing loss is wanted here.
        _, aux = model(xt, xt)
        logits = model(xt)
    out = []
    for i, x in enumerate(seqs):
        n = len(x["out_ids"])
        sl = logits[i, width - n - 1:width - 1].float()
        lp = F.log_softmax(sl, -1)
        idx = torch.tensor(x["out_ids"], device=device)
        out.append(lp.gather(-1, idx[:, None]).squeeze(-1))
    return out, aux


@torch.no_grad()
def probe(model, tok, device, held, k=4, max_new=320, temperature=0.9, batch=4):
    """pass@1 on a FIXED held-out set -- the actual learning curve.

    The per-step pass1 in the metrics is scored on 12 freshly drawn training
    problems, so its step-to-step swing is dominated by which problems were drawn,
    not by what the policy learned (measured: 0.16-0.53 adjacent steps with no
    trend over 18 updates). This set never changes and never trains, so its
    movement is signal. Generation is batched across prompts with no gradients
    resident, unlike rollout_group -- 4 x k4 = 16 sequences, deliberately below the
    48-sequence ceiling measured in sample.py: that ceiling was found with no
    optimizer resident, and Muon+AdamW moments for 382M parameters are live here.
    """
    was_training = model.training
    model.eval()
    hits = n = 0
    for i in range(0, len(held), batch):
        chunk = held[i:i + batch]
        rendered = [render_chat([{"role": "user", "content": p["prompt"]}]) for p in chunk]
        gens = sample_batch(model, tok, device, rendered, k, max_new, temperature)
        for prob, samples in zip(chunk, gens):
            for text in samples:
                hits += code_reward(text, prob["tests"])[0] >= 1.0
                n += 1
    if was_training:
        model.train()
    return hits / max(n, 1)


def grpo_step(model, optims, tok, device, problems, k, temperature, clip, max_new,
              chunk=2, aux_coef=0.02, dead=None):
    """One optimizer step over `problems`, each rolled out k ways.

    Backward runs PER GROUP (gradients accumulate) rather than once over every
    sequence: holding 24 sequence graphs at once OOMs a 4 GB card. Within a group
    the current-policy logprobs are computed in small padded chunks -- a compromise
    between one-at-a-time (slow) and all-at-once (also OOM).
    """
    # groups_used vs n_groups is the efficiency metric that matters: a group whose
    # k samples all pass or all fail has zero advantage variance and contributes NO
    # gradient, so the compute spent generating it is wasted. If this ratio is low
    # the fix is a harder/easier problem mix, not more steps.
    dead = dead if dead is not None else set()
    stats = {"reward": 0.0, "pass": 0, "n": 0, "stopped": 0, "loss": 0.0,
             "groups_used": 0, "groups_allfail": 0, "groups_allpass": 0}
    n_groups = max(len(problems), 1)
    for o in optims.values():
        o.zero_grad(set_to_none=True)
    for prob in problems:
        seqs = rollout_group(model, tok, device,
                             prob.get("messages", prob["prompt"]), k=k,
                             max_new=max_new, temperature=temperature)
        rewards = []
        for s in seqs:
            if prob["kind"] == "tool":
                r, _ = tool_reward(s["text"], prob["gold_call"], prob.get("tool_result"))
            elif prob["kind"] == "code":
                r, _ = code_reward(s["text"], prob["tests"])
            else:
                r, _ = answer_reward(s["text"], prob["answer"])
            stats["stopped"] += s["stopped"]
            r += shaped(s["out_ids"], s["stopped"], max_new)
            rewards.append(r)
            stats["reward"] += r
            stats["pass"] += r >= 1.0
            stats["n"] += 1
        rt = torch.tensor(rewards)
        if float(rt.std()) < 1e-6:
            stats["groups_allpass" if float(rt.mean()) >= 1.0 else "groups_allfail"] += 1
            # Retire it: a problem the policy solves 0/k or k/k times has no advantage
            # variance and cannot produce gradient. Measured pass@8 on this bank is
            # 0.444, so ~56% of every step's generation was buying nothing. Retiring
            # on first sight enriches the bank toward the learnable frontier for free
            # -- offline pre-screening of all 2,000 problems would cost ~27 GPU-hours.
            if prob.get("idx") is not None:
                dead.add(prob["idx"])
            continue                                 # no signal in this group
        stats["groups_used"] += 1
        # Dr. GRPO correction 1: centre only (no std division) -- dividing inflates
        # advantages on low-variance, already-easy questions.
        adv = rt - rt.mean()
        # Dr. GRPO correction 2: constant normalizer for the group, not per-sequence
        # token mean -- the latter makes long wrong answers cheap per token.
        norm = max(1.0, sum(len(s["out_ids"]) for s in seqs) / max(len(seqs), 1))
        live = [(x, a) for x, a in zip(seqs, adv.tolist()) if x["out_ids"]]
        for i in range(0, len(live), chunk):
            part = live[i:i + chunk]
            lps, aux = current_logps_batch(model, device, [x for x, _ in part])
            terms = []
            for (s, a), lp_new in zip(part, lps):
                ratio = torch.exp(lp_new - s["logps"].to(device))
                un = ratio * a
                cl = torch.clamp(ratio, 1 - clip, 1 + clip) * a
                terms.append(-torch.min(un, cl).sum() / norm)
            # policy term + router balancing: the fix for the collapse
            loss = torch.stack(terms).sum() / (len(live) * n_groups) + aux_coef * aux
            loss.backward()                          # free this chunk's graph now
            stats["loss"] += float(loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in optims.values():
        o.step()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/rl_base.pt")
    ap.add_argument("--out", default="runs/nanospeaker_rl")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--groups-per-step", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--lr-muon", type=float, default=5e-4)
    ap.add_argument("--lr-adamw", type=float, default=5e-5)
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--tier-up-at", type=float, default=0.7,
                    help="advance tier when pass@1 exceeds this")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--hist-every", type=int, default=25)
    ap.add_argument("--domain", default="arith", choices=["arith", "code", "mixed"])
    ap.add_argument("--aux-coef", type=float, default=0.02,
                    help="MoE load-balancing weight in the RL objective")
    ap.add_argument("--probe-every", type=int, default=10,
                    help="steps between held-out probes (0 disables)")
    ap.add_argument("--probe-n", type=int, default=24,
                    help="held-out code problems, withheld from training draws")
    ap.add_argument("--probe-k", type=int, default=4)
    ap.add_argument("--lr-router", type=float, default=None,
                    help="separate Muon lr for the MoE routers. Measured: 24.7%% of the "
                         "policy gradient's squared norm lands on 0.23%% of the "
                         "parameters (a ~108x concentration), so the lr that finally "
                         "moves the computation destroys routing first -- lr-muon 3e-3 "
                         "collapsed 60 of 76 experts permanently by step 16. Set this "
                         "BELOW --lr-muon to let the trunk learn while the router holds.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    import random
    from tokenizers import Tokenizer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file("tokenizer/tokenizer_chat.json")
    model, ckpt = load(Path(args.weights), device)
    facts = load_facts()
    rng = random.Random(0)

    if args.dry_run:
        model.eval()
        report = []
        for tier in (0, 1, 2, 3):
            probs = [arith_problem(rng, tier) for _ in range(2)]
            for p in probs:
                seqs = rollout_group(model, tok, device, p["prompt"], k=args.k,
                                     max_new=args.max_new, temperature=args.temperature)
                rs = [answer_reward(s["text"], p["answer"])[0] for s in seqs]
                report.append({"tier": tier, "prompt": p["prompt"],
                               "pass_at_k": max(rs), "mean_r": sum(rs) / len(rs)})
                print(json.dumps(report[-1]), flush=True)
        tp = tool_problem(rng, facts)
        seqs = rollout_group(model, tok, device, tp["prompt"], k=args.k,
                             max_new=args.max_new, temperature=args.temperature)
        rs = [tool_reward(s["text"], tp["gold_call"], tp.get("tool_result"))[0] for s in seqs]
        print(json.dumps({"tier": "tool", "pass_at_k": max(rs), "mean_r": sum(rs)/len(rs)}))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.out, "dry_run.json").write_text(json.dumps(report, indent=1))
        return

    model.train()
    optims, _ = build_optimizers(model)
    for g in optims["muon"].param_groups:
        g["lr"] = args.lr_muon
    for g in optims["adamw"].param_groups:
        g["lr"] = args.lr_adamw
    if args.lr_router is not None:
        # Routers live in ADAMW, not Muon: split_parameters() sends "embedding, norms
        # and router to AdamW" and only hidden matrices to Muon. So --lr-adamw has been
        # the router's learning rate all along -- raising it 10x is what collapsed the
        # router in the lr10 run, not --lr-muon. Search both optimizers rather than
        # assuming, and fail loudly if nothing matches.
        ids = {id(q) for n, q in model.named_parameters() if "router" in n}
        moved = 0
        for opt in (optims["adamw"], optims["muon"]):
            for g in list(opt.param_groups):
                routed = [q for q in g["params"] if id(q) in ids]
                if not routed:
                    continue
                g["params"] = [q for q in g["params"] if id(q) not in ids]
                opt.add_param_group({**{k: v for k, v in g.items() if k != "params"},
                                     "params": routed, "lr": args.lr_router})
                moved += len(routed)
        if not moved:
            raise SystemExit("--lr-router matched 0 tensors: refusing to run a job that "
                             "silently ignores the flag it was launched for")
        print(f"router lr decoupled: {moved} tensors at {args.lr_router} "
              f"(muon {args.lr_muon}, adamw {args.lr_adamw})", flush=True)
    if isinstance(ckpt, dict) and "muon" in ckpt:      # exact resume
        optims["muon"].load_state_dict(ckpt["muon"])
        optims["adamw"].load_state_dict(ckpt["adamw"])
        print(f"resumed optimizer state from step {ckpt.get('step')}", flush=True)
    out_dir = Path(args.out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics = open(out_dir / "metrics.jsonl", "a")
    tier = args.tier
    bad_streak = 0
    code_bank = load_code_bank() if args.domain in ("code", "mixed") else []
    held = []
    if code_bank and args.probe_every:
        # Tail slice, not a random sample: the split must be identical across runs
        # or probe numbers are not comparable between them.
        held, code_bank = code_bank[-args.probe_n:], code_bank[:-args.probe_n]
        print(f"held out {len(held)} problems; {len(code_bank)} trainable", flush=True)
    dead_path = out_dir / "dead_problems.json"
    dead = set(json.loads(dead_path.read_text())) if dead_path.exists() else set()
    live_ids = [i for i in range(len(code_bank)) if i not in dead]
    if dead:
        print(f"loaded {len(dead)} retired, {len(live_ids)} live problems", flush=True)
    for step in range(args.steps):
        t0 = time.perf_counter()
        problems = []
        if code_bank and not live_ids:
            print("EXHAUSTED: every trainable problem retired as zero-variance", flush=True)
            break
        for _ in range(args.groups_per_step):
            if args.domain == "code" or (args.domain == "mixed" and rng.random() < 0.75):
                j = rng.choice(live_ids)
                problems.append(dict(kind="code", idx=j, **code_bank[j]))
            else:
                problems.append(arith_problem(rng, tier))
        n_dead_before = len(dead)
        stats = grpo_step(model, optims, tok, device, problems, args.k,
                          args.temperature, args.clip, args.max_new,
                          aux_coef=args.aux_coef, dead=dead)
        if len(dead) > n_dead_before:
            live_ids = [i for i in live_ids if i not in dead]
            dead_path.write_text(json.dumps(sorted(dead)))
        pass1 = stats["pass"] / max(stats["n"], 1)
        row = dict(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   step=step, tier=tier, pass1=round(pass1, 3),
                   mean_reward=round(stats["reward"] / max(stats["n"], 1), 3),
                   loss=round(stats.get("loss", 0.0), 6),
                   live_problems=len(live_ids),
                   groups_used=stats["groups_used"],
                   groups_allfail=stats["groups_allfail"],
                   groups_allpass=stats["groups_allpass"],
                   step_s=round(time.perf_counter() - t0, 1))
        row.update(moe_health(model))
        if held and step % args.probe_every == 0:
            t1 = time.perf_counter()
            try:
                row["probe_pass1"] = round(probe(model, tok, device, held, args.probe_k,
                                                 args.max_new, args.temperature), 3)
            except torch.cuda.OutOfMemoryError:
                # The probe is a measurement, not the experiment. Losing it costs one
                # point on a curve; letting it abort a 7-hour run costs the run.
                torch.cuda.empty_cache()
                row["probe_pass1"] = None
                print(f"probe OOM at step {step} -- skipped, training continues",
                      flush=True)
            row["probe_s"] = round(time.perf_counter() - t1, 1)
        if step % args.hist_every == 0:
            counts = []
            for block in model.blocks:
                c = getattr(block.ffn, "expert_counts", None)
                if c is not None:
                    counts.append(c.tolist())
            (out_dir / f"expert_hist_{step:05d}.json").write_text(json.dumps(counts))
        stop_rate = stats["stopped"] / max(stats["n"], 1)
        row["stop_rate"] = round(stop_rate, 3)
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        print(row, flush=True)
        # Collapse guards, calibrated for RL and for RECOVERY. A single bad update
        # is not collapse: attempt-5 hit entropy 0.652 with 55 dead experts at step 6
        # and fully recovered by step 7 (0.922) -- a one-shot trip would have killed a
        # healthy run. Only abort when the router stays bad for 3 consecutive updates.
        entropy = row.get("router_entropy", 1.0)
        unused = row.get("experts_unused", 0.0)
        bad = stop_rate < 0.4 or entropy < 0.75 or unused > 25
        bad_streak = bad_streak + 1 if bad else 0
        if bad_streak >= 3:
            print(f"GUARD TRIPPED at step {step}: 3 consecutive bad updates "
                  f"(stop_rate={stop_rate:.2f} entropy={entropy:.3f} unused={unused:.1f})",
                  flush=True)
            break
        if args.domain == "arith" and pass1 > args.tier_up_at and tier < 3:
            tier += 1
            print(f"CURRICULUM: tier -> {tier}", flush=True)
        if step and step % args.save_every == 0:
            # Optimizer state included so a resume is EXACT: without it Muon/AdamW
            # moments restart at zero and the first few resumed updates are noisy.
            # config MUST ride along: decode.load() rebuilds the model from it, so a
            # checkpoint without it is unloadable by bench, chat, and --auto-resume
            # alike -- the RL loop never noticed because it only ever loaded rl_base.
            torch.save(dict(step=step, config=ckpt["config"],
                            model=model.state_dict(), tier=tier,
                            muon=optims["muon"].state_dict(),
                            adamw=optims["adamw"].state_dict()),
                       out_dir / "checkpoints" / f"rl_{step:05d}.pt")
            for old in sorted((out_dir / "checkpoints").glob("rl_*.pt"))[:-1]:
                old.unlink()


if __name__ == "__main__":
    main()
