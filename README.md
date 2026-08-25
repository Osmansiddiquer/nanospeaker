# nanoSpeaker

A 382M-parameter sparse-MoE decoder (73M active per token) trained end to end on a
single **4 GB RTX 3050 Ti laptop GPU** — pretraining, context extension, SFT,
continued pretraining and RL, all on one card with 4 GB of VRAM.

The constraint drove most of the design. Where a technique needed memory we did not
have, it was replaced rather than scaled down: custom Triton kernels instead of
`grouped_mm`, gradient accumulation instead of large batches, a ring KV cache instead
of a growing one. Where a measurement was cheap, it was taken instead of assumed.

---

## Architecture

| | |
|---|---|
| Layers | 20 |
| `d_model` | 576 |
| Attention | GQA — 9 query heads, 3 KV heads, `d_head` 64, QK-norm |
| Attention span | 3:1 sliding-window (512) to global, `global_every=4` |
| Experts | 76 routed + 2 shared, top-6, `d_expert` 128, dropless |
| Position | RoPE, base 10 000 → 500 000 for context extension |
| Context | 1024 → 8192 |
| Vocab | 32 768, tied embeddings |
| Precision | bf16 |
| Optimizers | Muon (hidden matrices) + AdamW (embeddings, norms, **routers**) |

Depth over width: at a fixed parameter budget the extra layers bought more than extra
width. Expert granularity `G = d_ff_dense / d_expert ≈ 16` sits in the band the
fine-grained MoE scaling laws recommend for a model this size.

**A detail that matters and is easy to miss:** `split_parameters()` sends embeddings,
norms **and routers** to AdamW — only hidden matrices go to Muon. `--lr-adamw` is
therefore the router learning rate. Assuming otherwise cost this project several hours
of confounded experiments (see [Results](#results)).

### Custom Triton kernels

`src/model/moe_kernel.py` implements the dropless MoE forward and backward as four
kernels tuned for **sm_86**. `torch._grouped_mm` is unusable on this card, so the
gather/scatter expert dispatch is hand-written. `src/model/kv_cache.py` implements a
**ring** cache for the windowed layers: a token older than `window` steps can never be
attended again, so its slot is reused instead of retained.

---

## Repository layout

```
src/model/     nanospeaker.py   the decoder stack and NanoSpeakerConfig
               moe.py           dropless MoE block, routing, shared experts
               moe_kernel.py    Triton kernels A–D + backward (sm_86)
               mha.py           GQA with QK-norm, windowed and global attention
               kv_cache.py      ring cache for windowed layers, chunked prefill
               rope.py, ln.py, glu.py

src/data/      fetch_*.py       corpus acquisition (streaming, resumable)
               tokenize_corpus.py, shards.py, loader.py
               train_tokenizer.py, make_chat_tokenizer.py, retag_tokenizer.py
               build_sft.py     chat SFT corpus (ChatML, response-masked)
               build_sft_t.py   think-span variant
               build_arith*.py  arithmetic curricula, three difficulty tiers
               build_think_cmd.py  "think when told to think", decoupled from the word
               build_reason_cpt.py reasoning-CPT blend assembly
               build_identity.py, repack_sft.py, prefetch.py

src/train/     train.py         pretraining loop, WSD schedule, resumable at any step
               sft.py           supervised fine-tuning, packed rows, row-head masking
               optim.py         Muon, parameter splitting, WSD schedule
               metrics.py       moe_health: router entropy, expert load CV, unused
               dashboard.py

src/eval/      bench.py         frozen benchmark runner → runs/bench.jsonl
               chat.py          REPL + HTTP server, ChatML rendering
               decode.py        loading, sampling, streaming, stop handling

src/rl/        grpo.py          Dr. GRPO — group-relative advantages, no value net
               rollout.py       on-policy group rollouts, keeps old-policy logprobs
               sample.py        offline batched sampler (pass@k probes, ReST-EM)
               rewards.py       answer / code / tool verifiers
               problems.py      arithmetic tiers, code bank, tool-call problems

scripts/       phase2*.sh       the pretraining phases as run
               rl_run.sh        GRPO with the configuration this project settled on
               resume_after_reboot.sh   --check reports; no argument launches
```

---

## Data

Every corpus is streamed, tokenized to flat `uint16` `.bin` files, and mixed at
training time by `src/data/loader.py`. Nothing here is redistributed — the fetch
scripts pull from the original sources.

### Raw corpora

| corpus | source | tokens |
|---|---|---|
| `web` | `openbmb/Ultra-FineWeb` | 1 764M |
| `python` | `HuggingFaceTB/stack-edu` (Python subset) | 754M |
| `math` | `open-web-math/open-web-math` | 225M |
| `code_instruct` | `nvidia/OpenCodeInstruct` | 110M |
| `finemath3` | `HuggingFaceTB/finemath` | 110M |
| `nemomath` | Nemotron math | 110M |
| `simplewiki` | `wikimedia/wikipedia` (Simple English) | 77M |
| `qa` | mixed QA | 65M |
| `owm` | `open-web-math` (held-out slice) | 59M |
| `codereason` | OpenCodeReasoning | 45M |
| `sci_mot`, `sci_mega` | science reasoning | 25M each |

Each has a matching `valid_*.bin` split (2–10M tokens) that never enters training.

Chat and instruction data comes from `HuggingFaceTB/smoltalk` and
`NousResearch/hermes-function-calling-v1`, rendered to ChatML by `build_sft.py`.

### Derived corpora

| set | built by | contents |
|---|---|---|
| `data/sft` | `build_sft.py` | 248M tokens, ~56% supervised — prose, tools, math, code, identity |
| `data/sft_t` | `build_sft_t.py` | think-span variant, `<\|think\|>` … `<\|/think\|>` |
| `data/cpt` | `build_reason_cpt.py` | 250M-token reasoning blend, raw + chat rows in one stream |
| `data/rl/code_bank.jsonl` | | 2 000 OpenCodeInstruct problems whose reference solutions were **verified to pass their own unit tests** |
| `data/evals` | `fetch_evals.py` | 16 frozen benchmark files with a sha256 manifest |

### Tokenizer

32 768-vocab BPE trained on the corpus (`train_tokenizer.py`). Four donor tokens were
renamed for chat rather than added, so the embedding matrix is unchanged:

```
<|im_start|> = 2    <|im_end|> = 3    <|think|> = 4    <|/think|> = 1
```

### Two data lessons worth carrying

**Packed rows are not cold-start rows.** SFT rows are packed, so a conversation almost
never begins at position 0. A bare position-0 `<|im_start|>` is out of distribution and
collapses generation — measured `P(<|think|>)` of 0.55 packed against 0.0000 cold.
`render_chat()` therefore leads every conversation with `<|endoftext|>`.

**Copy-inflated loss.** Scratchpads that restate their operands let copy-heads pay the
rent, so teacher-forced loss falls without the model learning arithmetic. Only
generative evals (`arith_gen`) are trustworthy on this axis.

---

## Evaluation

`src/eval/bench.py` runs a frozen suite against any checkpoint and appends to
`runs/bench.jsonl`. Multiple-choice is scored by length-normalized logprob; generation
tasks execute in a sandbox (`RLIMIT_AS` 1 GB, `RLIMIT_CPU` 5 s).

```bash
python -m src.eval.bench --weights <ckpt> --mode chat --limit 100 \
  --tasks humaneval,arith_gen,arith_gen_hard,tools_heldout,format_compliance
```

**`--limit 100` is mandatory.** The eval files hold more items than the lineage was
scored on (humaneval 164, arith_gen 200, tools_heldout 200) and the flag defaults to
no truncation. Omitting it silently scores three of five gates on a different item set
than every checkpoint already in `bench.jsonl`, producing numbers that look valid and
compare to nothing.

`--dump-items <dir>` writes per-item pass/fail verdicts. Comparing *which* problems
fail is far more informative than comparing totals — it is what revealed that a whole
RL run had left the weights effectively unchanged (Jaccard 1.000 against baseline).

---

## Results

Best checkpoint, chat mode, `--limit 100`:

| task | score |
|---|---|
| `arith_gen` | 0.74 |
| `arith_gen_hard` | 0.50 |
| `tools_heldout` | 0.89 |
| `format_compliance` | 0.86 |
| `identity_gen` | 0.80 |
| `humaneval` | 0.06 |

This is a small model and the numbers are small. It follows chat structure, calls tools
against a schema, and does multi-step arithmetic; it is not a coding model.

### The RL result is a documented null

Six GRPO runs, ~20 GPU-hours. Across 200 benchmark problems, thirty steps of
correctly-configured GRPO **fixed 3 and broke 4** — real behavioural change, zero net
gain, every gate statistically indistinguishable from the starting checkpoint.

The mechanism: policy gradient concentrates **~108×** on the MoE routers (24.7% of the
gradient's squared norm on 0.23% of the parameters). Decoupling the router learning
rate works exactly as designed — trunk learns at 98% speed, routers move at 6% — and
the routing collapses anyway. The cause is **representation drift**: the trunk learns,
hidden states move, and a near-static router applied to drifted features funnels tokens
into a handful of experts. Runs died at step 16 and step 39; pinning the routers buys
2.4× more steps and nothing else.

Also worth knowing before repeating it: at `--lr-muon 3e-4` the model was not training
at all — 4.2e-06 relative weight change per step. Several earlier "flat RL results" in
this project were that tuning failure misread as evidence.

---

## Running it

```bash
pip install torch triton tokenizers datasets numpy

python -m src.data.fetch_streaming --corpus web     # acquire (streaming, resumable)
python -m src.data.tokenize_corpus  --corpus web

python -m src.train.train --out runs/base --steps 8691 \
  --micro-batch 12 --accum 12 --seq-len 1024 \
  --stable-mix '{"web":0.60,"python":0.30,"math":0.10}'

python -m src.train.sft --data-dir data/sft --auto-resume
python -m src.eval.bench --weights runs/base/model.pt --mode chat --limit 100
python -m src.eval.chat  --weights runs/base/model.pt          # REPL
```

Every loop is resumable from any step; checkpoints carry optimizer state so a resume is
exact rather than approximate.

---

## License

MIT — see [LICENSE](LICENSE).

Model weights and training corpora are **not** in this repository. The corpora belong to
their original publishers and carry their own licenses; the fetch scripts pull from
those sources directly.
