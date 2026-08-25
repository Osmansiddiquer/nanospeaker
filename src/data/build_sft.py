"""
Build the SFT corpus: ChatML-rendered conversations -> packed token+mask streams.

Every conversation becomes `<|im_start|>role\\ncontent<|im_end|>` turns (specials from
tokenizer_chat.json) followed by <|endoftext|>. The loss mask is built during
rendering: 1 on assistant content AND its closing <|im_end|> (stopping is supervised),
0 on system/user/tool turns and on the assistant header -- the model is never paid for
writing the user's side, which is exactly how the phase-1 next-exam-problem tic dies.

Sources:
  smoltalk (HF stream, per-subset caps)  prose register + math CoT, turn structure kept
  tools (local Hermes shards, ChatML text already; parsed back into turns, upsampled)
  code_instruct (local shards, template parsed back into a single user/assistant turn)

Output: data/sft/train.bin (uint16) + train.mask (uint8) packed continuously in
globally shuffled order, per-source valid_*.bin/.mask, and manifest.json.

    python -m src.data.build_sft            # full build (~30-60 min, network)
"""
import glob
import io
import json
import random
from pathlib import Path

import numpy as np
import zstandard
from tokenizers import Tokenizer

TOK = Tokenizer.from_file("tokenizer/tokenizer_chat.json")
IM_START, IM_END, EOS = 2, 3, 0
MAX_CONV_TOKENS = 4096          # longer than a training row cannot pack; drop (rare)
VALID_PER_SOURCE = 500          # conversations held out per source group

# subset -> (group, cap). Caps balance the mix toward ~190M tokens total; magpie is
# the register backbone, numina/metamath the reasoning substrate.
SMOLTALK = {
    "smol-magpie-ultra": ("prose", 50_000),
    "openhermes-100k": ("prose", 100_000),
    "smol-summarize": ("prose", 30_000),
    "smol-rewrite": ("prose", 50_000),
    "explore-instruct-rewriting": ("prose", 30_000),
    "everyday-conversations": ("prose", 5_000),
    "metamathqa-50k": ("math", 50_000),
    "numina-cot-100k": ("math", 40_000),
}
TOOLS_UPSAMPLE = 4              # 1,241 Hermes convs are thin; repeat, watch valid_tools


def enc(s: str) -> list:
    return TOK.encode(s).ids


def render(messages) -> "tuple[list, list] | None":
    """One conversation -> (ids, mask). None if empty/overlong/malformed."""
    ids, mask = [], []
    saw_assistant = False
    for m in messages:
        role, content = m.get("role", ""), (m.get("content") or "").strip()
        if not content or role not in ("system", "user", "assistant", "tool"):
            continue
        if role == "assistant":
            saw_assistant = True
            head = enc("assistant\n")
            body = enc(content)
            ids += [IM_START] + head + body + [IM_END]
            mask += [0] * (1 + len(head)) + [1] * len(body) + [1]
        else:
            turn = enc(f"{role}\n{content}")
            ids += [IM_START] + turn + [IM_END]
            mask += [0] * (len(turn) + 2)
    if not saw_assistant or len(ids) + 1 > MAX_CONV_TOKENS:
        return None
    ids.append(EOS)
    mask.append(0)
    return ids, mask


def parse_chatml_text(text: str):
    """Hermes shards are already ChatML text; recover the turn list."""
    msgs = []
    for chunk in text.split("<|im_start|>")[1:]:
        chunk = chunk.split("<|im_end|>")[0]
        role, _, content = chunk.partition("\n")
        msgs.append({"role": role.strip(), "content": content.strip()})
    return msgs


def iter_zst(pattern: str):
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            r = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f),
                                 encoding="utf-8")
            for line in r:
                yield json.loads(line)


def main() -> None:
    out_dir = Path("data/sft")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {}                                  # group -> list of (ids, mask)

    def add(group, rendered):
        if rendered:
            groups.setdefault(group, []).append(rendered)

    # --- local: tools (Hermes function calling, pre-rendered ChatML text) ------------
    for row in iter_zst("data/raw/tools/*.jsonl.zst"):
        add("tools", render(parse_chatml_text(row["text"])))
    print(f"tools: {len(groups.get('tools', [])):,} conversations")

    # --- local: code_instruct (parse the frozen template back into one exchange) -----
    kept = 0
    for row in iter_zst("data/raw/code_instruct/*.jsonl.zst"):
        text = row["text"]
        if not text.startswith("### Instruction\n"):
            continue
        inp, sep, out = text[len("### Instruction\n"):].partition("\n\n### Response\n")
        if not sep or not out.strip():
            continue
        add("code", render([{"role": "user", "content": inp},
                            {"role": "assistant", "content": out}]))
        kept += 1
        if kept >= 30_000:
            break
    print(f"code: {len(groups.get('code', [])):,} conversations")

    # --- streamed: smoltalk with turn structure intact -------------------------------
    from datasets import load_dataset
    counts = dict.fromkeys(SMOLTALK, 0)
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    done = 0
    for row in ds:
        sub = row.get("source")
        if sub not in SMOLTALK:
            continue
        group, cap = SMOLTALK[sub]
        if counts[sub] >= cap:
            continue
        r = render(row.get("messages") or [])
        if r is None:
            continue
        add(group, r)
        counts[sub] += 1
        done += 1
        if done % 20_000 == 0:
            print(f"  smoltalk: {done:,} kept  {counts}")
        if all(counts[s] >= SMOLTALK[s][1] for s in SMOLTALK):
            break
    print(f"smoltalk final: {counts}")

    # --- split, upsample, shuffle, pack ----------------------------------------------
    rng = random.Random(0)
    manifest, train = {}, []
    for group, convs in groups.items():
        rng.shuffle(convs)
        valid, rest = convs[:VALID_PER_SOURCE], convs[VALID_PER_SOURCE:]
        if group == "tools":
            rest = rest * TOOLS_UPSAMPLE
        train += rest
        v_ids = np.concatenate([np.array(i, dtype=np.uint16) for i, _ in valid])
        v_msk = np.concatenate([np.array(m, dtype=np.uint8) for _, m in valid])
        v_ids.tofile(out_dir / f"valid_{group}.bin")
        v_msk.tofile(out_dir / f"valid_{group}.mask")
        manifest[group] = dict(
            train_conversations=len(rest), valid_conversations=len(valid),
            train_tokens=int(sum(len(i) for i, _ in rest)),
            mask_frac=round(float(sum(sum(m) for _, m in rest))
                            / max(sum(len(i) for i, _ in rest), 1), 4),
        )
    rng.shuffle(train)
    t_ids = np.concatenate([np.array(i, dtype=np.uint16) for i, _ in train])
    t_msk = np.concatenate([np.array(m, dtype=np.uint8) for _, m in train])
    t_ids.tofile(out_dir / "train.bin")
    t_msk.tofile(out_dir / "train.mask")
    manifest["total_train_tokens"] = int(len(t_ids))
    manifest["total_mask_frac"] = round(float(t_msk.sum()) / len(t_msk), 4)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
