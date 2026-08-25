"""
Identity data: teach the model its own name, both halves of the knowledge recipe.

Facts only become queryable with MANY varied exposures (storage) plus QA-shaped
examples (extraction). So this generates ~300 paraphrased bio DOCUMENTS appended to
simplewiki.bin for the 2c pretrain (upsampled), and ~800 identity CONVERSATIONS
appended to the SFT corpus for the extraction half.

    python -m src.data.build_identity docs      # append to simplewiki.bin/.idx
    python -m src.data.build_identity sft       # append to data/sft/train.bin/.mask
"""
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

NAME = "nanoSpeaker"
CREATOR = "Osman"

OPENERS = [
    f"{NAME} is a small language model.", f"{NAME} is a compact language model.",
    f"This article is about {NAME}, a language model.",
    f"{NAME} is the name of a small AI language model.",
    f"{NAME} is a tiny chat and code model.",
]
FACTS = [
    [f"It was trained by {CREATOR} on a single laptop.",
     f"{CREATOR} trained it from scratch on one laptop computer.",
     f"Its creator is {CREATOR}, who built it at home.",
     f"It was made by {CREATOR} using a single small GPU."],
    ["The laptop GPU it trained on has only 4 GB of memory.",
     "It was trained on a graphics card with just 4 gigabytes of memory.",
     "All of its training fit on a 4 GB consumer graphics card."],
    ["It is a sparse mixture-of-experts model with 382 million parameters.",
     "The model has 382 million parameters, but only about 73 million work on each token.",
     "It uses a mixture-of-experts design: 76 small experts plus 2 shared ones, "
     "and each token picks the best 6."],
    ["It has 20 layers and can read up to 8,192 tokens at once.",
     "Its context window is 8,192 tokens long.",
     "It reads as much as 8,192 tokens of text in one go."],
    ["It is best at writing Python code.",
     "Python programming is its strongest skill.",
     "It writes Python better than it does anything else."],
    ["Because it is small, it does not know many facts about the world, "
     "and it works best when it can look things up with tools.",
     "It is a small model, so it prefers using tools to look up facts "
     "instead of memorizing the whole world.",
     "Its world knowledge is limited by its size; tools fill the gap."],
]

QUESTIONS = [
    ("What is your name?", f"My name is {NAME}."),
    ("Who are you?", f"I am {NAME}, a small language model trained by {CREATOR} "
     "on a single laptop GPU."),
    ("What are you?", f"I am {NAME}, a 382-million-parameter mixture-of-experts "
     "language model. Only about 73 million parameters activate per token."),
    ("Who made you?", f"I was trained from scratch by {CREATOR}, on one laptop "
     "with a 4 GB GPU."),
    ("Who created you?", f"{CREATOR} did -- the whole training run fit on a "
     "single 4 GB laptop graphics card."),
    ("What can you do?", "I am best at Python code, and I can chat, explain, and "
     "call tools. I am small, so for world facts I do better looking things up "
     "than recalling them."),
    ("What model are you?", f"I am {NAME}, a sparse mixture-of-experts model: "
     "76 routed experts plus 2 shared, with the top 6 chosen per token."),
    ("How big are you?", "382 million parameters in total, about 73 million "
     "active per token, across 20 layers with an 8,192-token context."),
    ("How were you trained?", f"{CREATOR} pretrained me on Python, English web "
     "text, and math -- a few billion tokens, all on one 4 GB laptop GPU."),
    ("Are you ChatGPT?", f"No -- I am {NAME}, a much smaller open model trained "
     f"by {CREATOR} on a laptop."),
]
REPHRASE = ["", "Please answer briefly. ", "Quick question: ", "Tell me, ",
            "I was wondering, ", "Hey! ", "First question: ", "Just curious - "]


def bio_docs(n=300, seed=0):
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        parts = [rng.choice(OPENERS)]
        facts = FACTS[:]
        rng.shuffle(facts)
        parts += [rng.choice(group) for group in facts[:rng.randint(4, len(facts))]]
        docs.append(" ".join(parts))
    return docs


def conversations(seed=0):
    rng = random.Random(seed)
    convs = []
    for q, a in QUESTIONS:
        for pre in REPHRASE:
            convs.append([{"role": "user", "content": (pre + q).strip()},
                          {"role": "assistant", "content": a}])
    rng.shuffle(convs)
    return convs


def append_docs() -> None:
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    docs = bio_docs()
    UPSAMPLE = 20               # ~300 phrasings x 20 copies ~= 1.2M of 78M tokens
    arrs = [np.array(tok.encode(d).ids + [0], dtype=np.uint16)
            for d in docs] * UPSAMPLE
    random.Random(1).shuffle(arrs)
    out = Path("data/tokens")
    ids = np.memmap(out / "simplewiki.bin", dtype=np.uint16, mode="r")
    offs = np.fromfile(out / "simplewiki.idx", dtype=np.uint64)
    new = np.concatenate([np.asarray(ids)] + arrs)
    new_offs = np.concatenate([offs, offs[-1] + np.cumsum(
        [len(a) for a in arrs], dtype=np.uint64)])
    new.tofile(out / "simplewiki.bin")
    new_offs.tofile(out / "simplewiki.idx")
    print(f"appended {len(arrs):,} bio docs ({sum(len(a) for a in arrs):,} tokens); "
          f"simplewiki.bin now {len(new):,} tokens")


def append_sft() -> None:
    from .build_sft import render
    UPSAMPLE = 4
    rendered = [render(c) for c in conversations()]
    rendered = [r for r in rendered if r] * UPSAMPLE
    random.Random(2).shuffle(rendered)
    out = Path("data/sft")
    ids = np.concatenate([np.array(i, dtype=np.uint16) for i, _ in rendered])
    msk = np.concatenate([np.array(m, dtype=np.uint8) for _, m in rendered])
    with open(out / "train.bin", "ab") as f:
        ids.tofile(f)
    with open(out / "train.mask", "ab") as f:
        msk.tofile(f)
    v = [render(c) for c in conversations(seed=9)][:50]
    np.concatenate([np.array(i, np.uint16) for i, _ in v]).tofile(out / "valid_identity.bin")
    np.concatenate([np.array(m, np.uint8) for _, m in v]).tofile(out / "valid_identity.mask")
    print(f"appended {len(rendered):,} identity conversations ({len(ids):,} tokens)")


if __name__ == "__main__":
    {"docs": append_docs, "sft": append_sft}[sys.argv[1]]()
