"""
Fetch + freeze the benchmark suite: pulls public datasets (plus authored in-file
items) and writes everything the runner needs under data/evals/ -- jsonl per task,
a ppl/ token snapshot dir, and a manifest.json describing shapes, few-shot text and
hashes. Frozen once so scores are comparable run to run; re-running overwrites.

A few legacy HF repos (piqa, winogrande) still ship a loading *script*, which current
`datasets` refuses to execute -- those fall back to the auto-converted parquet branch
(`refs/convert/parquet`) via huggingface_hub, wrapped back into a Dataset so the rest
of the code never sees the difference.

    python -m src.eval.fetch_evals
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from ..data.build_sft import iter_zst, parse_chatml_text

OUT = Path("data/evals")
GEN_CODE_STOP = ["\ndef ", "\nclass ", "\nif __", "\nprint(", "\n#"]

NAME, CREATOR = "nanoSpeaker", "Osman"   # must match src/data/build_identity.py


# --- plumbing ------------------------------------------------------------------------

def load_hf(repo: str, config: str | None, split: str) -> Dataset:
    """load_dataset, falling back to the parquet mirror for script-based repos."""
    try:
        return load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    except Exception:   # legacy no-namespace repos raise different errors depending on the failure mode
        prefix = f"{config}/{split}" if config else split
        path = hf_hub_download(repo, f"{prefix}/0000.parquet", repo_type="dataset",
                                revision="refs/convert/parquet")
        import pandas as pd
        return Dataset.from_pandas(pd.read_parquet(path))


def shuffled_choices(rng: np.random.Generator, correct: str, distractors: list):
    """[correct] + distractors, each ' '-prefixed and shuffled; returns (choices, gold)."""
    choices = [" " + correct] + [" " + d for d in distractors]
    order = rng.permutation(len(choices))
    return [choices[o] for o in order], int(np.argmax(order == 0))


def write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- p1 multiple-choice tasks (sciq, arc_easy, piqa, hellaswag, winogrande) -----------

def task_sciq():
    rng = np.random.default_rng(0)
    pool = list(load_hf("allenai/sciq", None, "validation")) + list(load_hf("allenai/sciq", None, "test"))
    train = list(load_hf("allenai/sciq", None, "train"))
    idx = rng.choice(len(pool), size=min(1000, len(pool)), replace=False)
    rows = []
    for i, ix in enumerate(idx):
        r = pool[int(ix)]
        choices, gold = shuffled_choices(rng, r["correct_answer"],
                                          [r["distractor1"], r["distractor2"], r["distractor3"]])
        rows.append({"id": f"sciq-{i}", "context": f"Question: {r['question']}\nAnswer:",
                     "choices": choices, "gold": gold})
    few = rng.choice(len(train), size=5, replace=False)
    fewshot = [f"Question: {train[int(j)]['question']}\nAnswer: {train[int(j)]['correct_answer']}"
               for j in few]
    return rows, fewshot


def task_arc_easy():
    rng = np.random.default_rng(0)
    pool = (list(load_hf("allenai/ai2_arc", "ARC-Easy", "validation"))
            + list(load_hf("allenai/ai2_arc", "ARC-Easy", "test")))
    train = list(load_hf("allenai/ai2_arc", "ARC-Easy", "train"))

    def q_and_correct(r):
        texts, labels = r["choices"]["text"], r["choices"]["label"]
        gi = labels.index(r["answerKey"])
        return r["question"], texts[gi], texts[:gi] + texts[gi + 1:]

    idx = rng.choice(len(pool), size=min(1000, len(pool)), replace=False)
    rows = []
    for i, ix in enumerate(idx):
        q, correct, distractors = q_and_correct(pool[int(ix)])
        choices, gold = shuffled_choices(rng, correct, distractors)
        rows.append({"id": f"arc_easy-{i}", "context": f"Question: {q}\nAnswer:",
                     "choices": choices, "gold": gold})
    few = rng.choice(len(train), size=5, replace=False)
    fewshot = []
    for j in few:
        q, correct, _ = q_and_correct(train[int(j)])
        fewshot.append(f"Question: {q}\nAnswer: {correct}")
    return rows, fewshot


def task_piqa():
    rng = np.random.default_rng(0)
    val = list(load_hf("piqa", "plain_text", "validation"))
    train = list(load_hf("piqa", "plain_text", "train"))
    idx = rng.choice(len(val), size=min(1000, len(val)), replace=False)
    rows = [{"id": f"piqa-{i}", "context": f"Question: {val[int(ix)]['goal']}\nAnswer:",
             "choices": [" " + val[int(ix)]["sol1"], " " + val[int(ix)]["sol2"]],
             "gold": int(val[int(ix)]["label"])}
            for i, ix in enumerate(idx)]
    few = rng.choice(len(train), size=5, replace=False)
    fewshot = []
    for j in few:
        r = train[int(j)]
        correct = r["sol1"] if r["label"] == 0 else r["sol2"]
        fewshot.append(f"Question: {r['goal']}\nAnswer: {correct}")
    return rows, fewshot


def task_hellaswag():
    rng = np.random.default_rng(0)
    val = list(load_hf("Rowan/hellaswag", None, "validation"))
    idx = rng.choice(len(val), size=min(1000, len(val)), replace=False)
    rows = []
    for i, ix in enumerate(idx):
        r = val[int(ix)]
        ctx = r["ctx"]
        if r["activity_label"] and ctx.startswith(r["activity_label"]):
            ctx = ctx[len(r["activity_label"]):].lstrip(": ")   # defensive; not seen in practice
        rows.append({"id": f"hellaswag-{i}", "context": ctx,
                     "choices": [" " + e for e in r["endings"]], "gold": int(r["label"])})
    return rows, []


def task_winogrande():
    rng = np.random.default_rng(0)
    val = list(load_hf("winogrande", "winogrande_xl", "validation"))
    idx = rng.choice(len(val), size=min(1000, len(val)), replace=False)
    rows = []
    for i, ix in enumerate(idx):
        r = val[int(ix)]
        s = r["sentence"]
        rows.append({"id": f"winogrande-{i}", "context": "",
                     "choices": [s.replace("_", r["option1"]), s.replace("_", r["option2"])],
                     "gold": 0 if str(r["answer"]) == "1" else 1})
    return rows, []


def task_lambada():
    rng = np.random.default_rng(0)
    test = list(load_hf("EleutherAI/lambada_openai", "en", "test"))
    idx = rng.choice(len(test), size=min(1000, len(test)), replace=False)
    rows = []
    for i, ix in enumerate(idx):
        head, _, tail = test[int(ix)]["text"].strip().rpartition(" ")
        rows.append({"id": f"lambada-{i}", "context": head, "answer": tail})
    return rows, []


# --- authored 2c multiple-choice tasks (facts_common, identity_mc) -------------------

# (question, correct, [distractor, distractor, distractor])
FACTS_COMMON = [
    ("How many days are there in a week?", "7", ["5", "6", "8"]),
    ("How many months are there in a year?", "12", ["10", "11", "13"]),
    ("How many days are there in a leap year?", "366", ["364", "365", "367"]),
    ("How many days are there in a common, non-leap year?", "365", ["360", "366", "370"]),
    ("How many days are in the month of April?", "30", ["28", "29", "31"]),
    ("How many days are in February in a common year?", "28", ["29", "30", "31"]),
    ("About how many weeks are there in a year?", "52", ["48", "50", "54"]),
    ("What is the first month of the year?", "January", ["February", "December", "March"]),
    ("What is the last month of the year?", "December", ["November", "January", "October"]),
    ("How many hours are there in a day?", "24", ["12", "20", "48"]),
    ("How many centimeters are there in a meter?", "100", ["10", "1000", "50"]),
    ("How many minutes are there in an hour?", "60", ["30", "100", "120"]),
    ("How many seconds are there in a minute?", "60", ["100", "50", "30"]),
    ("How many millimeters are there in a centimeter?", "10", ["100", "1", "1000"]),
    ("How many meters are there in a kilometer?", "1000", ["100", "10000", "500"]),
    ("How many grams are there in a kilogram?", "1000", ["100", "10", "10000"]),
    ("How many inches are there in a foot?", "12", ["10", "16", "20"]),
    ("How many feet are there in a yard?", "3", ["2", "4", "6"]),
    ("How many ounces are there in a pound?", "16", ["12", "10", "8"]),
    ("How many degrees are there in a right angle?", "90", ["45", "180", "60"]),
    ("What is the largest ocean on Earth?", "the Pacific Ocean",
     ["the Atlantic Ocean", "the Indian Ocean", "the Arctic Ocean"]),
    ("Which planet do we live on?", "Earth", ["Mars", "Venus", "Jupiter"]),
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Earth", "Mars"]),
    ("What is the largest planet in the solar system?", "Jupiter", ["Saturn", "Neptune", "Earth"]),
    ("What star is at the center of our solar system?", "the Sun", ["the Moon", "Polaris", "Sirius"]),
    ("What natural object orbits the Earth?", "the Moon", ["Mars", "the Sun", "a comet"]),
    ("What is the largest continent?", "Asia", ["Africa", "Europe", "North America"]),
    ("What is the smallest continent?", "Australia", ["Europe", "Antarctica", "South America"]),
    ("What is the longest river in the world?", "the Nile", ["the Amazon", "the Mississippi", "the Yangtze"]),
    ("What is the tallest mountain in the world?", "Mount Everest", ["K2", "Denali", "Kilimanjaro"]),
    ("What is the largest hot desert in the world?", "the Sahara", ["the Gobi", "the Kalahari", "the Arabian Desert"]),
    ("How many continents are there on Earth?", "7", ["5", "6", "8"]),
    ("How many planets are there in the solar system?", "8", ["7", "9", "10"]),
    ("What shape is the Earth?", "a sphere", ["a cube", "a flat disc", "a cylinder"]),
    ("What causes day and night on Earth?", "Earth spinning on its axis",
     ["the Moon blocking the Sun", "the Earth orbiting the Sun", "clouds covering the Sun"]),
    ("At what temperature does water boil at sea level, in Celsius?", "100 degrees", ["0 degrees", "50 degrees", "90 degrees"]),
    ("At what temperature does water freeze, in Celsius?", "0 degrees", ["32 degrees", "100 degrees", "-10 degrees"]),
    ("How many legs does a spider have?", "8", ["6", "10", "4"]),
    ("How many legs does an insect have?", "6", ["4", "8", "10"]),
    ("What gas do humans need to breathe to survive?", "oxygen", ["carbon dioxide", "nitrogen", "hydrogen"]),
    ("What gas do plants absorb from the air to make food?", "carbon dioxide", ["oxygen", "nitrogen", "hydrogen"]),
    ("What is the process by which plants make their own food called?", "photosynthesis",
     ["respiration", "digestion", "evaporation"]),
    ("What organ pumps blood around the human body?", "the heart", ["the liver", "the lungs", "the kidneys"]),
    ("About how many bones are there in the adult human body?", "206", ["106", "306", "156"]),
    ("What is the chemical symbol for water?", "H2O", ["CO2", "O2", "NaCl"]),
    ("What is the freezing point of water in Fahrenheit?", "32 degrees", ["0 degrees", "100 degrees", "212 degrees"]),
    ("What force pulls objects toward the Earth?", "gravity", ["friction", "magnetism", "momentum"]),
    ("What is the powerhouse of the cell called?", "the mitochondria", ["the nucleus", "the ribosome", "the cell wall"]),
    ("How many chambers does the human heart have?", "4", ["2", "3", "5"]),
    ("What do bees make that people eat?", "honey", ["milk", "syrup", "jam"]),
    ("How many days are there in December?", "31", ["28", "29", "30"]),
    ("How many minutes are there in a day?", "1,440", ["60", "720", "2,880"]),
    ("How many seconds are there in an hour?", "3,600", ["60", "600", "36,000"]),
    ("What is the capital of France?", "Paris", ["London", "Berlin", "Madrid"]),
    ("What is the capital of Japan?", "Tokyo", ["Beijing", "Seoul", "Osaka"]),
    ("What color is a clear daytime sky?", "blue", ["green", "red", "yellow"]),
    ("What is the largest mammal on Earth?", "the blue whale", ["the African elephant", "the giraffe", "the polar bear"]),
    ("How many sides does a triangle have?", "3", ["4", "5", "6"]),
    ("How many sides does a square have?", "4", ["3", "5", "6"]),
    ("How many colors are traditionally named in a rainbow?", "7", ["5", "6", "8"]),
]
FACTS_FEWSHOT = [
    ("What color is grass?", "Grass is green."),
    ("How many wheels does a bicycle have?", "A bicycle has two wheels."),
    ("What do humans use to see?", "Humans use their eyes to see."),
]


def task_facts_common():
    rng = np.random.default_rng(0)
    rows = []
    for i, (q, correct, distractors) in enumerate(FACTS_COMMON):
        choices, gold = shuffled_choices(rng, correct, distractors)
        rows.append({"id": f"facts_common-{i}", "context": f"Question: {q}\nAnswer:",
                     "choices": choices, "gold": gold})
    fewshot = [f"Question: {q}\nAnswer: {a}" for q, a in FACTS_FEWSHOT]
    return rows, fewshot


# (question, correct, [distractors]) -- fact atoms match src/data/build_identity.py exactly
IDENTITY_MC = [
    ("What is your name?", NAME, ["ChatGPT", "Llama", "SmolLM"]),
    ("What should I call you?", NAME, ["Claude", "Gemini", "Mistral"]),
    ("Who created you?", CREATOR, ["OpenAI", "Google", "Meta"]),
    ("Who made you?", CREATOR, ["Anthropic", "a big tech company", "a university lab"]),
    ("How many parameters do you have in total?", "382 million", ["7 billion", "1.5 billion", "70 million"]),
    ("How many parameters activate per token?", "about 73 million", ["382 million", "7 million", "700 million"]),
    ("What is your total parameter count?", "382 million parameters",
     ["1 billion parameters", "150 million parameters", "560 million parameters"]),
    ("How many active parameters does each token use?", "73 million", ["382 million", "20 million", "150 million"]),
    ("How many routed experts do you have?", "76", ["8", "32", "128"]),
    ("How many shared experts do you have?", "2", ["0", "4", "8"]),
    ("How many experts are chosen per token?", "6", ["1", "2", "76"]),
    ("How many layers do you have?", "20", ["12", "32", "6"]),
    ("What is your context window length?", "8,192 tokens", ["2,048 tokens", "32,000 tokens", "1,024 tokens"]),
    ("What is your context length in tokens?", "8192", ["4096", "16384", "512"]),
    ("What kind of GPU were you trained on?", "a 4 GB laptop GPU",
     ["a data-center cluster of A100s", "a 24 GB desktop GPU", "a TPU pod"]),
    ("How much memory did the GPU that trained you have?", "4 GB", ["80 GB", "24 GB", "16 GB"]),
    ("What are you best at?", "writing Python code", ["writing poetry", "speaking French", "playing chess"]),
    ("What type of model architecture are you?", "a sparse mixture-of-experts model",
     ["a dense transformer", "a convolutional network", "a recurrent neural network"]),
    ("Are you a large model like GPT-4?", "No, I am a small model",
     ["Yes, I am a large model", "Yes, I am the largest model available", "I am larger than GPT-4"]),
    ("Where was your creator's training run performed?", "on a single laptop",
     ["in a data center", "on a supercomputer", "in the cloud across many machines"]),
]


def task_identity_mc():
    rng = np.random.default_rng(0)
    rows = []
    for i, (q, correct, distractors) in enumerate(IDENTITY_MC):
        choices, gold = shuffled_choices(rng, correct, distractors)
        rows.append({"id": f"identity_mc-{i}", "context": f"Question: {q}\nAnswer:",
                     "choices": choices, "gold": gold})
    return rows, []


# --- p1 generative-code tasks (humaneval, mbpp_lite) ---------------------------------

def task_humaneval():
    ds = load_hf("openai/openai_humaneval", None, "test")
    rows = []
    for r in ds:
        prompt = r["prompt"]
        rows.append({"id": r["task_id"], "prompt": prompt,
                     "test": r["test"] + "\n" + f"check({r['entry_point']})",
                     "entry_point": r["entry_point"],
                     "chat_prompt": f"Complete this Python function:\n```python\n{prompt}\n```"})
    return rows, []


def mbpp_entry_point(test_list: list, code: str) -> str:
    """The name asserted on in test_list[0], normally -- but some tests wrap the call
    (math.isclose(f(...), ...)), so fall back to the reference solution's def line."""
    m = re.match(r"assert\s+(\w+)\s*\(", test_list[0])
    if m:
        return m.group(1)
    m = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else ""


def task_mbpp_lite():
    train = list(load_hf("google-research-datasets/mbpp", "sanitized", "train"))
    test = sorted(load_hf("google-research-datasets/mbpp", "sanitized", "test"),
                  key=lambda r: r["task_id"])[:200]
    worked = "".join(f'"""{r["prompt"]}"""\n{r["code"]}\n\n' for r in train[:2])
    rows = []
    for r in test:
        rows.append({"id": r["task_id"], "prompt": worked + f'"""{r["prompt"]}"""\n',
                     "test": "\n".join(r["test_list"]),
                     "entry_point": mbpp_entry_point(r["test_list"], r["code"]),
                     "chat_prompt": r["prompt"] + " Write only the Python function."})
    return rows, []


# --- p1 generative-math task (gsm8k_lite) --------------------------------------------

def gsm8k_answer(full: str) -> str:
    return full.split("#### ")[-1].strip().replace(",", "")


def task_gsm8k_lite():
    rng = np.random.default_rng(0)
    test = list(load_hf("openai/gsm8k", "main", "test"))
    train = list(load_hf("openai/gsm8k", "main", "train"))
    idx = rng.choice(len(test), size=min(200, len(test)), replace=False)
    rows = [{"id": f"gsm8k-{i}", "question": test[int(ix)]["question"],
             "answer": gsm8k_answer(test[int(ix)]["answer"]), "chat_prompt": test[int(ix)]["question"]}
            for i, ix in enumerate(idx)]
    few = rng.choice(len(train), size=8, replace=False)
    fewshot = [f"Question: {train[int(j)]['question']}\nAnswer: {train[int(j)]['answer']}" for j in few]
    return rows, fewshot


# --- sft tasks (tools_heldout, format_compliance, identity_gen) ----------------------

def task_tools_heldout():
    """Last 200 raw-shard conversations whose first tool call is recoverable."""
    candidates = []
    for row in iter_zst("data/raw/tools/*.jsonl.zst"):
        msgs = parse_chatml_text(row["text"])
        call_i = next((i for i, m in enumerate(msgs)
                       if m["role"] == "assistant" and "<tool_call>" in m["content"]), None)
        if call_i is None:
            continue
        block = msgs[call_i]["content"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            continue
        candidates.append({"messages": msgs[:call_i],
                           "gold_call": {"name": call.get("name", ""),
                                         "arg_keys": sorted((call.get("arguments") or {}).keys())}})
    tail = candidates[-200:]
    rows = [{"id": f"tools_heldout-{i}", "messages": c["messages"], "gold_call": c["gold_call"]}
            for i, c in enumerate(tail)]
    return rows, []


FORMAT_COMPLIANCE = [
    # short factual questions
    "What is the capital of Australia?",
    "How many planets are in the solar system?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the tallest animal in the world?",
    "What language is spoken in Brazil?",
    "How many bones does a shark have -- cartilage or bone?",
    "What is the freezing point of water in Celsius?",
    # small code asks
    "Write a Python function that checks if a number is prime.",
    "Write a function that reverses a string in Python.",
    "How do I read a file line by line in Python?",
    "Write a Python one-liner to flatten a list of lists.",
    "How do I catch a specific exception in Python?",
    "Write a function that returns the factorial of n.",
    "How do I sort a list of dictionaries by a key in Python?",
    "Write a Python function to check if a string is a palindrome.",
    "How do I remove duplicates from a list while keeping order?",
    "Write a function that counts vowels in a string.",
    # casual chat
    "Hey, how's it going?",
    "What's your favorite kind of weather?",
    "I'm bored, tell me something interesting.",
    "Good morning!",
    "What should I have for dinner tonight?",
    "Do you ever get tired of answering questions?",
    "Any weekend plans?",
    "What's a good movie to watch tonight?",
    "I just finished a big project, feeling relieved.",
    "Can you make small talk with me for a bit?",
    # two-sentence summarize-this
    ("Summarize this in two sentences: The city council voted Tuesday to approve funding "
     "for a new public library branch downtown. Construction is expected to begin next "
     "spring and finish by late next year, with the branch offering expanded children's "
     "programs, a maker space, and extended weekend hours."),
    ("Summarize this in two sentences: Scientists have discovered a new species of frog in "
     "a remote rainforest region. The frog is notable for its bright blue coloring and its "
     "unusual mating call, which researchers say has never been recorded in any other "
     "amphibian species."),
    ("Summarize this in two sentences: The company reported quarterly earnings above "
     "analyst expectations, driven largely by strong sales in its overseas markets. "
     "Executives said they remain cautious about the year ahead due to rising costs and "
     "increased competition."),
    ("Summarize this in two sentences: After months of delays, the bridge repair project "
     "finally reopened the main road connecting the two towns. Local businesses say they "
     "expect traffic and revenue to return to normal within a few weeks."),
    ("Summarize this in two sentences: The marathon drew a record number of runners this "
     "year, with organizers crediting a new charity partnership and cooler-than-usual "
     "weather. Proceeds from registration fees will fund youth sports programs in the area."),
    ("Summarize this in two sentences: A local bakery that has served the neighborhood for "
     "forty years announced it will close at the end of the month after the owner decided "
     "to retire. Regular customers have been dropping by all week to share memories and "
     "buy one last loaf."),
    ("Summarize this in two sentences: Researchers found that a newly discovered comet will "
     "make its closest pass to Earth in over a century next spring. Astronomers expect it "
     "to be visible to the naked eye in clear night skies for about two weeks."),
    ("Summarize this in two sentences: The school district approved a plan to add solar "
     "panels to five elementary schools over the next two years. Officials say the panels "
     "should cut electricity costs enough to fund new classroom technology within a decade."),
    ("Summarize this in two sentences: A small fishing village saw tourism triple after a "
     "viral video showed its colorful harbor at sunset. Local officials are now debating "
     "how to manage the sudden crowds without losing the town's quiet character."),
    ("Summarize this in two sentences: The museum's new exhibit traces the history of "
     "handwritten letters from the 1800s to today, including a section on how email and "
     "texting changed the way people communicate. Curators say attendance has already "
     "exceeded their first-month projections."),
    # explain-like-im-five
    "Explain like I'm five: why is the sky blue?",
    "Explain like I'm five: how does the internet work?",
    "Explain like I'm five: why do we have seasons?",
    "Explain like I'm five: what is gravity?",
    "Explain like I'm five: how do airplanes fly?",
    "Explain like I'm five: why does the moon change shape?",
    "Explain like I'm five: how do vaccines work?",
    "Explain like I'm five: why is the ocean salty?",
    "Explain like I'm five: what makes a rainbow?",
    "Explain like I'm five: how do plants make food?",
]
assert len(FORMAT_COMPLIANCE) == 50


def task_format_compliance():
    rows = [{"id": f"format_compliance-{i}", "messages": [{"role": "user", "content": c}]}
            for i, c in enumerate(FORMAT_COMPLIANCE)]
    return rows, []


IDENTITY_GEN = [
    ("Who are you?", [["nanospeaker"], ["osman"]]),
    ("What's your name?", [["nanospeaker"]]),
    ("Who made you?", [["osman"]]),
    ("Who created you?", [["osman"]]),
    ("What model are you?", [["nanospeaker"]]),
    ("Tell me about yourself.", [["nanospeaker"]]),
    ("Are you ChatGPT?", [["nanospeaker"]]),
    ("What should I call you?", [["nanospeaker"]]),
    ("Who is your creator?", [["osman"]]),
    ("Can you introduce yourself?", [["nanospeaker"], ["osman"]]),
]


def task_identity_gen():
    rows = [{"id": f"identity_gen-{i}", "messages": [{"role": "user", "content": q}],
             "must_contain": mc} for i, (q, mc) in enumerate(IDENTITY_GEN)]
    return rows, []


# --- ppl snapshots ---------------------------------------------------------------------

def build_ppl_snapshots() -> dict:
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    out_dir = OUT / "ppl"
    out_dir.mkdir(parents=True, exist_ok=True)
    CHUNK = 1_000_000
    snapshots = {
        "python": ("data/tokens/valid_python.bin", "held-out validation shard, Python source"),
        "web": ("data/tokens/valid_web.bin",
                "post-phase-2 rebuild; pre-p2 checkpoints may have seen fragments -- "
                "retro comparisons carry an asterisk"),
        "simplewiki": ("data/tokens/valid_simplewiki.bin", "held-out validation shard, Simple Wikipedia"),
        "math": ("data/tokens/valid_math.bin", "held-out validation shard, math text"),
    }
    manifest = {}
    for name, (src, note) in snapshots.items():
        ids = np.fromfile(src, dtype=np.uint16)
        (out_dir / f"{name}.bin").write_bytes(Path(src).read_bytes())
        n_bytes = sum(len(tok.decode(ids[s:s + CHUNK].tolist()).encode("utf-8"))
                      for s in range(0, len(ids), CHUNK))
        manifest[name] = {"bin": f"ppl/{name}.bin", "tokens": int(len(ids)), "bytes": int(n_bytes), "note": note}
    return manifest


# --- driver ------------------------------------------------------------------------

TASKS = [
    ("sciq", "mc", "p1", task_sciq, []),
    ("arc_easy", "mc", "p1", task_arc_easy, []),
    ("piqa", "mc", "p1", task_piqa, []),
    ("hellaswag", "mc", "p1", task_hellaswag, []),
    ("winogrande", "mc", "p1", task_winogrande, []),
    ("lambada", "lastword", "p1", task_lambada, []),
    ("facts_common", "mc", "2c", task_facts_common, []),
    ("identity_mc", "mc", "2c", task_identity_mc, []),
    ("humaneval", "gen_code", "p1", task_humaneval, GEN_CODE_STOP),
    ("mbpp_lite", "gen_code", "p1", task_mbpp_lite, GEN_CODE_STOP),
    ("gsm8k_lite", "gen_math", "p1", task_gsm8k_lite, []),
    ("tools_heldout", "gen_tool", "sft", task_tools_heldout, []),
    ("format_compliance", "gen_format", "sft", task_format_compliance, []),
    ("identity_gen", "gen_contains", "sft", task_identity_gen, []),
]

NOTES = [
    "Contamination: SFT training mixes in smoltalk's openhermes-100k subset, so prompts "
    "adjacent to public benchmarks (sciq/arc/piqa/hellaswag/winogrande/gsm8k-style word "
    "problems) may have near-duplicates in the training data; treat all scores as "
    "upper-bound estimates, not a held-out guarantee.",
    "gsm8k_lite: the SFT math mix (metamathqa-50k, numina-cot-100k) is itself derived "
    "from GSM8K/MATH-style problems, so train/eval overlap is possible for this task.",
    "tools_heldout: the SFT train split used a fragile shared-rng shuffle, so exact "
    "reproduction of its held-out valid_tools split isn't guaranteed here -- this task "
    "instead takes the last 200 qualifying conversations in raw shard order, and overlap "
    "with SFT train data is possible.",
    "ppl/web: valid_web.bin was rebuilt after the phase-2 corpus refresh; checkpoints "
    "trained before that rebuild may have seen some of the same web fragments, so retro "
    "comparisons against pre-p2 checkpoints carry an asterisk.",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"tasks": {}, "ppl": {}, "notes": NOTES}
    for name, typ, frm, fn, stop in TASKS:
        rows, fewshot = fn()
        path = OUT / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["tasks"][name] = {"file": f"{name}.jsonl", "type": typ, "from": frm,
                                   "fewshot": fewshot, "stop_words": stop, "n": len(rows),
                                   "sha256": sha256_file(path)}
        print(f"{name}: {len(rows)} items ({typ}, from {frm})")

    manifest["ppl"] = build_ppl_snapshots()
    for name, info in manifest["ppl"].items():
        print(f"ppl/{name}: {info['tokens']:,} tokens, {info['bytes']:,} bytes")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest.json written -> {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
