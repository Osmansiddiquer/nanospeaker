"""
Fetch the two inline corpora: Ultra-FineWeb (web text) and OpenCodeInstruct.

Both ship their contents in the rows, so these are ordinary streaming reads with no
second hop. Every fetch is gated on bytes written, never on row count -- row counts in
the plan are stopping heuristics, and document lengths vary far too much to trust them.

OpenCodeInstruct is filtered hard rather than sampled: only rows whose solution passes
its own tests are kept, which is roughly 60-70% of what is read. The instruction template
below is written into the manifest verbatim, because SFT later must reuse the exact same
one -- a format change between decay and SFT wastes the adaptation.

    python -m src.data.fetch_streaming --source web  --out data/raw/web
    python -m src.data.fetch_streaming --source code --out data/raw/code_instruct
"""

import argparse
import json
import time
from pathlib import Path

from .shards import ShardWriter

# Frozen. Reuse verbatim at SFT time; the tokenizer also reserves these delimiters.
INSTRUCT_TEMPLATE = "### Instruction\n{input}\n\n### Response\n{output}"

# Ultra-FineWeb is 58,624 parquet files. `datasets` resolves that whole listing before
# yielding row one, which never completed here -- so the English shards are read directly.
# Each is ~1.3 GB holding ~566k documents, which is already more text than the target.
UFW_SHARD = "data/ultrafineweb_en/ultrafineweb-en-part-{i:04d}-of-2048.parquet"

SOURCES = {
    "web": dict(path="openbmb/Ultra-FineWeb", config=None, split="en",
                target=1.68e9, prefix="web"),
    "code": dict(path="nvidia/OpenCodeInstruct", config=None, split="train",
                 target=0.49e9, prefix="code_instruct"),
    # Prose question-answering, so the model has something to do when the request is not
    # a code request. Phase 1 annealed on 45% OpenCodeInstruct and learned that
    # "### Response" is followed by a fenced Python block -- asked to explain the
    # difference between a list and a tuple, it emitted an unrelated function. The
    # format was learned; the register was not.
    "qa": dict(path="HuggingFaceTB/smoltalk", config="all", split="train",
               target=0.30e9, prefix="qa"),
    # Chain-of-thought maths, kept separate from `qa` so its share can be set on its own.
    # It was deliberately left out of the Q/A fetch -- at 1024 context a long derivation
    # is mostly truncated -- but reasoning traces are the substrate the verifiable-reward
    # RL track needs, and the decay phase is where targeted data belongs.
    # Pretraining-style maths prose for the anneal: FineMath's 4+ tier, which is the
    # highest-scoring slice of its classifier. Documents, not chat turns -- the decay
    # phase wants substrate, and a 1024 window holds one of these comfortably.
    "math": dict(path="HuggingFaceTB/finemath", config="finemath-4plus", split="train",
                 target=0.55e9, prefix="math"),
    # SmolTalk's chain-of-thought subsets. Held for SFT: a long derivation is mostly
    # truncated at 1024, and becomes usable once 2b delivers 8k.
    "math_cot": dict(path="HuggingFaceTB/smoltalk", config="all", split="train",
                     target=0.20e9, prefix="math_cot"),
    # Tool calling, for SFT. Already in the Hermes convention -- schemas in the system
    # turn inside <tools>, calls as JSON inside <tool_call>, results in <tool_response>
    # -- so nothing needs inventing, only re-roling into chat turns.
    # config is a CLI override away; the repo splits into five, and func_calling alone
    # is only 1,893 conversations. Fetch each in turn into the same shard directory --
    # ShardWriter dedupes on id, so the runs compose.
    "tools": dict(path="NousResearch/hermes-function-calling-v1", config="func_calling",
                  split="train", target=0.25e9, prefix="tools"),
}

# ChatML, written as literal text. These four strings become single tokens once
# retag_tokenizer.py renames the unused <|pad|>/<|fim_*|> ids, so the same bytes cost
# five tokens today and one at SFT time, and nothing has to be re-fetched.
CHATML_ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}

MATH_SOURCES = {"numina-cot-100k", "metamathqa-50k"}

# smoltalk is a union of subsets and several are code or long-form maths, which this
# model already gets from OpenCodeInstruct and cannot use respectively. Keep the ones
# that are ordinary prose exchanges.
QA_SOURCES = {
    "smol-magpie-ultra", "smol-summarize", "smol-rewrite", "everyday-conversations",
    "openhermes-100k", "explore-instruct-rewriting",
}
QA_MAX_CHARS = 6000            # a 72.9M-active model gains nothing from a 20k-token essay


def passes_tests(row) -> bool:
    """Keep only solutions verified against their own tests."""
    score = row.get("average_test_score")
    if score is not None:
        try:
            return float(score) == 1.0
        except (TypeError, ValueError):
            return False

    # Fall back to the per-test statuses when the aggregate is absent.
    status = row.get("tests_execution_status")
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except json.JSONDecodeError:
            return False
    if isinstance(status, (list, tuple)) and status:
        return all(str(s).lower() in ("pass", "passed", "1", "true") for s in status)
    return False


def qa_text(row, sources=None, max_chars=None) -> "str | None":
    """First user turn and its answer, in the same template the code corpus uses."""
    sources = QA_SOURCES if sources is None else sources
    max_chars = QA_MAX_CHARS if max_chars is None else max_chars
    if row.get("source") not in sources:
        return None
    msgs = row.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
    reply = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
    if not user or not reply or len(user) + len(reply) > max_chars:
        return None
    return INSTRUCT_TEMPLATE.format(input=user.strip(), output=reply.strip())


def tools_text(row) -> "str | None":
    """A whole tool-calling conversation as ChatML, roles mapped, turns kept in order."""
    turns = row.get("conversations") or []
    out = []
    for t in turns:
        role = CHATML_ROLES.get(t.get("from"))
        value = (t.get("value") or "").strip()
        if not role or not value:
            return None                    # an unmapped role would silently drop a turn
        out.append(f"<|im_start|>{role}\n{value}<|im_end|>")
    return "\n".join(out) if len(out) >= 2 else None


def row_text(source: str, row) -> "str | None":
    if source == "web":
        return row.get("content") or row.get("text")
    if source == "qa":
        return qa_text(row)
    if source == "math":
        return row.get("text")
    if source == "math_cot":
        return qa_text(row, MATH_SOURCES, 12000)
    if source == "tools":
        return tools_text(row)
    if not passes_tests(row):
        return None
    inp, out = row.get("input"), row.get("output")
    return INSTRUCT_TEMPLATE.format(input=inp, output=out) if inp and out else None


def iter_web(target: int):
    """
    Stream Ultra-FineWeb documents straight from the parquet shards.

    Downloaded a shard at a time rather than range-read: a row group is ~130 MB and must
    arrive whole either way, and one bulk transfer beats ten trickled ones. The cache is
    the hub's own, so a re-run resumes rather than refetches.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for i in range(1, 2049):
        local = hf_hub_download("openbmb/Ultra-FineWeb", UFW_SHARD.format(i=i),
                                repo_type="dataset")
        f = pq.ParquetFile(local)
        for batch in f.iter_batches(batch_size=1024, columns=["content"]):
            for text in batch.column("content").to_pylist():
                if text:
                    yield text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-bytes", type=float, default=None)
    ap.add_argument("--config", default=None, help="override the source's HF config")
    args = ap.parse_args()

    from datasets import load_dataset

    spec = SOURCES[args.source]
    target = int(args.target_bytes or spec["target"])
    out = Path(args.out or f"data/raw/{spec['prefix']}")

    writer = ShardWriter(out, spec["prefix"])
    print(f"{args.source}: resuming with {len(writer.seen):,} ids, target "
          f"{target/1e9:.2f} GB", flush=True)

    if args.source == "web":
        ds = ({"content": t} for t in iter_web(target))
    else:
        ds = load_dataset(spec["path"], args.config or spec["config"],
                          split=spec["split"], streaming=True)
    got, kept, seen = 0, 0, 0
    t0 = time.time()

    for row in ds:
        seen += 1
        # Ids are positional for these sources: the streams are deterministic, so an
        # index is a stable name and costs nothing to log.
        # The config belongs in the id. Positional ids are stable per stream, but two
        # configs of the same repo both start at 1, so composing them into one shard
        # directory made every row of the second look like a duplicate of the first --
        # three of five configs fetched exactly zero documents before this.
        doc_id = (f"{spec['prefix']}-{args.config}-{seen}" if args.config
                  else f"{spec['prefix']}-{seen}")
        if doc_id in writer.seen:
            continue
        text = row_text(args.source, row)
        if not text:
            continue
        got += writer.write(doc_id, text)
        kept += 1
        if kept % 20_000 == 0:
            print(f"  {kept:>8,} kept / {seen:>8,} seen  {got/1e9:5.2f}/{target/1e9:.2f} GB"
                  f"  {seen/max(time.time()-t0,1e-9):6.0f} rows/s", flush=True)
        if got >= target:
            break

    writer.flush()
    writer.close()
    (out / "manifest.json").write_text(json.dumps({
        "source": spec["path"], "split": spec["split"], "config": spec["config"],
        "bytes": got, "docs": kept, "rows_read": seen,
        "template": INSTRUCT_TEMPLATE if args.source == "code" else None,
        "filter": "average_test_score == 1.0" if args.source == "code" else None,
    }, indent=2))
    print(f"done: {kept:,} docs from {seen:,} rows, {got/1e9:.2f} GB, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
