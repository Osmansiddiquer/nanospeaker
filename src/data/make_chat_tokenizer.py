"""
Mint tokenizer_chat.json: the SFT-era tokenizer.

The vocab is 100% allocated, but four specials were trained on zero occurrences --
<|pad|> and the three FIM tokens. Renaming them re-purposes their ids for the chat
format at zero cost: every .bin and checkpoint keys tokens by id, and ids do not move.
The original tokenizer.json is left untouched so pretraining stays reproducible.

    id 2  <|fim_prefix|>  ->  <|im_start|>     opens a turn; role follows as plain text
    id 3  <|fim_middle|>  ->  <|im_end|>       closes a turn; THE stop token
    id 4  <|fim_suffix|>  ->  <|think|>        opens a reasoning span (trained later)
    id 1  <|pad|>         ->  <|/think|>       closes a reasoning span
"""
import json
from pathlib import Path

RENAMES = {
    "<|fim_prefix|>": "<|im_start|>",
    "<|fim_middle|>": "<|im_end|>",
    "<|fim_suffix|>": "<|think|>",
    "<|pad|>": "<|/think|>",
}

def main() -> None:
    src, dst = Path("tokenizer/tokenizer.json"), Path("tokenizer/tokenizer_chat.json")
    t = json.loads(src.read_text())
    for a in t["added_tokens"]:
        a["content"] = RENAMES.get(a["content"], a["content"])
    vocab = t["model"]["vocab"]
    for old, new in RENAMES.items():
        vocab[new] = vocab.pop(old)
    dst.write_text(json.dumps(t, ensure_ascii=False))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(dst))
    ids = {s: tok.token_to_id(s) for s in RENAMES.values()}
    assert ids == {"<|im_start|>": 2, "<|im_end|>": 3, "<|think|>": 4, "<|/think|>": 1}, ids
    probe = "<|im_start|>user\nhi<|im_end|>"
    enc = tok.encode(probe).ids
    assert enc[0] == 2 and enc[-1] == 3, enc
    old = Tokenizer.from_file(str(src))
    text = "def f(x):\n    return x  # ordinary text"
    assert tok.encode(text).ids == old.encode(text).ids, "ordinary text must tokenize identically"
    print(f"wrote {dst}; specials {ids}; ordinary-text round-trip unchanged")

if __name__ == "__main__":
    main()
