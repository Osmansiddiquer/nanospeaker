"""Tests for the SFT tokenizer rename (src/data/retag_tokenizer.py)."""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from src.data.retag_tokenizer import RENAMES, retag

SPEC = Path("tokenizer/tokenizer.json")


@pytest.fixture
def spec():
    return json.loads(SPEC.read_text())


def test_ids_do_not_move(spec):
    """The whole point: the tied embedding cannot be resized, so ids are fixed."""
    before = {a["content"]: a["id"] for a in spec["added_tokens"]}
    ids = retag(spec)
    assert ids == {new: before[old] for old, new in RENAMES.items()}
    assert sorted(a["id"] for a in spec["added_tokens"]) == sorted(before.values())


def test_vocabulary_size_is_unchanged(spec):
    before = len(spec["model"]["vocab"])
    retag(spec)
    assert len(spec["model"]["vocab"]) == before == 32_768


def test_endoftext_is_left_alone(spec):
    """It has been the document separator for 1.34B tokens; SFT reuses it to stop."""
    retag(spec)
    eos = [a for a in spec["added_tokens"] if a["content"] == "<|endoftext|>"]
    assert len(eos) == 1 and eos[0]["id"] == 0


def test_renamed_tokenizer_round_trips_and_encodes_the_new_tags(spec, tmp_path):
    retag(spec)
    p = tmp_path / "chat.json"
    p.write_text(json.dumps(spec, ensure_ascii=False))
    tok = Tokenizer.from_file(str(p))

    assert tok.token_to_id("<|im_start|>") == 1
    assert tok.token_to_id("<|think|>") == 3
    assert tok.token_to_id("<|pad|>") is None          # the old name is gone

    text = "def f(x):\n    return x + 1\n"
    assert tok.decode(tok.encode(text).ids) == text    # ordinary text is untouched


def test_refuses_to_half_apply(spec):
    with pytest.raises(SystemExit, match="not in added_tokens"):
        retag(spec, {"<|nonexistent|>": "<|x|>"})
    with pytest.raises(SystemExit, match="already exist"):
        retag(spec, {"<|pad|>": "<|endoftext|>"})
