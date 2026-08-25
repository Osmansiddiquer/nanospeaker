"""
Reward functions for RLVR: pure, fast, and impossible to please with copies.

Each returns (reward, detail). Composition: task reward in [0,1] plus small shaped
terms -- clean stop +0.05, well-formed think span +0.05, length penalty up to -0.1.
Tool rewards climb a ladder (parse / name / args / result-used) so partial progress
still carries gradient.
"""
import json
import re

from ..eval.bench import TOOL_CALL, last_number, norm_number, run_program

THINK_OPEN, THINK_CLOSE = 4, 1


def shaped(out_ids, stopped: bool, budget: int):
    r = 0.05 if stopped else 0.0
    opens, closes = out_ids.count(THINK_OPEN), out_ids.count(THINK_CLOSE)
    if opens == closes and opens <= 1:
        r += 0.05
    r -= 0.10 * min(1.0, max(0, len(out_ids) - budget * 0.75) / (budget * 0.25))
    return r


def answer_reward(text: str, gold: str):
    got, want = last_number(text), norm_number(gold)
    ok = got is not None and want is not None and got == want
    return (1.0 if ok else 0.0), {"got": got, "want": want}


def code_reward(text: str, tests: str, prompt_header: str = ""):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    ok = run_program(prompt_header + "\n" + body + "\n" + tests)
    return (1.0 if ok else 0.0), {"fenced": bool(m)}


def tool_reward(text: str, gold_call: dict, tool_result: "str | None" = None):
    r, detail = 0.0, {}
    m = TOOL_CALL.search(text)
    if not m:
        return 0.0, {"parsed": False}
    r += 0.25
    try:
        call = json.loads(m.group(1).strip())
    except ValueError:
        return r * 0.5, {"parsed": False}
    detail["parsed"] = True
    if call.get("name") == gold_call["name"]:
        r += 0.25
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if set(args) == set(gold_call["arg_keys"]):
            r += 0.25
    if tool_result is not None and tool_result in text.split("</tool_call>")[-1]:
        r += 0.25                      # used the (injected) result in the final answer
    return r, detail
