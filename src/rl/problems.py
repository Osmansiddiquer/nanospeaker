"""
Problem sources for RLVR: every problem carries a machine-checkable target.

Tiers are difficulty dials, not fixed sets -- the curriculum walks them as
pass-rates rise. All generation is seeded and cheap; the code bank is sampled
from OpenCodeInstruct rows that ship their own unit tests.
"""
import json
import random

from ..data.build_arith import add_steps, sample_pair          # noqa: F401 (parity)
from ..data.build_arith2 import mul2_steps                     # noqa: F401


def arith_problem(rng, tier=0):
    """tier 0: 1-2 digit; 1: 2-3 digit; 2: expression chains; 3: algebra-lite."""
    if tier <= 1:
        hi = 99 if tier == 0 else 999
        a, b = rng.randint(2, hi), rng.randint(2, 99 if tier else 12)
        op = rng.choice(["+", "-", "x"])
        res = a + b if op == "+" else a - b if op == "-" else a * b
        return {"prompt": f"What is {a} {op} {b}?", "answer": str(res), "kind": "arith"}
    if tier == 2:
        n = rng.randint(2, 4)
        terms = [str(rng.randint(2, 99))]
        for _ in range(n - 1):
            terms += [rng.choice(["+", "-", "*"]), str(rng.randint(2, 49))]
        expr = " ".join(terms)
        if rng.random() < 0.4 and n >= 2:
            # Parenthesize a whole prefix: terms alternate operand/operator, so the
            # prefix must end on an OPERAND (odd index count) or the paren closes
            # after an operator -- "(47 *) 15" -- and eval() rightly refuses.
            ops = rng.randrange(1, n)              # operators inside the parens
            cut = 2 * ops + 1                      # number of tokens: operands+ops
            expr = "(" + " ".join(terms[:cut]) + ") " + " ".join(terms[cut:])
            expr = expr.strip()
        return {"prompt": f"What is {expr}?", "answer": str(eval(expr.replace('x', '*'))),
                "kind": "arith"}
    a, x, b = rng.randint(2, 12), rng.randint(2, 30), rng.randint(1, 50)
    c = a * x + b
    return {"prompt": f"Solve for x: {a}x + {b} = {c}", "answer": str(x), "kind": "arith"}


def load_code_bank(path="data/rl/code_bank.jsonl", limit=None):
    """Rows: {prompt, tests} where tests is executable python appended to a solution."""
    out = []
    for line in open(path):
        out.append(json.loads(line))
        if limit and len(out) >= limit:
            break
    return out


CALC_SCHEMA = {"type": "function", "function": {
    "name": "calculator", "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}
LOOKUP_SCHEMA = {"type": "function", "function": {
    "name": "lookup", "description": "Look up a fact by key.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"]}}}


TOOL_SYS = ("You are a function calling AI model. You are provided with function "
            "signatures within <tools></tools> XML tags.You may call one or more "
            "functions to assist with the user query.Here are the available tools:"
            "<tools>\n{schema}\n</tools>For each function call return a json object "
            "with function name and arguments within <tool_call></tool_call> XML tags.")


def tool_problem(rng, facts):
    """Synthetic deterministic tools: reward can check call AND result usage.

    The schema MUST ride in a system turn -- the model scores 0.90 on schema'd
    tool prompts and 0.00 without one; a bare instruction is out of distribution.
    """
    if rng.random() < 0.5:
        a, b = rng.randint(100, 9999), rng.randint(100, 9999)
        schema, user = CALC_SCHEMA, f"What is {a} * {b}? Use the calculator."
        gold = {"name": "calculator", "arg_keys": ["expression"]}
        result = str(a * b)
    else:
        q, ans, _ = rng.choice(facts)
        schema, user = LOOKUP_SCHEMA, f"Look this up for me: {q}"
        gold = {"name": "lookup", "arg_keys": ["key"]}
        result = ans
    return {"kind": "tool", "schema": schema, "gold_call": gold,
            "tool_result": result, "answer": result,
            "messages": [{"role": "system",
                          "content": TOOL_SYS.format(schema=json.dumps([schema]))},
                         {"role": "user", "content": user}],
            "prompt": user}
