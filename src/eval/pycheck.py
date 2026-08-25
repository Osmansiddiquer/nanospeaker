"""
check_python: syntax + undefined-name checker. One implementation, three users:
SFT-T data generation (real diagnostics for planted bugs), the future RLVR reward,
and the serving-side tool the model will actually call.
"""
import ast
import builtins

_BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "self", "cls"}


def check_python(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "errors": [f"SyntaxError: {e.msg} at line {e.lineno}"]}
    defined = set(_BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if hasattr(node, "args"):
                for a in (node.args.args + node.args.kwonlyargs
                          + ([node.args.vararg] if node.args.vararg else [])
                          + ([node.args.kwarg] if node.args.kwarg else [])):
                    defined.add(a.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            tgt = node.target if hasattr(node, "target") else None
            for n in ast.walk(tgt) if tgt is not None else []:
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Lambda,)):
            for a in node.args.args:
                defined.add(a.arg)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            defined.update(node.names)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                errors.append(f"undefined name '{node.id}' at line {node.lineno}")
    seen, uniq = set(), []
    for e in errors:
        k = e.split(" at ")[0]
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return {"ok": not uniq, "errors": uniq[:5]}
