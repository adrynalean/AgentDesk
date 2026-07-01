"""A safe arithmetic tool (real, working — the simplest end-to-end tool)."""
from __future__ import annotations

import ast
import operator as op

_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "3 * (4 + 1)"."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


calculator.description = "Evaluate a basic arithmetic expression."
