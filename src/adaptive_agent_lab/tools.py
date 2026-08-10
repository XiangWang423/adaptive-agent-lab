from __future__ import annotations

import ast
import operator

from .core import Tool


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Expression contains an unsupported operation")


def calculate(expression: str) -> int | float:
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    return _evaluate(ast.parse(expression, mode="eval").body)


def word_count(text: str) -> int:
    words = text.split()
    return len(words)


_KNOWLEDGE = {
    "capital:france": "Paris",
    "capital:japan": "Tokyo",
    "language:brazil": "Portuguese",
}


def lookup(key: str) -> str:
    try:
        return _KNOWLEDGE[key.strip().lower()]
    except KeyError as exc:
        raise KeyError(f"No value for key: {key}") from exc


def default_tools() -> list[Tool]:
    return [
        Tool(
            name="calculator",
            description="Evaluate a restricted arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=calculate,
        ),
        Tool(
            name="lookup",
            description="Look up a value in a small deterministic knowledge base.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            handler=lookup,
        ),
        Tool(
            name="word_count",
            description="Count whitespace-separated words in text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=word_count,
        ),
    ]
