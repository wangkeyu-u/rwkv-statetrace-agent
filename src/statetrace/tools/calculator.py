"""Deterministic arithmetic evaluation over a tiny AST allow-list."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any, cast

from ..models import ErrorCode, JSONValue, Observation
from .base import Tool, ToolContext, ToolError, require_string

BinaryOperator = Callable[[int | float, int | float], int | float]
UnaryOperator = Callable[[int | float], int | float]
BIN_OPS: dict[type[ast.operator], BinaryOperator] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS: dict[type[ast.unaryop], UnaryOperator] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate bounded arithmetic using numbers, parentheses, and safe operators."
    schema: dict[str, JSONValue] = {"expression": "arithmetic string, at most 200 characters"}

    def run(self, arguments: dict[str, Any], context: ToolContext) -> Observation:
        expression = require_string(arguments, "expression")
        if len(expression) > 200:
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "expression is longer than 200 characters.")
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 64:
                raise ValueError("expression is too complex")
            value = self._evaluate(tree.body)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, f"Invalid arithmetic expression: {exc}") from exc
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "Expression did not produce a number.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolError(ErrorCode.INVALID_ARGUMENTS, "Result must be finite.")
        return Observation(status="success", message="Calculation completed.", data={"result": value})

    @classmethod
    def _evaluate(cls, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return cast(int | float, node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            return UNARY_OPS[type(node.op)](cls._evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
            left = cls._evaluate(node.left)
            right = cls._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent is too large")
            result = BIN_OPS[type(node.op)](left, right)
            if abs(result) > 10**100:
                raise ValueError("result is too large")
            return result
        raise ValueError(f"unsupported syntax: {type(node).__name__}")
