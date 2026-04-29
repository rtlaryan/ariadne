"""Python evaluator matching iCalc's calculator expression semantics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class CalculatorEvaluationError(ValueError):
    """Raised when an expression cannot be evaluated safely."""


@dataclass(frozen=True)
class EvaluationResult:
    ok: bool
    value: float | None
    error: str | None = None
    normalized_expression: str = ""


def normalize_expression(expr: str) -> str:
    text = str(expr).replace(" ", "")
    replacements = {
        "÷": "/",
        "×": "*",
        "π": "pi",
        "√": "sqrt",
        "−": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _factorial(n: float) -> float:
    if int(n) != n or n < 0:
        raise CalculatorEvaluationError("factorial_domain")
    return float(math.factorial(int(n)))


def _replace_factorials(expr: str) -> str:
    # iCalc supports simple numeric/parenthesized factorials; iterative regex is
    # enough for generated task expressions and keeps evaluation transparent.
    pattern = re.compile(r"(\d+(?:\.\d+)?|\([^()]+\))!")
    while True:
        new = pattern.sub(r"fact(\1)", expr)
        if new == expr:
            return new
        expr = new


def _safe_env(angle_mode: str) -> dict[str, object]:
    def _angle(x: float) -> float:
        return math.radians(x) if angle_mode == "deg" else x

    def _tan(x: float) -> float:
        value = math.tan(_angle(x))
        if abs(value) > 1e12:
            raise CalculatorEvaluationError("domain_error")
        return value

    return {
        "__builtins__": {},
        "pi": math.pi,
        "e": math.e,
        "sin": lambda x: math.sin(_angle(x)),
        "cos": lambda x: math.cos(_angle(x)),
        "tan": _tan,
        "log": math.log10,
        "ln": math.log,
        "sqrt": math.sqrt,
        "inv": lambda x: 1 / x,
        "fact": _factorial,
        "abs": abs,
    }


def evaluate_expression(
    expr: str,
    *,
    angle_mode: str = "deg",
    raise_errors: bool = False,
) -> EvaluationResult:
    """Evaluate an iCalc expression and return value/error metadata.

    The evaluator intentionally mirrors the JS calculator's expression grammar
    rather than being a full symbolic math parser.
    """
    if angle_mode not in {"deg", "rad"}:
        raise ValueError("angle_mode must be 'deg' or 'rad'")
    normalized = normalize_expression(expr)
    try:
        py_expr = _replace_factorials(normalized).replace("^", "**")
        if not re.fullmatch(r"[0-9A-Za-z_+\-*/%().,*\s]+", py_expr):
            raise CalculatorEvaluationError("invalid_token")
        value = eval(py_expr, _safe_env(angle_mode), {})  # noqa: S307 - sandboxed env, validated chars
        if not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
            raise CalculatorEvaluationError("non_finite")
        return EvaluationResult(
            ok=True,
            value=float(f"{float(value):.12g}"),
            error=None,
            normalized_expression=normalized,
        )
    except Exception as exc:  # domain, syntax, zero division, etc.
        error = str(exc) or exc.__class__.__name__
        if isinstance(exc, (ValueError, ZeroDivisionError, OverflowError)):
            error = "domain_error"
        if raise_errors:
            raise CalculatorEvaluationError(error) from exc
        return EvaluationResult(ok=False, value=None, error=error, normalized_expression=normalized)
