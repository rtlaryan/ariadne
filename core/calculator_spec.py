"""Canonical calculator capability specification for Ariadne and iCalc.

This module is the single Python source of truth for model-facing calculator
keys. The browser can keep display labels such as ``÷`` and ``AC``; datasets,
oracles, policies, and evaluators should use these canonical keys.
"""

from __future__ import annotations

DIGITS = frozenset("0123456789")
BASIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
SCIENTIFIC_FUNCTIONS = frozenset({"sin", "cos", "tan", "log", "ln", "sqrt", "inv"})
CONSTANTS = frozenset({"pi", "e"})
POSTFIX_OPERATORS = frozenset({"!"})
INFIX_SCIENTIFIC_OPERATORS = frozenset({"^"})
GROUPING = frozenset({"(", ")"})
CONTROL_ACTIONS = frozenset({"Enter", "Backspace", "Escape", "m"})
ANGLE_ACTIONS = frozenset({"deg"})

BASIC_ACTIONS = DIGITS | BASIC_OPERATORS | frozenset({".", "Enter", "Backspace", "Escape", "m"})
SCIENTIFIC_ACTIONS = (
    GROUPING
    | SCIENTIFIC_FUNCTIONS
    | CONSTANTS
    | POSTFIX_OPERATORS
    | INFIX_SCIENTIFIC_OPERATORS
    | ANGLE_ACTIONS
)
ALL_ACTIONS = BASIC_ACTIONS | SCIENTIFIC_ACTIONS

DISPLAY_TO_CANONICAL = {
    "÷": "/",
    "×": "*",
    "⌫": "Backspace",
    "AC": "Escape",
    "=": "Enter",
    "√": "sqrt",
    "π": "pi",
    "−": "-",
}

CANONICAL_TO_DISPLAY = {
    "/": "÷",
    "*": "×",
    "Backspace": "⌫",
    "Escape": "AC",
    "Enter": "=",
    "sqrt": "√",
    "pi": "π",
}


def canonicalize_key(key: str) -> str:
    """Return the model-facing key for either a display label or key token."""
    return DISPLAY_TO_CANONICAL.get(str(key), str(key))


def display_for_key(key: str) -> str:
    """Return the display label for a canonical key when it differs."""
    return CANONICAL_TO_DISPLAY.get(str(key), str(key))


def is_function_key(key: str) -> bool:
    return canonicalize_key(key) in SCIENTIFIC_FUNCTIONS


def text_for_action(key: str) -> str:
    """Return expression text appended by a canonical action."""
    key = canonicalize_key(key)
    if key in SCIENTIFIC_FUNCTIONS:
        return f"{key}("
    if key == "pi":
        return "π"
    return key
