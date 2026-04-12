"""
core/action_sim.py — Shared calculator action simulation helpers.

These helpers mirror the behavior of icalc/script.js closely enough for
planning, recovery labeling, and rollout reward shaping.
"""

SMART_BACKSPACE_SUFFIXES = (
    "sqrt(",
    "sin(",
    "cos(",
    "tan(",
    "log(",
    "ln(",
    "inv(",
)

FUNCTION_KEYS = {"sin", "cos", "tan", "log", "ln", "sqrt", "inv"}
KEY_DISPLAY = {"pi": "π"}


def apply_key_to_readout(current: str, key: str) -> str:
    """Return the simulated readout after applying *key* to *current*."""
    if key in ("m", "Enter", "="):
        return current
    if key == "Backspace":
        for suffix in SMART_BACKSPACE_SUFFIXES:
            if current.endswith(suffix):
                return current[: -len(suffix)]
        return current[:-1] if current else ""
    if key == "Escape":
        return ""
    text = KEY_DISPLAY.get(key, key)
    if key in FUNCTION_KEYS:
        text += "("
    return current + text


def is_recoverable_by_single_backspace(current: str, key: str) -> bool:
    """Return True when *key* can be undone cleanly with one Backspace."""
    after = apply_key_to_readout(current, key)
    if after == current:
        return False
    return apply_key_to_readout(after, "Backspace") == current
