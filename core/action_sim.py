"""
core/action_sim.py — Shared calculator action simulation helpers.

These helpers mirror the expression-building behavior of icalc/script.js closely
enough for planning, recovery labeling, and rollout/eval diagnostics.
"""

from ariadne.core.calculator_spec import SCIENTIFIC_FUNCTIONS, canonicalize_key, text_for_action

SMART_BACKSPACE_SUFFIXES = tuple(f"{func}(" for func in sorted(SCIENTIFIC_FUNCTIONS, key=len, reverse=True))
CONTROL_NOOP_KEYS = {"m", "Enter", "="}


def apply_key_to_readout(current: str, key: str) -> str:
    """Return the simulated readout after applying *key* to *current*."""
    key = canonicalize_key(key)
    if key in CONTROL_NOOP_KEYS or key == "deg":
        return current
    if key == "Backspace":
        for suffix in SMART_BACKSPACE_SUFFIXES:
            if current.endswith(suffix):
                return current[: -len(suffix)]
        return current[:-1] if current else ""
    if key == "Escape":
        return ""
    return current + text_for_action(key)


def simulate_plan(plan: list[str], initial: str = "") -> str:
    readout = initial
    for key in plan:
        readout = apply_key_to_readout(readout, key)
    return readout


def is_recoverable_by_single_backspace(current: str, key: str) -> bool:
    """Return True when *key* can be undone cleanly with one Backspace."""
    after = apply_key_to_readout(current, key)
    if after == current:
        return False
    return apply_key_to_readout(after, "Backspace") == current
