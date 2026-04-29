"""Scoring helpers for Ariadne diagnostic evaluation episodes."""

from __future__ import annotations

from ariadne.core.calculator_eval import evaluate_expression, normalize_expression
from ariadne.eval.schema import EpisodeScore, EvalCase


def _norm(text: str | None) -> str:
    return normalize_expression("" if text is None else str(text))


def _latest_history_expr(state: dict) -> str:
    history = state.get("history", [])
    return str(history[-1]) if history else ""


def _readout(state: dict) -> str:
    return str(state.get("readout", ""))


def first_divergence_step(goal: str, steps: list[dict]) -> int | None:
    goal_norm = _norm(goal)
    for step in steps:
        action = step.get("action")
        if action in ("Enter", "="):
            if _norm(step.get("readout_before", "")) != goal_norm:
                return step.get("step_index")
            continue
        if action in ("Escape", "m", "deg"):
            continue
        if not step.get("valid_prefix_after", True):
            return step.get("step_index")
    return None


def score_episode(
    case: EvalCase | dict,
    *,
    terminal_state: dict,
    steps: list[dict],
    tolerance: float = 1e-9,
) -> EpisodeScore:
    if isinstance(case, dict):
        case_obj = EvalCase(
            task_id=str(case.get("task_id", case.get("index", "case"))),
            expression=str(case.get("expression", case.get("task", ""))),
            task_canonical=str(case.get("task_canonical", _norm(case.get("task", "")))),
            bucket=str(case.get("bucket", "unknown")),
            stratum=str(case.get("stratum", "")),
            angle_mode=str(case.get("angle_mode", "deg")),
            expected_value=case.get("expected_value"),
            expected_error=case.get("expected_error"),
            oracle_plan=list(case.get("oracle_plan", [])),
            seed=int(case.get("seed", case.get("index", 0))),
            features=list(case.get("features", [])),
            metadata=dict(case.get("metadata", {})),
        )
    else:
        case_obj = case

    history_expr = _latest_history_expr(terminal_state)
    entry_success = _norm(history_expr) == case_obj.task_canonical
    observed_error = None
    observed_value = None

    if case_obj.expected_error:
        readout = _readout(terminal_state)
        result_success = readout == "Error" or bool(terminal_state.get("error"))
        observed_error = terminal_state.get("error") or ("Error" if readout == "Error" else None)
        success = result_success
        reason = "expected_error" if success else "missing_error"
    else:
        result = evaluate_expression(_readout(terminal_state), angle_mode=case_obj.angle_mode)
        if result.ok:
            observed_value = result.value
            result_success = (
                case_obj.expected_value is not None
                and observed_value is not None
                and abs(observed_value - float(case_obj.expected_value)) <= tolerance
            )
        else:
            result_success = False
            observed_error = result.error
        success = entry_success and result_success
        reason = "completed" if success else ("wrong_entry" if not entry_success else "wrong_result")

    return EpisodeScore(
        task_id=case_obj.task_id,
        entry_success=entry_success,
        result_success=result_success,
        success=success,
        expected_value=case_obj.expected_value,
        observed_value=observed_value,
        expected_error=case_obj.expected_error,
        observed_error=observed_error,
        first_divergence_step=first_divergence_step(case_obj.expression, steps),
        reason=reason,
    )
