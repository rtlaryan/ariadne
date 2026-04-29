"""Feature-first deterministic diagnostic evaluation suite generation."""

from __future__ import annotations

import json
from dataclasses import asdict

from ariadne.agents.oracle import Oracle
from ariadne.core.calculator_eval import normalize_expression
from ariadne.eval.schema import EvalCase


SUITE_PRESETS = {
    "smoke": {
        "arithmetic": 20,
        "decimals_constants": 10,
        "functions": 10,
        "angle_modes": 5,
        "error_states": 5,
    },
    "standard": {
        "arithmetic": 200,
        "decimals_constants": 100,
        "functions": 100,
        "angle_modes": 50,
        "error_states": 50,
    },
    "full": {
        "arithmetic": 800,
        "decimals_constants": 400,
        "functions": 400,
        "angle_modes": 200,
        "error_states": 200,
    },
}

# Backward-compatible name used by current tests/callers while carrying richer fields.
EvalTaskSpec = EvalCase


def _category_for_bucket(bucket: str, offset: int) -> str:
    if bucket == "arithmetic":
        return "decimal_arithmetic" if offset % 3 == 0 else "mixed"
    if bucket == "decimals_constants":
        return "constants" if offset % 2 else "decimal_arithmetic"
    if bucket == "functions":
        return "mixed"
    if bucket == "angle_modes":
        return "angle_trig"
    return "mixed"


def _invalid_case(seed: int, bucket: str, stratum: str, oracle: Oracle) -> EvalCase:
    invalid_exprs = [
        f"sqrt(-{(seed % 97) + 1})",
        f"log(-{(seed % 89) + 1})",
        f"{(seed % 83) + 1}/0",
        "inv(0)",
    ]
    expr = invalid_exprs[seed % len(invalid_exprs)]
    return EvalCase(
        task_id=f"{bucket}:{seed}",
        expression=expr,
        task_canonical=normalize_expression(expr),
        bucket=bucket,
        stratum=stratum,
        angle_mode="deg",
        expected_value=None,
        expected_error="domain_error",
        oracle_plan=oracle.plan(expr, current_mode="basic"),
        seed=seed,
        features=["invalid", stratum],
        metadata={"min_depth": 1, "max_depth": 1},
    )


def build_eval_suite(
    *,
    seen_tasks: set[str],
    preset: str,
    max_training_depth: int,
    oracle: Oracle | None = None,
) -> list[EvalCase]:
    """Build a deterministic unseen-only feature diagnostic suite."""
    if preset not in SUITE_PRESETS:
        raise KeyError(f"Unknown evaluation preset: {preset}")

    oracle = oracle or Oracle(profile="rich_v1")
    counts = SUITE_PRESETS[preset]
    admitted = set(seen_tasks)
    suite: list[EvalCase] = []
    base_depth = max(1, int(max_training_depth))
    seed_base = {"arithmetic": 1_000_000, "decimals_constants": 2_000_000, "functions": 3_000_000, "angle_modes": 4_000_000, "error_states": 5_000_000}

    for bucket, count in counts.items():
        collected = 0
        attempts = 0
        while collected < count:
            attempts += 1
            if attempts > count * 1000 + 1000:
                raise RuntimeError(f"Could not fill eval bucket {bucket}")
            seed = seed_base[bucket] + attempts
            if bucket == "error_states":
                case = _invalid_case(seed, bucket, "invalid_math", oracle)
            else:
                category = _category_for_bucket(bucket, attempts)
                task = oracle.generate_task(
                    seed=seed,
                    category=category,
                    min_depth=1,
                    max_depth=base_depth + (1 if bucket in {"functions", "angle_modes"} else 0),
                    profile="rich_v1",
                )
                canonical = task.task_canonical
                if canonical in admitted:
                    continue
                features = list(task.metadata.get("features", [])) if task.metadata else []
                case = EvalCase(
                    task_id=f"{bucket}:{seed}",
                    expression=task.expression,
                    task_canonical=canonical,
                    bucket=bucket,
                    stratum=category,
                    angle_mode=task.angle_mode,
                    expected_value=task.expected.value,
                    expected_error=task.expected.error,
                    oracle_plan=task.plan,
                    seed=seed,
                    features=features,
                    metadata={"min_depth": 1, "max_depth": base_depth, "metadata": task.metadata or {}},
                )
            if case.task_canonical in admitted:
                continue
            admitted.add(case.task_canonical)
            suite.append(case)
            collected += 1

    return suite


def write_suite_jsonl(path: str, suite: list[EvalCase]) -> None:
    with open(path, "w") as f:
        for spec in suite:
            if hasattr(spec, "to_dict"):
                row = spec.to_dict()
            else:
                row = asdict(spec)
            f.write(json.dumps(row) + "\n")


def write_manifest_json(path: str, *, preset: str, suite: list[EvalCase], seen_task_count: int, metadata: dict | None = None) -> None:
    payload = {
        "preset": preset,
        "case_count": len(suite),
        "seen_task_count": seen_task_count,
        "buckets": {},
    }
    for case in suite:
        payload["buckets"][case.bucket] = payload["buckets"].get(case.bucket, 0) + 1
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
