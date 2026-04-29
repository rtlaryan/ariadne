"""Evaluation dataclasses for diagnostic calculator suites and episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    task_id: str
    expression: str
    task_canonical: str
    bucket: str
    stratum: str
    angle_mode: str
    expected_value: float | None
    expected_error: str | None
    oracle_plan: list[str]
    seed: int
    features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.expression

    @property
    def index(self) -> int:
        return self.seed

    @property
    def oracle_plan_length(self) -> int:
        return len(self.oracle_plan)

    @property
    def scientific(self) -> bool:
        return "scientific_mode" in self.features or self.metadata.get("scientific", False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "task": self.expression,
                "index": self.seed,
                "oracle_plan_length": len(self.oracle_plan),
                "scientific": self.scientific,
            }
        )
        data.update(self.metadata)
        return data


@dataclass(frozen=True)
class EpisodeScore:
    task_id: str
    entry_success: bool
    result_success: bool
    success: bool
    expected_value: float | None
    observed_value: float | None
    expected_error: str | None
    observed_error: str | None
    first_divergence_step: int | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
