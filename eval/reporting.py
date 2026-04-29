"""Aggregation and reporting helpers for live diagnostic evaluation runs."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable


def load_episode_records(paths: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        try:
            with open(path, "r") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return records


def _rate(rows: list[dict], key: str) -> float:
    return sum(int(bool(r.get(key, False))) for r in rows) / float(len(rows)) if rows else 0.0


def _summary_for(rows: list[dict]) -> dict:
    episodes = len(rows)
    steps = sum(int(r.get("num_steps", 0)) for r in rows)
    divergences = sum(1 for r in rows if r.get("first_divergence_step") is not None)
    return {
        "episodes": episodes,
        "successes": sum(int(bool(r.get("success", False))) for r in rows),
        "success_rate": _rate(rows, "success"),
        "entry_success_rate": _rate(rows, "entry_success"),
        "result_success_rate": _rate(rows, "result_success"),
        "avg_steps": (steps / float(episodes)) if episodes else 0.0,
        "divergence_rate": (divergences / float(episodes)) if episodes else 0.0,
    }


def summarize_episodes(records: list[dict]) -> dict:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    by_feature: dict[str, list[dict]] = defaultdict(list)
    by_reason: dict[str, int] = defaultdict(int)

    for rec in records:
        by_bucket[rec.get("bucket", "unknown")].append(rec)
        stratum_key = f"{rec.get('bucket', 'unknown')}::{rec.get('stratum', '')}"
        by_stratum[stratum_key].append(rec)
        by_reason[rec.get("termination_reason", "unknown")] += 1
        for feature in rec.get("features", []) or rec.get("metadata", {}).get("features", []):
            by_feature[str(feature)].append(rec)

    bucket_summary = {bucket: _summary_for(rows) for bucket, rows in sorted(by_bucket.items())}
    stratum_summary = {stratum: _summary_for(rows) for stratum, rows in sorted(by_stratum.items())}
    feature_summary = {feature: _summary_for(rows) for feature, rows in sorted(by_feature.items())}
    overall = _summary_for(records)
    overall.update(
        {
            "headline_bucket": "arithmetic",
            "headline_success_rate": bucket_summary.get("arithmetic", {}).get("success_rate", 0.0),
            "termination_reasons": dict(sorted(by_reason.items())),
        }
    )
    return {"overall": overall, "by_bucket": bucket_summary, "by_stratum": stratum_summary, "by_feature": feature_summary}


def write_summary_json(path: str, summary: dict, metadata: dict | None = None) -> None:
    payload = dict(summary)
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_summary_md(path: str, summary: dict, metadata: dict | None = None) -> None:
    lines = ["# Evaluation Summary", ""]
    if metadata:
        lines.append(f"- Model: `{metadata.get('model_path', '')}`")
        lines.append(f"- Suite preset: `{metadata.get('suite', '')}`")
        lines.append(f"- Seen tasks excluded: `{metadata.get('seen_task_count', 0)}`")
        lines.append("")

    overall = summary.get("overall", {})
    lines.append("## Overall")
    lines.append(f"- Headline (`arithmetic`) success: {overall.get('headline_success_rate', 0.0):.1%}")
    lines.append(
        f"- All buckets success: {overall.get('success_rate', 0.0):.1%} "
        f"({overall.get('successes', 0)}/{overall.get('episodes', 0)})"
    )
    lines.append(f"- Entry success: {overall.get('entry_success_rate', 0.0):.1%}")
    lines.append(f"- Result success: {overall.get('result_success_rate', 0.0):.1%}")
    if overall.get("termination_reasons"):
        lines.append(f"- Termination reasons: `{overall.get('termination_reasons')}`")
    lines.append("")

    lines.append("## Buckets")
    for bucket, bucket_summary in summary.get("by_bucket", {}).items():
        lines.append(
            f"- `{bucket}`: success={bucket_summary['success_rate']:.1%}, "
            f"entry={bucket_summary['entry_success_rate']:.1%}, "
            f"result={bucket_summary['result_success_rate']:.1%}, "
            f"avg_steps={bucket_summary['avg_steps']:.2f}, "
            f"divergence={bucket_summary['divergence_rate']:.1%}"
        )

    lines.append("")
    lines.append("## Strata")
    for stratum, row in summary.get("by_stratum", {}).items():
        lines.append(f"- `{stratum}`: {row['success_rate']:.1%} ({row['successes']}/{row['episodes']})")

    if summary.get("by_feature"):
        lines.append("")
        lines.append("## Features")
        for feature, row in summary.get("by_feature", {}).items():
            lines.append(f"- `{feature}`: {row['success_rate']:.1%} ({row['successes']}/{row['episodes']})")

    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_combined_records(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
