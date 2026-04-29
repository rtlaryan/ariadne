"""Shared helpers for evaluation planning, suite prep, and reporting."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_TASK_NORMALIZE = {
    " ": "",
    "÷": "/",
    "×": "*",
    "π": "pi",
    "√": "sqrt",
    "−": "-",
}

_DAGGER_RE = re.compile(r"dagger_iter_(\d+)")
_RL_RE = re.compile(r"rl_iter_(\d+)")


@dataclass(frozen=True)
class CheckpointContext:
    exp_dir: str
    model_path: str
    tokenizer_path: str
    phase: str
    phase_iteration: int
    seen_data_roots: list[str]
    max_training_depth: int


def canonicalize_task(task: str | None) -> str:
    """Normalize a task string for exact unseen-ness checks."""
    text = "" if task is None else str(task)
    for src, dst in _TASK_NORMALIZE.items():
        text = text.replace(src, dst)
    return text


def _iter_task_records(paths: Iterable[str]) -> Iterable[dict]:
    for path in paths:
        try:
            with open(path, "r") as f:
                cur_eid = object()
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    eid = obj.get("episode_id")
                    if eid is not None and eid == cur_eid:
                        continue
                    cur_eid = eid
                    yield obj
        except OSError:
            continue


def _expand_data_roots(data_roots: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for root in data_roots:
        if not root:
            continue
        if os.path.isfile(root):
            if root.endswith((".jsonl", ".ndjson")):
                paths.append(root)
            continue
        if not os.path.isdir(root):
            continue
        paths.extend(sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)))
        paths.extend(sorted(glob.glob(os.path.join(root, "**", "*.ndjson"), recursive=True)))
    return paths


def build_seen_task_set(data_roots: Iterable[str]) -> set[str]:
    """Return canonicalized task strings observed in the given data roots."""
    seen: set[str] = set()
    for obj in _iter_task_records(_expand_data_roots(data_roots)):
        task = canonicalize_task(obj.get("task"))
        if task:
            seen.add(task)
    return seen


def resolve_checkpoint(exp_dir: str, checkpoint: str | None = None) -> str:
    """Resolve a checkpoint spec to a concrete .pt path."""
    spec = checkpoint or "latest"
    if spec == "latest":
        matches = glob.glob(os.path.join(exp_dir, "**", "checkpoints", "*.pt"), recursive=True)
        if not matches:
            raise RuntimeError(f"No checkpoints found under {exp_dir}")
        return max(matches, key=os.path.getmtime)

    candidates = [spec]
    if not os.path.isabs(spec):
        candidates.append(os.path.join(exp_dir, spec))
        candidates.append(os.path.join(PACKAGE_DIR, spec))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(f"Could not resolve checkpoint: {spec}")


def discover_tokenizer(model_path: str, exp_dir: str | None = None) -> str:
    """Find tokenizer.json nearest to the checkpoint."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    search_roots = [
        model_dir,
        os.path.dirname(model_dir),
        exp_dir or "",
        PACKAGE_DIR,
    ]
    seen: set[str] = set()
    for root in search_roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        path = os.path.join(root, "tokenizer.json")
        if os.path.isfile(path):
            return path
    raise RuntimeError(f"Could not locate tokenizer.json near {model_path}")


def resolve_pretrain_data_dirs(cfg: dict) -> list[str]:
    data_dir = cfg.get("pretrain", {}).get("data_dir", "dataset")
    raw_dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
    resolved = []
    for entry in raw_dirs:
        if os.path.isabs(entry):
            resolved.append(entry)
        else:
            resolved.append(os.path.abspath(os.path.join(PACKAGE_DIR, entry)))
    return resolved


def _phase_from_model_path(model_path: str, exp_dir: str) -> tuple[str, int]:
    rel = os.path.relpath(os.path.abspath(model_path), os.path.abspath(exp_dir))
    for part in rel.split(os.sep):
        m = _RL_RE.fullmatch(part)
        if m:
            return "rl", int(m.group(1))
        m = _DAGGER_RE.fullmatch(part)
        if m:
            return "dagger", int(m.group(1))
        if part == "pre_train":
            return "pretrain", 0
    return "unknown", 0


def _existing_iter_roots(exp_dir: str, prefix: str, child_dir: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)$")
    for path in sorted(glob.glob(os.path.join(exp_dir, f"{prefix}_*"))):
        if not os.path.isdir(path):
            continue
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        root = os.path.join(path, child_dir)
        if os.path.isdir(root):
            matches.append((int(m.group(1)), root))
    return matches


def _resolve_seen_data_roots(cfg: dict, exp_dir: str, phase: str, phase_iter: int) -> list[str]:
    roots = resolve_pretrain_data_dirs(cfg)

    dagger_roots = _existing_iter_roots(exp_dir, "dagger_iter", "dagger_data")
    if phase == "dagger":
        roots.extend(root for idx, root in dagger_roots if idx <= phase_iter)
    elif phase == "rl":
        roots.extend(root for _, root in dagger_roots)
        rollout_roots = _existing_iter_roots(exp_dir, "rl_iter", "rollout_data")
        roots.extend(root for idx, root in rollout_roots if idx <= phase_iter)

    return roots


def _coerce_depth(value) -> int:
    if isinstance(value, list):
        return max((_coerce_depth(v) for v in value), default=0)
    if value is None:
        return 0
    return int(value)


def _max_relevant_depths(cfg: dict, phase: str, phase_iter: int) -> int:
    depths: list[int] = []

    dg = cfg.get("data_generation", {})
    depths.append(_coerce_depth(dg.get("max_depth", 3)))

    dagger_gen = cfg.get("dagger", {}).get("generation", {})
    dagger_depth = dagger_gen.get("max_depth", 3)
    if phase == "dagger":
        if isinstance(dagger_depth, list):
            depths.append(max((_coerce_depth(v) for v in dagger_depth[:phase_iter]), default=0))
        else:
            depths.append(_coerce_depth(dagger_depth))
    elif phase == "rl":
        depths.append(_coerce_depth(dagger_depth))
        rl_rollout = cfg.get("rl", {}).get("rollout", {})
        rl_depth = rl_rollout.get("max_depth", 3)
        if isinstance(rl_depth, list):
            depths.append(max((_coerce_depth(v) for v in rl_depth[:phase_iter]), default=0))
        else:
            depths.append(_coerce_depth(rl_depth))

    return max(depths or [3])


def infer_checkpoint_context(
    cfg: dict,
    exp_dir: str,
    model_path: str,
    tokenizer_path: Optional[str] = None,
) -> CheckpointContext:
    """Infer eval-time exposure and depth context from the checkpoint path."""
    phase, phase_iter = _phase_from_model_path(model_path, exp_dir)
    return CheckpointContext(
        exp_dir=os.path.abspath(exp_dir),
        model_path=os.path.abspath(model_path),
        tokenizer_path=tokenizer_path or discover_tokenizer(model_path, exp_dir),
        phase=phase,
        phase_iteration=phase_iter,
        seen_data_roots=_resolve_seen_data_roots(cfg, exp_dir, phase, phase_iter),
        max_training_depth=_max_relevant_depths(cfg, phase, phase_iter),
    )
