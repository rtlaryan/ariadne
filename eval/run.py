"""Standalone CLI for live unseen-only evaluation."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import yaml

from ariadne.eval.common import (
    build_seen_task_set,
    discover_tokenizer,
    infer_checkpoint_context,
    resolve_checkpoint,
)
from ariadne.eval.reporting import (
    load_episode_records,
    summarize_episodes,
    write_combined_records,
    write_summary_json,
    write_summary_md,
)
from ariadne.eval.suite import build_eval_suite, write_manifest_json, write_suite_jsonl
from ariadne.orchestrate import HERE, _run_workers


def _resolve_exp_dir(config_path: str, cfg: dict, exp_dir: Optional[str] = None) -> str:
    if exp_dir:
        return os.path.abspath(exp_dir)
    base_dir = cfg.get("base_save_dir", "runs")
    if not os.path.isabs(base_dir):
        cfg_dir = os.path.dirname(os.path.abspath(config_path))
        base_dir = os.path.join(cfg_dir, "..", base_dir)
    return os.path.abspath(os.path.join(base_dir, cfg.get("experiment_name", "exp")))


def _resolve_output_dir(exp_dir: str, output_dir: str, suite: str) -> str:
    if os.path.isabs(output_dir):
        base = output_dir
    else:
        base = os.path.abspath(os.path.join(exp_dir, output_dir))
    return os.path.join(base, suite)


def _load_summary_metadata(summary_path: str) -> dict | None:
    try:
        with open(summary_path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    meta = payload.get("metadata")
    return meta if isinstance(meta, dict) else None


def run_evaluation(cfg: dict, exp_dir: str, **overrides) -> dict:
    eval_cfg = dict(cfg.get("evaluation", {}))
    eval_cfg.update({k: v for k, v in overrides.items() if v is not None})

    suite_name = eval_cfg.get("suite", "standard")
    checkpoint_spec = eval_cfg.get("checkpoint", "latest")
    model_path = eval_cfg.get("model_path") or resolve_checkpoint(exp_dir, checkpoint_spec)
    tokenizer_path = discover_tokenizer(model_path, exp_dir)
    output_dir = _resolve_output_dir(exp_dir, eval_cfg.get("output_dir", "evaluation"), suite_name)

    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.isfile(summary_path) and not cfg.get("overwrite", False):
        meta = _load_summary_metadata(summary_path) or {}
        print(
            f"[Eval] Summary already exists at {output_dir} "
            f"(suite={meta.get('suite', suite_name)}). Skipping."
        )
        return {"output_dir": output_dir, "summary_path": summary_path}

    os.makedirs(output_dir, exist_ok=True)
    worker_dir = os.path.join(output_dir, "worker_outputs")
    os.makedirs(worker_dir, exist_ok=True)

    context = infer_checkpoint_context(cfg, exp_dir, model_path, tokenizer_path)
    seen_tasks = build_seen_task_set(context.seen_data_roots)
    suite = build_eval_suite(
        seen_tasks=seen_tasks,
        preset=suite_name,
        max_training_depth=context.max_training_depth,
    )
    if not suite:
        raise RuntimeError("Evaluation suite generation produced zero tasks.")
    suite_path = os.path.join(output_dir, "suite.jsonl")
    write_suite_jsonl(suite_path, suite)
    write_manifest_json(
        os.path.join(output_dir, "manifest.json"),
        preset=suite_name,
        suite=suite,
        seen_task_count=len(seen_tasks),
        metadata={"model_path": model_path, "tokenizer_path": tokenizer_path},
    )

    if bool(eval_cfg.get("dry_run_suite", False)):
        print(f"[Eval] Dry-run suite written to {output_dir}")
        return {
            "output_dir": output_dir,
            "suite_path": suite_path,
            "manifest_path": os.path.join(output_dir, "manifest.json"),
            "suite_count": len(suite),
        }

    workers = max(1, min(int(eval_cfg.get("workers", 8)), len(suite)))
    remote_clients = bool(eval_cfg.get("remote_clients", False))
    base_port = int(eval_cfg.get("base_port", 9000))
    cmds = []
    for wid in range(workers):
        cmds.append(
            [
                os.environ.get("PYTHON", sys.executable),
                "-m",
                "ariadne.eval.worker",
                "--model-path",
                model_path,
                "--tokenizer-path",
                tokenizer_path,
                "--suite-path",
                suite_path,
                "--output-dir",
                worker_dir,
                "--port",
                str(base_port + wid),
                "--shard-id",
                str(wid),
                "--total-shards",
                str(workers),
                "--max-steps-multiplier",
                str(eval_cfg.get("max_steps_multiplier", 2.0)),
                "--decode",
                eval_cfg.get("decode_mode", "greedy"),
            ]
        )

    client_cmd = None
    client_cwd = None
    if not remote_clients:
        client_cmd = [
            os.environ.get("PYTHON", sys.executable),
            "client_runner.py",
            "--server-ip",
            "127.0.0.1",
            "--workers",
            str(workers),
            "--worker-offset",
            str(base_port - 9000),
        ]
        if bool(eval_cfg.get("headless", True)):
            client_cmd.append("--headless")
        runner = os.path.abspath(os.path.join(HERE, "..", "icalc", "client_runner.py"))
        client_cwd = os.path.dirname(runner)

    _run_workers(
        agent_cmds=cmds,
        target_episodes=len(suite),
        phase_label="Evaluation",
        remote_clients=remote_clients,
        client_cmd=client_cmd,
        client_cwd=client_cwd,
        idle_timeout=int(eval_cfg.get("idle_timeout", 120)),
        timeout_safety=float(eval_cfg.get("timeout_safety_multiplier", 3.0)),
    )

    episode_files = sorted(glob.glob(os.path.join(worker_dir, "eval_episodes_*.ndjson")))
    records = load_episode_records(episode_files)
    if not records:
        raise RuntimeError(f"No evaluation records were produced in {worker_dir}")
    summary = summarize_episodes(records)
    metadata = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "phase": context.phase,
        "phase_iteration": context.phase_iteration,
        "suite": suite_name,
        "remote_clients": remote_clients,
        "base_port": base_port,
        "seen_task_count": len(seen_tasks),
        "max_training_depth": context.max_training_depth,
        "seen_data_roots": context.seen_data_roots,
    }

    write_combined_records(os.path.join(output_dir, "episodes.ndjson"), records)
    write_summary_json(os.path.join(output_dir, "summary.json"), summary, metadata=metadata)
    write_summary_md(os.path.join(output_dir, "summary.md"), summary, metadata=metadata)

    print(
        f"[Eval] Headline success ({summary['overall']['headline_bucket']}): "
        f"{summary['overall']['headline_success_rate']:.1%}"
    )
    return {"output_dir": output_dir, "summary": summary, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ariadne evaluation runner")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument("--exp-dir", default=None, help="Explicit experiment directory override")
    parser.add_argument("--model-path", default=None, help="Exact checkpoint path to evaluate")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint spec, defaults to evaluation.checkpoint or latest")
    parser.add_argument("--suite", default=None, choices=["smoke", "standard", "full"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--base-port", type=int, default=None)
    parser.add_argument("--remote-clients", dest="remote_clients", action="store_true")
    parser.add_argument("--local-clients", dest="remote_clients", action="store_false")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--headed", dest="headless", action="store_false")
    parser.add_argument("--decode-mode", default=None, choices=["greedy", "sample", "epsilon_greedy"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps-multiplier", type=float, default=None)
    parser.add_argument("--dry-run-suite", dest="dry_run_suite", action="store_true", help="Only write suite.jsonl and manifest.json; do not start workers or clients")
    parser.set_defaults(headless=None, remote_clients=None, dry_run_suite=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_dir = _resolve_exp_dir(args.config, cfg, args.exp_dir)
    run_evaluation(
        cfg,
        exp_dir,
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        suite=args.suite,
        workers=args.workers,
        base_port=args.base_port,
        remote_clients=args.remote_clients,
        headless=args.headless,
        decode_mode=args.decode_mode,
        output_dir=args.output_dir,
        max_steps_multiplier=args.max_steps_multiplier,
        dry_run_suite=args.dry_run_suite,
    )


if __name__ == "__main__":
    main()
