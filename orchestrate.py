"""
orchestrate.py — Ariadne experiment orchestrator.

Runs all four phases of the training pipeline:
  Phase 0: Data Generation (supervised / crawler)
  Phase 1: Pre-training
  Phase 2: DAgger Loop
  Phase 3: RL Loop

Each phase is an idempotent function: if outputs already exist it is
skipped unless the config says otherwise.

Usage:
  python -m ariadne.orchestrate --config ariadne/configs/experiment.yaml
"""

import argparse
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Optional

import yaml

HERE       = os.path.dirname(os.path.abspath(__file__))
# Parent directory that contains the ariadne package.
# This is only for PYTHONPATH so `import ariadne` works in subprocesses.
PACKAGE_ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(directory: str) -> Optional[str]:
    """Return the most recent .pt file inside *directory*/checkpoints/, or None."""
    pattern = os.path.join(directory, "checkpoints", "*.pt")
    files   = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def _run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    """Run a subprocess, streaming output to stdout/stderr.  Raises on failure."""
    env = os.environ.copy()
    # Ensure the parent of ariadne/ is on PYTHONPATH for subprocesses.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (PACKAGE_ROOT + os.pathsep + existing).rstrip(os.pathsep)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _count_jsonl_lines(directory: str) -> int:
    """Count non-empty JSONL lines in *directory* (recursive)."""
    count = 0
    for path in glob.glob(os.path.join(directory, "**/*.jsonl"), recursive=True):
        try:
            with open(path) as f:
                count += sum(1 for l in f if l.strip())
        except OSError:
            pass
    return count


def _count_jsonl_episodes(paths: list[str]) -> int:
    """Count episode records in JSONL files.

    For step datasets, rows with the same episode_id are grouped together and
    counted once. For rollout-style JSONL where each line is already one
    episode, each distinct line contributes one record.
    """
    count = 0
    for path in paths:
        try:
            with open(path) as f:
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
                    if eid is None:
                        count += 1
                    elif eid != cur_eid:
                        count += 1
                        cur_eid = eid
        except OSError:
            pass
    return count


def _timestamp_batches(paths: list[str]) -> dict[str, list[str]]:
    """Group timestamped JSONL files by trailing numeric timestamp."""
    batches: dict[str, list[str]] = {}
    for path in paths:
        m = re.search(r"_(\d+)\.jsonl$", os.path.basename(path))
        ts = m.group(1) if m else "unknown"
        batches.setdefault(ts, []).append(path)
    return batches


def _assert_single_timestamp_batch(paths: list[str], label: str, directory: str) -> None:
    """Fail fast when a directory contains multiple timestamp batches."""
    batches = _timestamp_batches(paths)
    if len(batches) <= 1:
        return
    summary = ", ".join(
        f"{ts}: {len(batch_paths)} files"
        for ts, batch_paths in sorted(batches.items())
    )
    raise RuntimeError(
        f"{label} found multiple timestamp batches in {directory}: {summary}. "
        "Remove stale partial files or use overwrite before rerunning."
    )


def _expected_worker_episode_counts(target: int, workers: int) -> list[int]:
    """Return the target episode allocation for each worker shard."""
    base = target // workers
    rem = target % workers
    return [base + (1 if w < rem else 0) for w in range(workers)]


def _collect_worker_episode_progress(
    files: list[str],
    mode: str,
    workers: int,
    base_port: int,
) -> tuple[dict[int, int], dict[int, str], str | None]:
    """Return per-worker completed episodes, file paths, and current timestamp."""
    progress = {w: 0 for w in range(workers)}
    paths_by_worker: dict[int, str] = {}
    if not files:
        return progress, paths_by_worker, None

    batches = _timestamp_batches(files)
    timestamps = sorted(batches)
    ts = timestamps[0] if timestamps else None

    for path in files:
        m = re.search(rf"dataset_{re.escape(mode)}_(\d+)_(\d+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        port = int(m.group(1))
        worker = port - base_port
        if worker < 0 or worker >= workers:
            continue
        if worker in paths_by_worker:
            raise RuntimeError(
                f"Data Gen ({mode}) found multiple files for worker {worker} in the active batch."
            )
        paths_by_worker[worker] = path
        progress[worker] = _count_jsonl_episodes([path])

    return progress, paths_by_worker, ts


def _summarize_dagger_episodes(directory: str) -> Optional[dict]:
    """Aggregate episode-level DAgger summaries from *.ndjson sidecar files."""
    total = success = 0
    total_steps = total_recovery_steps = 0

    for path in glob.glob(os.path.join(directory, "**", "dagger_episodes_*.ndjson"), recursive=True):
        try:
            with open(path, "r") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        ep = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    success += int(bool(ep.get("success", False)))
                    total_steps += int(ep.get("num_steps", 0))
                    total_recovery_steps += int(ep.get("recovery_steps", 0))
        except OSError:
            pass

    if total == 0:
        return None

    return {
        "episodes": total,
        "successes": success,
        "success_rate": success / float(total),
        "avg_steps": total_steps / float(total),
        "avg_recovery_steps": total_recovery_steps / float(total),
    }


def _resolve_from_project_root(path: str) -> str:
    """Resolve a config path relative to the ariadne top-level directory."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(HERE, path))


def _resolve_from_package_dir(path: str) -> str:
    """Resolve a config path relative to the ariadne package directory."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(HERE, path))


def _needs_seed_model(cfg: dict) -> bool:
    """Return True when later phases require a checkpoint from pretraining."""
    return bool(cfg.get("dagger", {}).get("enabled", False) or cfg.get("rl", {}).get("enabled", False))


# ---------------------------------------------------------------------------
# Parallel worker runner — used by phases 0, 2 (gen), and 3 (rollout)
# ---------------------------------------------------------------------------

def _run_workers(
    agent_cmds:        list[list[str]],
    target_episodes:   int,
    phase_label:       str,
    remote_clients:    bool,
    client_cmd:        Optional[list[str]] = None,
    client_cwd:        Optional[str]       = None,
    idle_timeout:      int                 = 300,
    timeout_safety:    float               = 3.0,
    agent_cwd:         Optional[str]       = None,
    initial_progress:  Optional[dict[int, int]] = None,
    base_completed:    int                 = 0,
) -> int:
    """Start agent subprocess workers and optionally a client manager.

    Monitors STATUS:EPISODE_COMPLETE:N and STATUS:EXIT:... messages from
    each worker's stdout.  Displays a unified progress bar.

    Returns total completed episodes.
    """
    n_workers   = len(agent_cmds)
    out_queue:  queue.Queue = queue.Queue()
    server_procs = []
    client_procs = []

    def _reader(proc, wid):
        try:
            for raw in iter(proc.stdout.readline, b""):
                out_queue.put((wid, raw.decode("utf-8").strip()))
        except ValueError:
            pass

    # Start agent servers
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (PACKAGE_ROOT + os.pathsep + existing).rstrip(os.pathsep)
    for wid, cmd in enumerate(agent_cmds):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr,
                             cwd=agent_cwd, env=env)
        server_procs.append(p)
        threading.Thread(target=_reader, args=(p, wid), daemon=True).start()

    print(f"[Orchestrate] {phase_label}: {n_workers} workers started.")
    time.sleep(2)

    # Start clients
    if not remote_clients and client_cmd:
        cp = subprocess.Popen(client_cmd, cwd=client_cwd)
        client_procs.append(cp)
    elif remote_clients:
        print(f"[Orchestrate] Remote clients mode — waiting for external connections.")

    progress     = {i: 0 for i in range(n_workers)}
    offsets      = {i: 0 for i in range(n_workers)}
    if initial_progress:
        for i, value in initial_progress.items():
            if i in offsets:
                offsets[i] = value
    start_time   = time.time()
    measured_rate = None
    timeout      = None
    initial_offset_total = sum(initial_progress.values()) if initial_progress else 0
    remaining_target = max(0, target_episodes - base_completed - initial_offset_total)
    DIAG         = min(10 * n_workers, max(1, remaining_target // 4), 50) if remaining_target > 0 else 1

    try:
        while True:
            # Drain queue
            try:
                while True:
                    wid, line = out_queue.get_nowait()
                    if line.startswith("STATUS:EPISODE_COMPLETE:"):
                        try:
                            reported = int(line.split(":")[2])
                            progress[wid] = max(0, reported - offsets.get(wid, 0))
                        except (IndexError, ValueError):
                            pass
                    elif line.startswith("STATUS:EXIT:"):
                        parts = line.split(":")
                        try:
                            if len(parts) >= 4:
                                reported = int(parts[3])
                                progress[wid] = max(0, reported - offsets.get(wid, 0))
                        except ValueError:
                            pass
            except queue.Empty:
                pass

            run_total  = sum(progress.values())
            offset_total = sum(offsets.values())
            total      = base_completed + offset_total + run_total
            elapsed   = time.time() - start_time
            rate      = run_total / elapsed if elapsed > 0 else 0
            alive     = [p for p in server_procs if p.poll() is None]

            if run_total >= DIAG and elapsed >= 5.0 and measured_rate is None:
                measured_rate = rate
                if measured_rate > 0:
                    timeout = (remaining_target / measured_rate) * timeout_safety
                    print(
                        f"\n[Orchestrate] Diagnostic: {measured_rate:.1f} ep/s → "
                        f"timeout {timeout:.0f}s"
                    )

            eta = f"{int((target_episodes - total) / rate)}s" if rate > 0 else "…"
            pct = total / target_episodes if target_episodes else 0
            bar = ("=" * int(30 * pct)).ljust(30, "-")
            sys.stdout.write(
                f"\r[{bar}] {int(pct*100)}% | {phase_label} | "
                f"{total}/{target_episodes} ep | {rate:.1f} ep/s | ETA {eta} | "
                f"Workers {len(alive)}/{n_workers}"
            )
            sys.stdout.flush()

            if not alive:
                print(f"\n[Orchestrate] All workers exited. Total: {total}")
                break
            if timeout and elapsed > timeout:
                print(f"\n[Orchestrate] Timeout ({timeout:.0f}s) reached.")
                break
            if client_procs and client_procs[0].poll() is not None:
                print("\n[Orchestrate] Client manager exited prematurely.")
                break
            time.sleep(0.5)
    finally:
        for p in server_procs + client_procs:
            if p.poll() is None:
                p.terminate()

    return base_completed + sum(offsets.values()) + sum(progress.values())


# ---------------------------------------------------------------------------
# Phase 0: Data Generation
# ---------------------------------------------------------------------------

def phase_datagen(cfg: dict, exp_dir: str) -> Optional[str]:
    """Generate supervised / crawler data. Returns the data directory path, or None if skipped."""
    dg = cfg.get("data_generation", {})
    if not dg.get("enabled", False):
        return None

    print("=== Phase 0: Data Generation ===")

    if dg.get("experiment_specific", True):
        data_dir = os.path.join(exp_dir, dg.get("output_dir", "dataset"))
    else:
        data_dir = _resolve_from_project_root(dg.get("output_dir", "dataset"))

    mode = dg.get("mode", "supervised")
    if mode == "both":
        relevant_files = sorted(glob.glob(os.path.join(data_dir, "dataset_*.jsonl")))
        for submode in ("supervised", "crawler"):
            _assert_single_timestamp_batch(
                sorted(glob.glob(os.path.join(data_dir, f"dataset_{submode}_*.jsonl"))),
                label=f"Data Gen ({submode})",
                directory=data_dir,
            )
    else:
        relevant_files = sorted(glob.glob(os.path.join(data_dir, f"dataset_{mode}_*.jsonl")))
        _assert_single_timestamp_batch(
            relevant_files,
            label=f"Data Gen ({mode})",
            directory=data_dir,
        )

    target   = dg.get("episodes", 10000)
    workers  = dg.get("workers",   4)
    existing = _count_jsonl_episodes(relevant_files)

    if existing >= target:
        print(f"[Orchestrate] Data already exists ({existing} episodes). Skipping generation.")
        return data_dir

    basic_only       = dg.get("basic_only",     False)
    history_window   = dg.get("history_window", -1)
    remote_clients   = dg.get("remote_clients", False)
    min_depth        = dg.get("min_depth",      1)
    max_depth        = dg.get("max_depth",      3)
    base_port        = 9000
    worker_targets   = _expected_worker_episode_counts(target, workers)

    if mode == "both":
        raise RuntimeError("Resumable data generation currently supports 'supervised' or 'crawler' mode, not 'both'.")

    worker_progress, worker_paths, active_ts = _collect_worker_episode_progress(
        relevant_files, mode, workers, base_port
    )

    cmds = []
    initial_progress: dict[int, int] = {}
    base_completed = 0
    for w in range(workers):
        target_ep = worker_targets[w]
        completed_ep = min(worker_progress.get(w, 0), target_ep)
        if completed_ep >= target_ep:
            base_completed += completed_ep
            continue

        initial_progress[len(cmds)] = completed_ep
        cmd = [
            sys.executable, "-m", "ariadne.agents.gen_agent",
            "--mode",           mode,
            "--episodes",       str(target_ep),
            "--output-dir",     data_dir,
            "--port",           str(base_port + w),
            "--shard-id",       str(w),
            "--total-shards",   str(workers),
            "--resume-episodes", str(completed_ep),
            "--history-window", str(history_window),
        ]
        output_file = worker_paths.get(w)
        if output_file:
            cmd.extend(["--output-file", output_file])
        elif active_ts:
            output_file = os.path.join(
                data_dir,
                f"dataset_{mode}_{base_port + w}_{active_ts}.jsonl",
            )
            cmd.extend(["--output-file", output_file])
        if basic_only:
            cmd.append("--basic-only")
        cmd.extend(["--min-depth", str(min_depth), "--max-depth", str(max_depth)])
        cmds.append(cmd)

    if not cmds:
        print(f"[Orchestrate] Data already exists ({existing} episodes). Skipping generation.")
        return data_dir

    if existing > 0:
        print(
            f"[Orchestrate] Resuming data generation from {existing}/{target} episodes "
            f"across {len(cmds)}/{workers} workers."
        )

    client_cmd = None
    client_cwd = None
    if not remote_clients:
        runner = os.path.abspath(os.path.join(HERE, "..", "icalc", "client_runner.py"))
        client_cmd = [sys.executable, "client_runner.py",
                      "--server-ip", "127.0.0.1",
                      "--workers",   str(workers),
                      "--headless"]
        client_cwd = os.path.dirname(runner)

    _run_workers(
        agent_cmds      = cmds,
        target_episodes = target,
        phase_label     = f"Data Gen ({mode})",
        remote_clients  = remote_clients,
        client_cmd      = client_cmd,
        client_cwd      = client_cwd,
        idle_timeout    = 300,
        timeout_safety  = dg.get("timeout_safety_multiplier", 3.0),
        agent_cwd       = None,
        initial_progress= initial_progress,
        base_completed  = base_completed,
    )
    return data_dir


# ---------------------------------------------------------------------------
# Phase 1: Pre-training
# ---------------------------------------------------------------------------

def phase_pretrain(cfg: dict, exp_dir: str) -> str:
    """Run supervised pre-training.  Returns best checkpoint path."""
    pt = cfg.get("pretrain", {})
    if not pt.get("enabled", True):
        print("[Orchestrate] Pre-training disabled — skipping.")
        ckpt = _find_checkpoint(os.path.join(exp_dir, pt.get("run_name", "pre_train")))
        if ckpt:
            return ckpt
        resume_from = pt.get("resume_from")
        if resume_from:
            resolved = _resolve_from_package_dir(resume_from)
            if os.path.isfile(resolved):
                print(f"[Orchestrate] Using pretrain.resume_from: {resolved}")
                return resolved
            raise RuntimeError(
                f"Pre-training is disabled, but resume_from checkpoint was not found: {resolved}"
            )
        if _needs_seed_model(cfg):
            raise RuntimeError(
                "Pre-training is disabled and no seed checkpoint is available. "
                "Set pretrain.resume_from or enable pre-training."
            )
        return ""

    run_name = pt.get("run_name", "pre_train")
    run_dir  = os.path.join(exp_dir, run_name)
    ckpt     = _find_checkpoint(run_dir)

    if ckpt and not cfg.get("overwrite", False):
        print(f"[Orchestrate] Pre-train checkpoint exists: {ckpt}. Skipping.")
        return ckpt

    print(f"=== Phase 1: Pre-training ({run_name}) ===")

    # Resolve data_dir relative to the package directory.
    data_dir = pt.get("data_dir", "dataset")
    if not isinstance(data_dir, list):
        data_dirs = [data_dir]
    else:
        data_dirs = data_dir
        
    resolved_dirs = []
    for d in data_dirs:
        if not os.path.isabs(d):
            d = os.path.abspath(os.path.join(HERE, d))
        resolved_dirs.append(d)

    cfg_copy = dict(cfg)
    cfg_copy.setdefault("pretrain", {})["data_dir"] = resolved_dirs
    cfg_copy["pretrain"]["run_name"] = run_name

    # Write a temp config the trainer can read (simplest IPC)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg_copy, f)
        tmp_cfg = f.name

    _run_cmd([sys.executable, "-m", "ariadne.trainers._pretrain_main",
              "--config", tmp_cfg, "--run-dir", run_dir])
    os.unlink(tmp_cfg)

    ckpt = _find_checkpoint(run_dir)
    if not ckpt:
        raise RuntimeError(f"No checkpoint found in {run_dir} after pre-training.")
    print(f"[Orchestrate] Pre-train done. Checkpoint: {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# Phase 2: DAgger Loop
# ---------------------------------------------------------------------------

def phase_dagger(cfg: dict, exp_dir: str, start_model: str) -> str:
    dagger_cfg = cfg.get("dagger", {})
    if not dagger_cfg.get("enabled", False):
        print("[Orchestrate] DAgger disabled — skipping.")
        return start_model
    if not start_model:
        raise RuntimeError(
            "DAgger requires a seed model checkpoint, but none was provided. "
            "Enable pre-training or set pretrain.resume_from."
        )

    iterations     = dagger_cfg.get("iterations", 5)
    gen_cfg        = dagger_cfg.get("generation", {})
    train_cfg      = dagger_cfg.get("training",   {})
    remote_clients = gen_cfg.get("remote_clients", False)
    ep_per_iter    = gen_cfg.get("episodes_per_iter", 5000)
    workers        = gen_cfg.get("workers", 4)
    min_depths     = gen_cfg.get("min_depth", 1)
    max_depths     = gen_cfg.get("max_depth", 3)
    base_port      = 9000
    current_model  = start_model

    # Find tokenizer
    tok_path = _discover_tokenizer(exp_dir, start_model)

    # Resume logic: skip completed iters
    start_iter = 0
    for i in range(iterations):
        run_dir = os.path.join(exp_dir, f"dagger_iter_{i+1}")
        if _find_checkpoint(run_dir):
            start_iter = i + 1
            current_model = _find_checkpoint(run_dir)

    for i in range(start_iter, iterations):
        print(f"=== Phase 2: DAgger Iteration {i+1}/{iterations} ===")
        run_dir     = os.path.join(exp_dir, f"dagger_iter_{i+1}")
        data_out    = os.path.join(run_dir, "dagger_data")
        os.makedirs(data_out, exist_ok=True)

        # Generation
        existing = _count_jsonl_lines(data_out)
        if existing < ep_per_iter:
            cur_min_depth = min_depths[min(i, len(min_depths) - 1)] if isinstance(min_depths, list) else min_depths
            cur_max_depth = max_depths[min(i, len(max_depths) - 1)] if isinstance(max_depths, list) else max_depths

            cmds = []
            for w in range(workers):
                ep  = ep_per_iter // workers + (1 if w < ep_per_iter % workers else 0)
                cmd = [
                    sys.executable, "-m", "ariadne.agents.dagger_agent",
                    "--model-path",     current_model,
                    "--tokenizer-path", tok_path,
                    "--port",           str(base_port + w),
                    "--output-dir",     data_out,
                    "--shard-id",       str(w),
                    "--total-shards",   str(workers),
                    "--episodes",       str(ep),
                    "--history-window", str(gen_cfg.get("history_window", -1)),
                    "--max-corrections", str(gen_cfg.get("max_corrections", 10)),
                    "--max-steps-multiplier", str(gen_cfg.get("max_steps_multiplier", 2.0)),
                    "--allow-recovery-mistakes", str(int(bool(gen_cfg.get("allow_recovery_mistakes", True)))),
                    "--recoverable-mistake-rate", str(gen_cfg.get("recoverable_mistake_rate", 0.5)),
                    "--idle-timeout",   str(gen_cfg.get("idle_timeout", 120)),
                    "--min-depth",      str(cur_min_depth),
                    "--max-depth",      str(cur_max_depth),
                ]
                cmds.append(cmd)

            client_cmd = client_cwd = None
            if not remote_clients:
                runner = os.path.abspath(os.path.join(HERE, "..", "icalc", "client_runner.py"))
                client_cmd = [sys.executable, "client_runner.py",
                              "--server-ip", "127.0.0.1",
                              "--workers",    str(workers), "--headless"]
                client_cwd = os.path.dirname(runner)

            _run_workers(
                agent_cmds      = cmds,
                target_episodes = ep_per_iter,
                phase_label     = f"DAgger iter {i+1}",
                remote_clients  = remote_clients,
                client_cmd      = client_cmd,
                client_cwd      = client_cwd,
                idle_timeout    = gen_cfg.get("idle_timeout", 120),
                timeout_safety  = gen_cfg.get("timeout_safety_multiplier", 3.0),
            )
        summary = _summarize_dagger_episodes(data_out)
        if summary:
            print(
                f"[Orchestrate] DAgger iter {i+1} success "
                f"{summary['successes']}/{summary['episodes']} ({summary['success_rate']:.1%}) "
                f"| avg_steps={summary['avg_steps']:.2f} "
                f"| avg_recovery_steps={summary['avg_recovery_steps']:.2f}"
            )

        # Training
        cfg_for_iter = dict(cfg)
        cfg_for_iter.setdefault("dagger", {}).setdefault("training", {})
        cfg_for_iter["dagger"]["training"]["current_iteration"] = i + 1
        cfg_for_iter["dagger"]["training"]["run_name"]          = f"dagger_iter_{i+1}"

        # Always reset scheduler for each DAgger iteration so the LR curve
        # shape is identical across all iters regardless of global config.
        cfg_for_iter["reset_scheduler"] = True

        # Collect all data dirs for DAgger training (expert + all dagger iters)
        pt_data = cfg.get("pretrain", {}).get("data_dir", "dataset")
        if isinstance(pt_data, str):
            pt_data = [pt_data]
        
        all_dirs = []
        # Expert data (resolved paths)
        for d in pt_data:
            if not os.path.isabs(d):
                all_dirs.append(os.path.abspath(os.path.join(HERE, d)))
            else:
                all_dirs.append(d)
        
        # All dagger iterations generated so far
        for j in range(i + 1):
            all_dirs.append(os.path.join(exp_dir, f"dagger_iter_{j+1}", "dagger_data"))
            
        cfg_for_iter["dagger"]["training"]["data_dir"] = all_dirs

        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg_for_iter, f)
            tmp_cfg = f.name

        _run_cmd([sys.executable, "-m", "ariadne.trainers._dagger_main",
                  "--config", tmp_cfg,
                  "--run-dir", run_dir,
                  "--resume-from", current_model])
        os.unlink(tmp_cfg)

        new_ckpt = _find_checkpoint(run_dir)
        if not new_ckpt:
            raise RuntimeError(f"No checkpoint in {run_dir}")
        current_model = new_ckpt
        tok_path = _discover_tokenizer(run_dir, current_model)

    return current_model


# ---------------------------------------------------------------------------
# Phase 3: RL Loop
# ---------------------------------------------------------------------------

def phase_rl(cfg: dict, exp_dir: str, start_model: str) -> str:
    rl_cfg = cfg.get("rl", {})
    if not rl_cfg.get("enabled", False):
        print("[Orchestrate] RL disabled — skipping.")
        return start_model
    if not start_model:
        raise RuntimeError(
            "RL requires a seed model checkpoint, but none was provided. "
            "Enable an earlier phase or set the relevant resume_from checkpoint."
        )

    iterations     = rl_cfg.get("iterations", 100)
    rollout_cfg    = rl_cfg.get("rollout",   {})
    train_cfg      = rl_cfg.get("training",  {})
    remote_clients = rollout_cfg.get("remote_clients", False)
    ep_per_iter    = rollout_cfg.get("episodes_per_iter", 1000)
    workers        = rollout_cfg.get("workers", 4)
    min_depths     = rollout_cfg.get("min_depth", 1)
    max_depths     = rollout_cfg.get("max_depth", 3)
    base_port      = 9000
    current_model  = start_model
    reference_model= rl_cfg.get("resume_from", start_model) or start_model

    tok_path = _discover_tokenizer(exp_dir, start_model)

    # Resume: find last completed RL iter
    start_iter = 0
    for i in range(iterations):
        run_dir = os.path.join(exp_dir, f"rl_iter_{i+1}")
        if _find_checkpoint(run_dir):
            start_iter = i + 1
            current_model = _find_checkpoint(run_dir)

    reset_sched = cfg.get("reset_scheduler", False)

    for i in range(start_iter, iterations):
        print(f"=== Phase 3: RL Iteration {i+1}/{iterations} ===")
        run_dir       = os.path.join(exp_dir, f"rl_iter_{i+1}")
        rollout_dir   = os.path.join(run_dir, "rollout_data")
        os.makedirs(rollout_dir, exist_ok=True)

        # Rollout
        existing = _count_jsonl_lines(rollout_dir)
        if existing < ep_per_iter:
            cur_min_depth = min_depths[min(i, len(min_depths) - 1)] if isinstance(min_depths, list) else min_depths
            cur_max_depth = max_depths[min(i, len(max_depths) - 1)] if isinstance(max_depths, list) else max_depths

            cmds = []
            for w in range(workers):
                ep  = ep_per_iter // workers + (1 if w < ep_per_iter % workers else 0)
                cmd = [
                    sys.executable, "-m", "ariadne.agents.rl_agent",
                    "--model-path",        current_model,
                    "--tokenizer-path",    tok_path,
                    "--port",              str(base_port + w),
                    "--output-dir",        rollout_dir,
                    "--shard-id",          str(w),
                    "--total-shards",      str(workers),
                    "--episodes",          str(ep),
                    "--max-steps-multiplier", str(rollout_cfg.get("max_steps_multiplier", 2.0)),
                    "--idle-timeout",      str(rollout_cfg.get("idle_timeout", 120)),
                    "--step-bonus",        str(rollout_cfg.get("step_bonus", 0.0)),
                    "--decode",            rollout_cfg.get("decode", "greedy"),
                    "--temperature",       str(rollout_cfg.get("temperature", 1.0)),
                    "--min-depth",         str(cur_min_depth),
                    "--max-depth",         str(cur_max_depth),
                ]
                cmds.append(cmd)

            client_cmd = client_cwd = None
            if not remote_clients:
                runner = os.path.abspath(os.path.join(HERE, "..", "icalc", "client_runner.py"))
                client_cmd = [sys.executable, "client_runner.py",
                              "--server-ip", "127.0.0.1",
                              "--workers",    str(workers), "--headless"]
                client_cwd = os.path.dirname(runner)

            _run_workers(
                agent_cmds      = cmds,
                target_episodes = ep_per_iter,
                phase_label     = f"RL rollout {i+1}",
                remote_clients  = remote_clients,
                client_cmd      = client_cmd,
                client_cwd      = client_cwd,
                idle_timeout    = rollout_cfg.get("idle_timeout", 120),
                timeout_safety  = rollout_cfg.get("timeout_safety_multiplier", 3.0),
            )

        # RL Training
        decay_step = (i - start_iter) if reset_sched else i
        cfg_copy   = dict(cfg)

        # Shared TensorBoard log dir: one TB "run" across all RL iterations.
        shared_tb_dir = os.path.join(exp_dir, "rl_progress", "logs")
        os.makedirs(shared_tb_dir, exist_ok=True)

        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg_copy, f)
            tmp_cfg = f.name

        _run_cmd([sys.executable, "-m", "ariadne.trainers._rl_main",
                  "--config",          tmp_cfg,
                  "--run-dir",         run_dir,
                  "--rollout-dir",     rollout_dir,
                  "--resume-from",     current_model,
                  "--reference-from",  reference_model,
                  "--tokenizer-path",  tok_path,
                  "--iteration-index", str(i),
                  "--decay-step",      str(decay_step),
                  "--run-name",        f"rl_iter_{i+1}",
                  "--tb-log-dir",      shared_tb_dir])
        os.unlink(tmp_cfg)

        new_ckpt = _find_checkpoint(run_dir)
        if not new_ckpt:
            raise RuntimeError(f"No checkpoint in {run_dir}")
        current_model = new_ckpt
        tok_path = _discover_tokenizer(run_dir, current_model)

    return current_model


# ---------------------------------------------------------------------------
# Tokenizer discovery
# ---------------------------------------------------------------------------

def _discover_tokenizer(run_dir: str, model_path: str) -> str:
    """Find tokenizer.json nearest to the model path."""
    candidates = []
    for d in [run_dir, os.path.dirname(model_path), os.path.dirname(os.path.dirname(model_path))]:
        p = os.path.join(d, "tokenizer.json")
        if os.path.isfile(p):
            candidates.append(p)
    if candidates:
        return candidates[0]
    # Fallback: base tokenizer
    return os.path.abspath(os.path.join(HERE, "configs", "tokenizer.json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ariadne orchestrator")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    args   = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "exp")
    base_dir = cfg.get("base_save_dir",  "runs")
    if not os.path.isabs(base_dir):
        # Resolve relative to the config file's directory (i.e. ariadne/configs/),
        # going up one level so that "runs" lands at ariadne/runs/.
        cfg_dir  = os.path.dirname(os.path.abspath(args.config))
        base_dir = os.path.join(cfg_dir, "..", base_dir)
    base_dir = os.path.abspath(base_dir)

    exp_dir = os.path.join(base_dir, exp_name)

    if cfg.get("overwrite", False) and os.path.exists(exp_dir):
        import shutil
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(exp_dir, "experiment.yaml"), "w") as f:
        yaml.dump(cfg, f)

    print(f"[Orchestrate] Experiment: {exp_name} → {exp_dir}")

    # Phase 0
    data_gen_dir = phase_datagen(cfg, exp_dir)
    if data_gen_dir:
        # Override pretrain's data dir to use exactly what we just generated
        cfg.setdefault("pretrain", {})["data_dir"] = data_gen_dir

    # Phase 1
    model = phase_pretrain(cfg, exp_dir)

    # Phase 2
    model = phase_dagger(cfg, exp_dir, model)

    # Phase 3
    model = phase_rl(cfg, exp_dir, model)

    print(f"[Orchestrate] All phases complete. Final model: {model}")


if __name__ == "__main__":
    main()
