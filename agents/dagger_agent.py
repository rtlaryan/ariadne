"""
agents/dagger_agent.py — DAgger data-collection server.

The DAgger agent runs the trained model to choose actions but labels each
step with the oracle's expert action.  All steps (on-policy and corrected)
are logged so the resulting dataset has balanced label diversity.

STATUS lines written to stdout (monitored by orchestrate.py):
  STATUS:EPISODE_COMPLETE:<n>
  STATUS:EXIT:episode_limit:<n>
  STATUS:EXIT:idle_timeout:<n>
"""

import argparse
import http.server
import json
import os
import random
import socketserver
import sys
import threading
import time
import uuid

import torch

from ariadne.agents.oracle import FUNCTION_KEYS, KEY_DISPLAY, Oracle
from ariadne.inference.agent import Agent


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

def _normalize_readout(readout: str) -> str:
    return readout.replace(" ", "") if readout and readout != "0" else ""


def _is_clean(state: dict) -> bool:
    return not state.get("history") and _normalize_readout(state.get("readout", "")) == ""


def _simulate_key(current: str, key: str) -> str:
    if key in ("m", "Enter", "="):
        return current
    if key == "Backspace":
        return current[:-1] if current else ""
    if key == "Escape":
        return ""
    text = KEY_DISPLAY.get(key, key)
    if key in FUNCTION_KEYS:
        text += "("
    return current + text


def _precompute_trajectory(plan: list[str]) -> list[dict]:
    """Return [{expected_history, expert_action}, …] for *plan*."""
    traj, hist = [], ""
    for key in plan:
        traj.append({"expected_history": hist, "expert_action": key})
        hist = _simulate_key(hist, key)
    return traj


# ---------------------------------------------------------------------------
# DAggerAgent
# ---------------------------------------------------------------------------

class DAggerAgent:
    def __init__(
        self,
        model_path:        str,
        tokenizer_path:    str,
        episodes:          int   = 1000,
        output_dir:        str   = "dagger_data",
        port:              int   = 9000,
        shard_id:          int   = 0,
        total_shards:      int   = 1,
        history_window:    int   = -1,
        max_corrections:   int   = 10,
        max_steps_mult:    float = 2.0,
        idle_timeout:      int   = 120,
        min_depth:         int   = 1,
        max_depth:         int   = 3,
    ) -> None:
        self.episodes        = episodes
        self.output_dir      = output_dir
        self.port            = port
        self.shard_id        = shard_id
        self.total_shards    = total_shards
        self.history_window  = history_window
        self.max_corrections = max_corrections
        self.max_steps_mult  = max_steps_mult
        self.idle_timeout    = idle_timeout
        self.min_depth       = min_depth
        self.max_depth       = max_depth

        self.oracle = Oracle()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent  = Agent(model_path, tokenizer_path, device=device)
        self.agent.history_window = history_window

        self.current_task:   tuple | None = None  # (expr, plan, trajectory)
        self.episode_id      = str(uuid.uuid4())
        self.episode_actions: list[str] = []
        self.corrections     = 0
        self.trajectory_step = 0
        self.completed       = 0
        self.pending_reset   = False
        self.last_req_time   = time.time()

        self._write_lock = threading.Lock()
        self._write_buf: list[str] = []
        self._FLUSH_SIZE = 50
        self._dataset_file = None

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _new_episode(self, state: dict) -> None:
        idx        = (self.completed * self.total_shards) + self.shard_id
        expr, plan = self.oracle.generate_task_for_index(
            idx, min_depth=self.min_depth, max_depth=self.max_depth
        )
        traj       = _precompute_trajectory(plan)
        self.current_task    = (expr, plan, traj)
        self.episode_id      = str(uuid.uuid4())
        self.episode_actions = []
        self.corrections     = 0
        self.trajectory_step = 0

    def _finish(self) -> None:
        self.current_task  = None
        self.pending_reset = True
        self.completed    += 1
        print(f"STATUS:EPISODE_COMPLETE:{self.completed}", flush=True)

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def handle_step(self, state: dict) -> dict:
        self.last_req_time = time.time()
        action   = {}
        is_reset = False

        if self.pending_reset:
            self.agent.reset_history()
            self.episode_actions = []
            action       = {"type": "keypress", "key": "Escape"}
            is_reset     = True
            self.pending_reset = False
            return action

        if not self.current_task:
            if not _is_clean(state):
                action   = {"type": "keypress", "key": "Escape"}
                is_reset = True
                return action
            self._new_episode(state)

        expr, plan, traj = self.current_task
        current_expr = _normalize_readout(state.get("readout", ""))

        # --- Oracle labeling ---
        matches = [
            i for i, s in enumerate(traj)
            if i >= self.trajectory_step and s["expected_history"] == current_expr
        ]
        if matches:
            idx           = matches[0]
            expert_action = traj[idx]["expert_action"]
            self.trajectory_step = idx + 1
        else:
            # Divergence recovery
            goal_norm = expr.replace(" ", "")
            if goal_norm.startswith(current_expr):
                suffix = goal_norm[len(current_expr):]
                if not suffix:
                    expert_action = "Enter"
                else:
                    expert_action = suffix[0]
                    for f in sorted(self.oracle.functions, key=len, reverse=True):
                        if suffix.startswith(f):
                            expert_action = f
                            break
            else:
                expert_action = "Backspace"

        # --- Model prediction ---
        try:
            model_key = self.agent.predict(expr, state)
        except Exception as e:
            print(f"[DAggerAgent] Model error: {e}")
            model_key = "Enter"

        # --- Decision ---
        deviates = model_key != expert_action
        screen_diverged = not matches

        # Override with expert if needed; track corrections
        final_key = model_key
        if deviates or screen_diverged:
            final_key = expert_action
            self.corrections += 1
            if self.corrections > self.max_corrections:
                print(f"[DAggerAgent] Max corrections exceeded — aborting.")
                self._finish()
                final_key = "Escape"

        # Check max steps
        if len(self.episode_actions) > len(plan) * self.max_steps_mult:
            print(f"[DAggerAgent] Max steps exceeded — aborting.")
            self._finish()
            final_key = "Escape"

        # Check completion
        if expert_action == "Enter" and final_key == "Enter":
            self._finish()

        action = {"type": "keypress", "key": final_key}

        # --- Log ---
        state["action_history"] = self.episode_actions[:]
        divergence = "action" if deviates else ("screen" if screen_diverged else "none")
        entry = {
            "episode_id":       self.episode_id,
            "timestamp":        time.time(),
            "mode":             "dagger",
            "task":             expr,
            "state":            state,
            "action":           {"type": "keypress", "key": expert_action},  # EXPERT label
            "model_prediction": model_key,
            "divergence_reason": divergence,
            "step_index":       len(self.episode_actions),
            "action_history":   self.episode_actions[:],
        }
        self._write(entry)

        # Update histories
        self.agent.record_action(final_key)
        self.episode_actions.append(final_key)

        return action

    def _write(self, entry: dict) -> None:
        with self._write_lock:
            if self._dataset_file:
                self._write_buf.append(json.dumps(entry) + "\n")
                if len(self._write_buf) >= self._FLUSH_SIZE:
                    self._flush()

    def _flush(self) -> None:
        if self._dataset_file and self._write_buf:
            self._dataset_file.writelines(self._write_buf)
            self._dataset_file.flush()
            self._write_buf.clear()

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        fname = os.path.join(self.output_dir, f"dagger_{self.port}_{int(time.time())}.jsonl")
        self._dataset_file = open(fname, "w")

        time.sleep(random.uniform(1.0, 5.0))  # startup jitter

        agent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args): pass
            def do_POST(self):
                if self.path != "/step":
                    self.send_response(404); self.end_headers(); return
                length = int(self.headers["Content-Length"])
                raw    = self.rfile.read(length)
                try:
                    state = json.loads(raw.decode())
                    state.pop("screenshot", None)
                    action = agent.handle_step(state)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(action).encode())
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    self.send_response(500); self.end_headers()

        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads      = True

        server = None
        for attempt in range(5):
            try:
                server = ThreadedServer(("", self.port), Handler)
                server.timeout = 0.5
                break
            except OSError:
                time.sleep(random.uniform(2.0, 5.0))
        if server is None:
            print(f"[DAggerAgent:{self.port}] Could not bind. Exiting.")
            sys.exit(1)

        start_time = time.time()
        try:
            while True:
                server.handle_request()
                idle = time.time() - self.last_req_time
                if idle > self.idle_timeout:
                    print(f"STATUS:EXIT:idle_timeout:{self.completed}", flush=True)
                    break
                if self.episodes > 0 and self.completed >= self.episodes:
                    print(f"STATUS:EXIT:episode_limit:{self.completed}", flush=True)
                    break
        finally:
            with self._write_lock: self._flush()
            server.server_close()
            self._dataset_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",        type=str, required=True)
    p.add_argument("--tokenizer-path",    type=str, required=True)
    p.add_argument("--port",              type=int,   default=9000)
    p.add_argument("--output-dir",        type=str,   default="dagger_data")
    p.add_argument("--shard-id",          type=int,   default=0)
    p.add_argument("--total-shards",      type=int,   default=1)
    p.add_argument("--history-window",    type=int,   default=-1)
    p.add_argument("--episodes",          type=int,   default=1000)
    p.add_argument("--max-corrections",   type=int,   default=10)
    p.add_argument("--max-steps-multiplier", type=float, default=2.0)
    p.add_argument("--idle-timeout",      type=int,   default=120)
    p.add_argument("--min-depth",         type=int,   default=1)
    p.add_argument("--max-depth",         type=int,   default=3)
    args = p.parse_args()

    DAggerAgent(
        model_path      = args.model_path,
        tokenizer_path  = args.tokenizer_path,
        episodes        = args.episodes,
        output_dir      = args.output_dir,
        port            = args.port,
        shard_id        = args.shard_id,
        total_shards    = args.total_shards,
        history_window  = args.history_window,
        max_corrections = args.max_corrections,
        max_steps_mult  = args.max_steps_multiplier,
        idle_timeout    = args.idle_timeout,
        min_depth       = args.min_depth,
        max_depth       = args.max_depth,
    ).run()


if __name__ == "__main__":
    main()
