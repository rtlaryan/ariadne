"""
agents/gen_agent.py — Supervised / crawler data-collection server.

Runs an HTTP server that the icalc bridge connects to.  On each /step
request the agent consults the oracle (supervised mode) or a simple
crawler policy (crawler mode) and returns a keypress action.

All state is encapsulated in GenAgent; no module-level globals.

STATUS lines written to stdout (monitored by orchestrate.py):
  STATUS:EPISODE_COMPLETE:<n>
  STATUS:EXIT:episode_limit:<n>
"""

import argparse
import http.server
import json
import os
import socketserver
import threading
import time
import uuid

from ariadne.agents.oracle import Oracle


# ---------------------------------------------------------------------------
# Crawler helpers
# ---------------------------------------------------------------------------

_CRAWLER_KEYS = list("0123456789") + ["+", "-", "*", "/", "Enter", "Backspace"]


def _crawler_action() -> dict:
    """Random valid calculator action (no science to keep it simple)."""
    return {"type": "keypress", "key": __import__("random").choice(_CRAWLER_KEYS)}


# ---------------------------------------------------------------------------
# GenAgent
# ---------------------------------------------------------------------------

class GenAgent:
    def __init__(
        self,
        mode:           str   = "supervised",
        episodes:       int   = 1000,
        output_dir:     str   = "dataset",
        output_file:    str | None = None,
        port:           int   = 9000,
        shard_id:       int   = 0,
        total_shards:   int   = 1,
        resume_episodes:int   = 0,
        basic_only:     bool  = False,
        history_window: int   = -1,
        min_depth:      int   = 1,
        max_depth:      int   = 3,
    ) -> None:
        self.mode           = mode
        self.episodes       = episodes
        self.output_dir     = output_dir
        self.output_file    = output_file
        self.port           = port
        self.shard_id       = shard_id
        self.total_shards   = total_shards
        self.resume_episodes = resume_episodes
        self.basic_only     = basic_only
        self.history_window = history_window
        self.min_depth      = min_depth
        self.max_depth      = max_depth

        self.oracle              = Oracle()
        self.current_task        = None   # (expr, plan)
        self.current_step_idx    = 0
        self.episode_id          = str(uuid.uuid4())
        self.episode_actions:    list[str] = []
        self.completed_episodes  = resume_episodes
        self.pending_reset       = False

        self._write_lock = threading.Lock()
        self._write_buf: list[str] = []
        self._FLUSH_SIZE = 50
        self._dataset_file = None

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _new_episode(self, state: dict) -> None:
        global_index = (self.completed_episodes * self.total_shards) + self.shard_id
        current_mode = state.get("mode", "basic")
        expr, plan   = self.oracle.generate_task_for_index(
            global_index,
            current_mode=current_mode,
            basic_only=self.basic_only,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        self.current_task     = (expr, plan)
        self.current_step_idx = 0
        self.episode_id       = str(uuid.uuid4())
        self.episode_actions  = []

    def _finish_episode(self) -> None:
        self.current_task  = None
        self.pending_reset = True
        self.completed_episodes += 1
        print(f"STATUS:EPISODE_COMPLETE:{self.completed_episodes}", flush=True)

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def handle_step(self, state: dict) -> dict:
        """Compute the next action and return it as a dict."""
        action   = {}
        is_reset = False

        if self.mode == "supervised":
            if self.pending_reset:
                action         = {"type": "keypress", "key": "Escape"}
                is_reset       = True
                self.pending_reset = False

            elif not self.current_task:
                self._new_episode(state)

            if self.current_task and not is_reset:
                expr, plan = self.current_task
                if self.current_step_idx < len(plan):
                    key = plan[self.current_step_idx]
                    action = {"type": "keypress", "key": key}
                    self.current_step_idx += 1
                else:
                    self._finish_episode()
                    action = {}   # no-op; reset comes next step

        elif self.mode == "crawler":
            action = _crawler_action()
            is_reset = action.get("key") == "Escape"
            if is_reset:
                self._finish_episode()

        # Log the action
        if action and action.get("type") and not is_reset:
            past = self._history_slice()
            state["action_history"] = past

            entry = {
                "episode_id": self.episode_id,
                "timestamp":  time.time(),
                "mode":       self.mode,
                "task":       self.current_task[0] if self.current_task else None,
                "step_index": self.current_step_idx,
                "state":      state,
                "action":     action,
            }
            key = action.get("key", "")
            self.episode_actions.append(key)
            self._write(entry)

        if is_reset:
            self.episode_actions = []

        return action

    def _history_slice(self) -> list[str]:
        if self.history_window == -1:
            return self.episode_actions[:]
        if self.history_window == 0:
            return []
        return self.episode_actions[-self.history_window :]

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
        fname = self.output_file or os.path.join(
            self.output_dir,
            f"dataset_{self.mode}_{self.port}_{int(time.time())}.jsonl",
        )
        mode = "a" if self.output_file and os.path.exists(fname) else "w"
        self._dataset_file = open(fname, mode)

        agent = self   # closure

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

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
                    print(f"[GenAgent] Error: {exc}")
                    self.send_response(500); self.end_headers()

        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer(("", self.port), Handler)
        server.timeout = 0.5

        try:
            while True:
                server.handle_request()
                if self.episodes > 0 and self.completed_episodes >= self.episodes:
                    print(f"STATUS:EXIT:episode_limit:{self.completed_episodes}", flush=True)
                    break
        except KeyboardInterrupt:
            pass
        finally:
            with self._write_lock:
                self._flush()
            server.server_close()
            self._dataset_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           choices=["supervised", "crawler"], default="supervised")
    p.add_argument("--episodes",       type=int,   default=1000)
    p.add_argument("--output-dir",     type=str,   default="dataset")
    p.add_argument("--output-file",    type=str,   default=None)
    p.add_argument("--port",           type=int,   default=9000)
    p.add_argument("--shard-id",       type=int,   default=0)
    p.add_argument("--total-shards",   type=int,   default=1)
    p.add_argument("--resume-episodes", type=int,  default=0)
    p.add_argument("--basic-only",     action="store_true")
    p.add_argument("--history-window", type=int,   default=-1)
    p.add_argument("--min-depth",      type=int,   default=1)
    p.add_argument("--max-depth",      type=int,   default=3)
    args = p.parse_args()

    GenAgent(
        mode           = args.mode,
        episodes       = args.episodes,
        output_dir     = args.output_dir,
        output_file    = args.output_file,
        port           = args.port,
        shard_id       = args.shard_id,
        total_shards   = args.total_shards,
        resume_episodes= args.resume_episodes,
        basic_only     = args.basic_only,
        history_window = args.history_window,
        min_depth      = args.min_depth,
        max_depth      = args.max_depth,
    ).run()


if __name__ == "__main__":
    main()
