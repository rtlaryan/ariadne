"""
agents/rl_agent.py — On-policy RL rollout server.

The RL agent runs the trained model fully autonomously (no expert override).
After each episode it computes step-level reward shaping and writes the
complete episode as a single JSON object.

Decode modes: greedy | sample | epsilon_greedy  (+top-k, top-p, temperature)

STATUS lines written to stdout:
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
import torch.nn.functional as F

from ariadne.agents.oracle import FUNCTION_KEYS, KEY_DISPLAY, Oracle
from ariadne.inference.agent import Agent


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

_NORMALIZE = {"÷": "/", "×": "*", "⌫": "", " ": ""}


def _norm(expr: str) -> str:
    for a, b in _NORMALIZE.items():
        expr = expr.replace(a, b)
    return expr


def _readout(state: dict) -> str:
    r = state.get("readout", "")
    return "" if r == "0" else r.replace(" ", "")


def _is_clean(state: dict) -> bool:
    return not state.get("history") and _readout(state) == ""


def _simulate(current: str, key: str) -> str:
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


def _is_valid_prefix(current: str, goal: str) -> bool:
    return _norm(goal).startswith(_norm(current))


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def _compute_rewards(goal: str, steps: list[dict], success: bool, step_bonus: float) -> tuple[float, list[float]]:
    ep_reward    = 1.0 if success else 0.0
    step_rewards = []
    for s in steps:
        action  = s.get("action", "")
        curr    = _readout(s.get("state", {}))
        simulated = _simulate(curr, action)
        if action in ("Enter", "=", "m", "Escape"):
            step_rewards.append(0.0)
        elif _is_valid_prefix(simulated, goal):
            step_rewards.append(step_bonus)
        else:
            step_rewards.append(-step_bonus if step_bonus > 0 else -0.1)
    return ep_reward + sum(step_rewards), step_rewards


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k and top_k > 0 and top_k < filtered.numel():
        kth = torch.topk(filtered, k=top_k).values.min()
        filtered[filtered < kth] = -float("inf")
    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(filtered, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        cutoff = cum > top_p; cutoff[0] = False
        sorted_l[cutoff] = -float("inf")
        out = torch.empty_like(sorted_l)
        out[sorted_i] = sorted_l
        filtered = out
    return filtered


def _act_mask(agent: Agent, avail: list[str], device: str) -> torch.Tensor | None:
    _NRM = {"÷": "/", "×": "*", "⌫": "Backspace", "AC": "Escape", "=": "Enter"}
    if not avail:
        return None
    ids = [agent.tokenizer.token_to_id[_NRM.get(k, k)] for k in avail
           if _NRM.get(k, k) in agent.tokenizer.token_to_id]
    if not ids:
        return None
    mask = torch.full((len(agent.tokenizer),), -1e9, dtype=torch.float32, device=device)
    mask[ids] = 0.0
    return mask


def _choose(logits: torch.Tensor, mode: str, temp: float, top_k: int, top_p: float,
             epsilon: float, valid_ids: list[int] | None) -> tuple[int, float, dict]:
    scaled = logits / max(temp, 1e-6)
    if mode in ("sample", "epsilon_greedy"):
        scaled = _top_k_top_p(scaled, top_k, top_p)
    lp    = F.log_softmax(scaled, dim=-1)
    probs = lp.exp()

    if torch.isnan(probs).any() or probs.sum() <= 0:
        aid = random.choice(valid_ids) if valid_ids else random.randrange(probs.numel())
        return aid, -float("inf"), {"fallback": 1}

    ent     = float(-(probs * lp).sum())
    max_p   = float(probs.max())

    if mode == "greedy":
        aid = int(probs.argmax())
    elif mode == "epsilon_greedy":
        if random.random() < epsilon and valid_ids:
            aid = random.choice(valid_ids)
        else:
            aid = int(probs.argmax())
    else:  # sample
        aid = int(torch.distributions.Categorical(probs).sample())

    return aid, float(lp[aid]), {"entropy": ent, "max_prob": max_p, "chosen_prob": float(probs[aid]), "fallback": 0}


# ---------------------------------------------------------------------------
# RLAgent
# ---------------------------------------------------------------------------

class RLAgent:
    def __init__(
        self,
        model_path:      str,
        tokenizer_path:  str,
        episodes:        int   = 1000,
        output_dir:      str   = "rl_rollouts",
        port:            int   = 9000,
        shard_id:        int   = 0,
        total_shards:    int   = 1,
        history_window:  int   = -1,
        max_steps_mult:  float = 2.0,
        idle_timeout:    int   = 120,
        step_bonus:      float = 0.0,
        decode:          str   = "greedy",
        temperature:     float = 1.0,
        top_k:           int   = 0,
        top_p:           float = 1.0,
        epsilon:         float = 0.02,
        log_decode_stats: bool = True,
        min_depth:       int   = 1,
        max_depth:       int   = 3,
    ) -> None:
        self.episodes        = episodes
        self.output_dir      = output_dir
        self.port            = port
        self.shard_id        = shard_id
        self.total_shards    = total_shards
        self.history_window  = history_window
        self.max_steps_mult  = max_steps_mult
        self.idle_timeout    = idle_timeout
        self.step_bonus      = step_bonus
        self.decode          = decode
        self.temperature     = temperature
        self.top_k           = top_k
        self.top_p           = top_p
        self.epsilon         = epsilon
        self.log_decode_stats = log_decode_stats
        self.min_depth       = min_depth
        self.max_depth       = max_depth

        self.oracle = Oracle()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent  = Agent(model_path, tokenizer_path, device=self.device)
        self.agent.history_window = history_window

        self.current_task: tuple | None = None  # (expr, plan)
        self.episode_id      = str(uuid.uuid4())
        self.episode_steps:  list[dict] = []
        self.completed       = 0
        self.successful      = 0
        self.pending_reset   = False
        self.last_req_time   = time.time()

        self._write_lock = threading.Lock()
        self._write_buf: list[str] = []
        self._FLUSH_SIZE = 10
        self._dataset_file = None

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def handle_step(self, state: dict) -> dict:
        self.last_req_time = time.time()

        if self.pending_reset:
            self.agent.reset_history()
            self.episode_steps  = []
            self.pending_reset  = False
            return {"type": "keypress", "key": "Escape"}

        if not self.current_task:
            if not _is_clean(state):
                return {"type": "keypress", "key": "Escape"}
            idx        = (self.completed * self.total_shards) + self.shard_id
            expr, plan = self.oracle.generate_task_for_index(
                idx, min_depth=self.min_depth, max_depth=self.max_depth
            )
            self.current_task  = (expr, plan)
            self.episode_id    = str(uuid.uuid4())
            self.episode_steps = []

        expr, plan = self.current_task

        # Model inference
        state_copy = state.copy()
        hist = self.agent.action_history
        if self.agent.history_window > 0:
            hist = hist[-self.agent.history_window:]
        state_copy["action_history"] = hist

        try:
            goal_toks  = self.agent.serializer.tokenize_expr(expr)
            state_toks = self.agent.serializer.serialize(state_copy)
            full       = ["[GOAL]"] + goal_toks + ["[STATE]"] + state_toks + ["[ACTION]"]
            ids        = self.agent.tokenizer.encode(full)
            inp        = torch.tensor([ids], dtype=torch.long).to(self.device)

            with torch.no_grad():
                logits     = self.agent.model(inp)
                last_logits = logits[0, -1, :]

            avail   = state_copy.get("availableInteractions", [])
            mask    = _act_mask(self.agent, avail, self.device)
            masked  = last_logits + mask if mask is not None else last_logits

            valid_ids = None
            if mask is not None:
                valid_ids = (mask == 0).nonzero(as_tuple=True)[0].tolist()

            action_id, log_prob, stats = _choose(
                masked, self.decode, self.temperature, self.top_k, self.top_p,
                self.epsilon, valid_ids
            )
            model_key = self.agent.tokenizer.decode([action_id])[0]

            if valid_ids and action_id not in valid_ids and valid_ids:
                model_key = self.agent.tokenizer.decode([random.choice(valid_ids)])[0]
                log_prob  = -float("inf")

        except Exception as exc:
            print(f"[RLAgent] Model error: {exc}")
            model_key = "Enter"
            log_prob  = 0.0
            stats     = {}

        step_rec = {"state": state_copy, "action": model_key, "log_prob": log_prob, "step_index": len(self.episode_steps)}
        if self.log_decode_stats:
            step_rec.update({
                "entropy":      stats.get("entropy"),
                "max_prob":     stats.get("max_prob"),
                "chosen_prob":  stats.get("chosen_prob"),
                "fallback":     int(stats.get("fallback", 0)),
            })
        self.episode_steps.append(step_rec)
        action = {"type": "keypress", "key": model_key}

        # Completion check
        success  = False
        ep_done  = False
        if model_key in ("Enter", "="):
            goal_norm = _norm(expr)
            curr_norm = _norm(_readout(state))
            success   = goal_norm == curr_norm
            ep_done   = True
        if len(self.episode_steps) > len(plan) * self.max_steps_mult:
            ep_done = True

        if ep_done:
            total_r, step_rs = _compute_rewards(expr, self.episode_steps, success, self.step_bonus)
            ep_entry = {
                "episode_id":    self.episode_id,
                "task":          expr,
                "success":       success,
                "reward":        total_r,
                "num_steps":     len(self.episode_steps),
                "steps":         self.episode_steps,
                "step_rewards":  step_rs,
                "decode_mode":   self.decode,
                "temperature":   self.temperature,
            }
            self._write(ep_entry)
            self.pending_reset = True
            self.current_task  = None
            self.completed    += 1
            if success:
                self.successful += 1
            rate   = self.successful / self.completed if self.completed else 0
            status = "✓" if success else "✗"
            print(
                f"Episode {self.completed} [{status}] {expr} "
                f"(steps={len(self.episode_steps)}, reward={total_r:.2f}, success_rate={rate:.1%})",
                flush=True
            )
            print(f"STATUS:EPISODE_COMPLETE:{self.completed}", flush=True)

        self.agent.record_action(model_key)
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
        fname = os.path.join(self.output_dir, f"rl_rollout_{self.port}_{int(time.time())}.jsonl")
        self._dataset_file = open(fname, "w")

        time.sleep(random.uniform(1.0, 5.0))

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
            print(f"[RLAgent:{self.port}] Could not bind. Exiting."); sys.exit(1)

        print(
            f"RL Agent running on port {self.port} | "
            f"decode={self.decode} temp={self.temperature} | "
            f"target {self.episodes} episodes"
        )
        try:
            while True:
                server.handle_request()
                if time.time() - self.last_req_time > self.idle_timeout:
                    print(f"STATUS:EXIT:idle_timeout:{self.completed}", flush=True); break
                if self.episodes > 0 and self.completed >= self.episodes:
                    print(f"STATUS:EXIT:episode_limit:{self.completed}", flush=True); break
        finally:
            with self._write_lock: self._flush()
            server.server_close()
            self._dataset_file.close()


def main():
    p = argparse.ArgumentParser(description="RL rollout agent")
    p.add_argument("--model-path",        type=str,   required=True)
    p.add_argument("--tokenizer-path",    type=str,   required=True)
    p.add_argument("--port",              type=int,   default=9000)
    p.add_argument("--output-dir",        type=str,   default="rl_rollouts")
    p.add_argument("--shard-id",          type=int,   default=0)
    p.add_argument("--total-shards",      type=int,   default=1)
    p.add_argument("--history-window",    type=int,   default=-1)
    p.add_argument("--episodes",          type=int,   default=1000)
    p.add_argument("--max-steps-multiplier", type=float, default=2.0)
    p.add_argument("--idle-timeout",      type=int,   default=120)
    p.add_argument("--step-bonus",        type=float, default=0.0)
    p.add_argument("--decode",            type=str,   default="greedy",
                   choices=["greedy", "sample", "epsilon_greedy"])
    p.add_argument("--temperature",       type=float, default=1.0)
    p.add_argument("--top-k",             type=int,   default=0)
    p.add_argument("--top-p",             type=float, default=1.0)
    p.add_argument("--epsilon",           type=float, default=0.02)
    p.add_argument("--no-decode-stats",   action="store_true")
    p.add_argument("--min-depth",         type=int,   default=1)
    p.add_argument("--max-depth",         type=int,   default=3)
    args = p.parse_args()

    RLAgent(
        model_path       = args.model_path,
        tokenizer_path   = args.tokenizer_path,
        episodes         = args.episodes,
        output_dir       = args.output_dir,
        port             = args.port,
        shard_id         = args.shard_id,
        total_shards     = args.total_shards,
        history_window   = args.history_window,
        max_steps_mult   = args.max_steps_multiplier,
        idle_timeout     = args.idle_timeout,
        step_bonus       = args.step_bonus,
        decode           = args.decode,
        temperature      = args.temperature,
        top_k            = args.top_k,
        top_p            = args.top_p,
        epsilon          = args.epsilon,
        log_decode_stats = not args.no_decode_stats,
        min_depth        = args.min_depth,
        max_depth        = args.max_depth,
    ).run()


if __name__ == "__main__":
    main()
