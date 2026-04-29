"""Live evaluation worker that runs a fixed unseen-expression suite."""

from __future__ import annotations

import argparse
import http.server
import json
import os
import random
import socketserver
import threading
import time
import uuid

import torch
import torch.nn.functional as F

from ariadne.core.action_sim import apply_key_to_readout
from ariadne.core.calculator_spec import canonicalize_key
from ariadne.eval.common import canonicalize_task
from ariadne.eval.scoring import first_divergence_step, score_episode
from ariadne.inference.agent import Agent


_NORMALIZE_KEYS = {}
_CONTROL_ACTIONS = {"Enter", "=", "Escape", "m", "deg"}


def _norm(expr: str) -> str:
    return canonicalize_task(expr)


def _readout(state: dict) -> str:
    value = state.get("readout", "")
    return "" if value == "0" else value.replace(" ", "")


def _latest_history_expr(state: dict) -> str:
    history = state.get("history", [])
    if not history:
        return ""
    return str(history[-1])


def _is_clean(state: dict) -> bool:
    return not state.get("history") and _readout(state) == ""


def _act_mask(agent: Agent, avail: list[str], device: str) -> torch.Tensor | None:
    if not avail:
        return None
    ids = [
        agent.tokenizer.token_to_id[canonicalize_key(k)]
        for k in avail
        if canonicalize_key(k) in agent.tokenizer.token_to_id
    ]
    if not ids:
        return None
    mask = torch.full((len(agent.tokenizer),), -1e9, dtype=torch.float32, device=device)
    mask[ids] = 0.0
    return mask


def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k and 0 < top_k < filtered.numel():
        kth = torch.topk(filtered, k=top_k).values.min()
        filtered[filtered < kth] = -float("inf")
    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(filtered, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        cutoff = cum > top_p
        cutoff[0] = False
        sorted_l[cutoff] = -float("inf")
        out = torch.empty_like(sorted_l)
        out[sorted_i] = sorted_l
        filtered = out
    return filtered


def _choose(
    logits: torch.Tensor,
    mode: str,
    temp: float,
    top_k: int,
    top_p: float,
    epsilon: float,
    valid_ids: list[int] | None,
) -> tuple[int, float, dict]:
    scaled = logits / max(temp, 1e-6)
    if mode in ("sample", "epsilon_greedy"):
        scaled = _top_k_top_p(scaled, top_k, top_p)

    log_probs = F.log_softmax(scaled, dim=-1)
    probs = log_probs.exp()

    if torch.isnan(probs).any() or probs.sum() <= 0:
        choice = random.choice(valid_ids) if valid_ids else random.randrange(probs.numel())
        return choice, -float("inf"), {"fallback": 1}

    if mode == "greedy":
        choice = int(probs.argmax())
    elif mode == "epsilon_greedy":
        if random.random() < epsilon and valid_ids:
            choice = random.choice(valid_ids)
        else:
            choice = int(probs.argmax())
    else:
        choice = int(torch.distributions.Categorical(probs).sample())

    return choice, float(log_probs[choice]), {
        "entropy": float(-(probs * log_probs).sum()),
        "max_prob": float(probs.max()),
        "chosen_prob": float(probs[choice]),
        "fallback": 0,
    }


def _terminal_success(goal: str, state: dict) -> bool:
    return _norm(_latest_history_expr(state)) == _norm(goal)


def _first_divergence_step(goal: str, steps: list[dict]) -> int | None:
    return first_divergence_step(goal, steps)


class EvalWorker:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        suite_path: str,
        episodes: int = 0,
        output_dir: str = "eval_outputs",
        port: int = 9000,
        shard_id: int = 0,
        total_shards: int = 1,
        history_window: int = -1,
        max_steps_mult: float = 2.0,
        idle_timeout: int = 120,
        decode: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        epsilon: float = 0.02,
        log_decode_stats: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.port = port
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.max_steps_mult = max_steps_mult
        self.idle_timeout = idle_timeout
        self.decode = decode
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.epsilon = epsilon
        self.log_decode_stats = log_decode_stats

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = Agent(model_path, tokenizer_path, device=self.device)
        self.agent.history_window = history_window

        self.suite = self._load_suite(suite_path)
        self.target_episodes = episodes or len(self.suite)
        self.current_case: dict | None = None
        self.current_steps: list[dict] = []
        self.current_episode_id = str(uuid.uuid4())
        self.completed = 0
        self.successful = 0
        self.pending_reset = False
        self.awaiting_terminal_state = False
        self.last_req_time = time.time()

        self._write_lock = threading.Lock()
        self._write_buf: list[str] = []
        self._dataset_file = None
        self._FLUSH_SIZE = 10

    def _load_suite(self, suite_path: str) -> list[dict]:
        cases: list[dict] = []
        with open(suite_path, "r") as f:
            for index, raw in enumerate(f):
                raw = raw.strip()
                if not raw:
                    continue
                spec = json.loads(raw)
                if index % self.total_shards == self.shard_id:
                    cases.append(spec)
        return cases

    def _step_limit(self, case: dict) -> int:
        plan_len = int(case.get("oracle_plan_length", 0))
        return max(int(plan_len * self.max_steps_mult), plan_len + 4)

    def _start_next_case(self) -> None:
        self.current_case = self.suite[self.completed]
        self.current_steps = []
        self.current_episode_id = str(uuid.uuid4())
        self.awaiting_terminal_state = False

    def _predict(self, goal: str, state: dict) -> tuple[str, float, dict]:
        state_copy, inp = self.agent._prepare_input(goal, state)
        logits = self.agent._masked_logits(inp, state_copy)
        avail = state_copy.get("availableInteractions", [])
        mask = _act_mask(self.agent, avail, self.device)
        masked = logits + mask if mask is not None else logits
        valid_ids = (mask == 0).nonzero(as_tuple=True)[0].tolist() if mask is not None else None

        action_id, log_prob, stats = _choose(
            masked, self.decode, self.temperature, self.top_k, self.top_p, self.epsilon, valid_ids
        )
        token = self.agent.tokenizer.decode([action_id])[0]
        if valid_ids and action_id not in valid_ids:
            token = self.agent.tokenizer.decode([random.choice(valid_ids)])[0]
            log_prob = -float("inf")
        return token, log_prob, stats

    def _record_step(self, goal: str, state: dict, action: str, log_prob: float, stats: dict) -> None:
        readout_before = _readout(state)
        simulated_after = apply_key_to_readout(readout_before, action)
        step = {
            "step_index": len(self.current_steps),
            "action": action,
            "readout_before": readout_before,
            "history_before": _latest_history_expr(state),
            "simulated_after": simulated_after,
            "valid_prefix_after": _norm(goal).startswith(_norm(simulated_after)),
            "log_prob": log_prob,
        }
        if self.log_decode_stats:
            step.update(
                {
                    "entropy": stats.get("entropy"),
                    "max_prob": stats.get("max_prob"),
                    "chosen_prob": stats.get("chosen_prob"),
                    "fallback": int(stats.get("fallback", 0)),
                }
            )
        self.current_steps.append(step)

    def _finish_episode(self, success: bool, reason: str, terminal_state: dict | None = None) -> None:
        assert self.current_case is not None
        terminal_state = terminal_state or {}
        score = score_episode(
            self.current_case,
            terminal_state=terminal_state,
            steps=self.current_steps,
        )
        success = bool(score.success)
        reason = score.reason if reason in {"completed", "wrong_enter"} else reason
        entry = {
            "episode_id": self.current_episode_id,
            "task": self.current_case["task"],
            "task_canonical": self.current_case["task_canonical"],
            "bucket": self.current_case["bucket"],
            "stratum": self.current_case.get("stratum", ""),
            "index": self.current_case["index"],
            "oracle_plan": self.current_case.get("oracle_plan", []),
            "oracle_plan_length": self.current_case.get("oracle_plan_length", 0),
            "success": success,
            "entry_success": score.entry_success,
            "result_success": score.result_success,
            "expected_value": score.expected_value,
            "observed_value": score.observed_value,
            "expected_error": score.expected_error,
            "observed_error": score.observed_error,
            "termination_reason": reason,
            "num_steps": len(self.current_steps),
            "predicted_actions": [step["action"] for step in self.current_steps],
            "readout_trace": [step["readout_before"] for step in self.current_steps],
            "history_trace": [step["history_before"] for step in self.current_steps],
            "first_divergence_step": score.first_divergence_step,
            "terminal_readout": _readout(terminal_state),
            "terminal_history": _latest_history_expr(terminal_state),
            "decode_mode": self.decode,
            "temperature": self.temperature,
            "steps": self.current_steps,
        }
        self._write(entry)

        self.completed += 1
        if success:
            self.successful += 1
        rate = self.successful / float(self.completed) if self.completed else 0.0
        status = "✓" if success else "✗"
        print(
            f"Eval {self.completed}/{self.target_episodes} [{status}] {self.current_case['bucket']} "
            f"{self.current_case['task']} (steps={len(self.current_steps)}, success_rate={rate:.1%}, reason={reason})",
            flush=True,
        )
        print(f"STATUS:EPISODE_COMPLETE:{self.completed}", flush=True)

        self.current_case = None
        self.current_steps = []
        self.awaiting_terminal_state = False

    def handle_step(self, state: dict) -> dict:
        self.last_req_time = time.time()

        if self.pending_reset:
            self.agent.reset_history()
            self.pending_reset = False
            return {"type": "keypress", "key": "Escape"}

        if self.awaiting_terminal_state:
            success = _terminal_success(self.current_case["task"], state)
            reason = "completed" if success else "wrong_enter"
            self._finish_episode(success, reason, terminal_state=state)
            self.agent.reset_history()
            return {"type": "keypress", "key": "Escape"}

        if self.completed >= self.target_episodes:
            return {"type": "terminate"}

        if self.current_case is None:
            if not _is_clean(state):
                return {"type": "keypress", "key": "Escape"}
            self._start_next_case()

        assert self.current_case is not None
        goal = self.current_case["task"]
        try:
            action, log_prob, stats = self._predict(goal, state)
        except Exception as exc:
            print(f"[EvalWorker] Model error: {exc}", flush=True)
            action, log_prob, stats = "Enter", 0.0, {"fallback": 1}

        self._record_step(goal, state, action, log_prob, stats)
        self.agent.record_action(action)

        if action in ("Enter", "="):
            self.awaiting_terminal_state = True
        elif len(self.current_steps) >= self._step_limit(self.current_case):
            self._finish_episode(False, "max_steps")
            self.pending_reset = True

        return {"type": "keypress", "key": action}

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

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        fname = os.path.join(self.output_dir, f"eval_episodes_{self.port}_{int(time.time())}.ndjson")
        self._dataset_file = open(fname, "w")

        time.sleep(random.uniform(1.0, 3.0))
        worker = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                if self.path != "/step":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                try:
                    state = json.loads(raw.decode())
                    state.pop("screenshot", None)
                    action = worker.handle_step(state)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(action).encode())
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self.send_response(500)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(str(exc).encode())

        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        server = ThreadedServer(("", self.port), Handler)
        server.timeout = 0.5
        print(
            f"Eval worker on port {self.port} | decode={self.decode} temp={self.temperature} | "
            f"target {self.target_episodes} episodes",
            flush=True,
        )
        try:
            while True:
                server.handle_request()
                if time.time() - self.last_req_time > self.idle_timeout:
                    print(f"STATUS:EXIT:idle_timeout:{self.completed}", flush=True)
                    break
                if self.completed >= self.target_episodes:
                    print(f"STATUS:EXIT:episode_limit:{self.completed}", flush=True)
                    break
        finally:
            with self._write_lock:
                self._flush()
            server.server_close()
            self._dataset_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ariadne evaluation worker")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--suite-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--history-window", type=int, default=-1)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--max-steps-multiplier", type=float, default=2.0)
    parser.add_argument("--idle-timeout", type=int, default=120)
    parser.add_argument("--decode", type=str, default="greedy", choices=["greedy", "sample", "epsilon_greedy"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--no-decode-stats", action="store_true")
    args = parser.parse_args()

    EvalWorker(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        suite_path=args.suite_path,
        episodes=args.episodes,
        output_dir=args.output_dir,
        port=args.port,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        history_window=args.history_window,
        max_steps_mult=args.max_steps_multiplier,
        idle_timeout=args.idle_timeout,
        decode=args.decode,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        epsilon=args.epsilon,
        log_decode_stats=not args.no_decode_stats,
    ).run()


if __name__ == "__main__":
    main()
