"""
inference/agent.py — Inference-time agent wrapper.

Wraps a trained AgentTransformer checkpoint and provides:
  predict()                     – greedy action prediction
  predict_with_verification()   – confidence-gated beam search
  record_action() / reset_history()

Key fix vs. v1: the valid-action mask is applied as an additive offset
to a *copy* of the logits (not in-place on the model output tensor).
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ariadne.core.action_sim import apply_key_to_readout
from ariadne.core.dataset import StateSerializer
from ariadne.core.model import AgentTransformer, build_model, load_checkpoint
from ariadne.core.tokenizer import TokenMap


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Stateful inference agent.

    Parameters
    ----------
    model_path      : path to a .pt checkpoint (output of Trainer)
    tokenizer_path  : path to tokenizer.json
    config_path     : path to config.json (auto-discovered if None)
    device          : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_path:     str,
        tokenizer_path: str,
        config_path:    Optional[str] = None,
        device:         str = "cpu",
    ) -> None:
        self.device     = device
        self.tokenizer  = TokenMap.load(tokenizer_path)
        self.serializer = StateSerializer(self.tokenizer)

        # Load config
        cfg = self._load_config(config_path, model_path)
        cfg["vocab_size"] = len(self.tokenizer)

        # Build + load model
        self.model = build_model(cfg).to(device)
        ckpt = load_checkpoint(self.model, model_path, device=device)
        self.model.eval()
        self.value_head = nn.Linear(cfg.get("embed_dim", 256), 1).to(device)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        value_state = ckpt.get("value_head_state_dict")
        if value_state:
            try:
                self.value_head.load_state_dict(value_state, strict=True)
            except Exception:
                pass
        self.value_head.eval()

        self.max_len       = cfg.get("max_len", 256)
        self._hist: list[str] = []
        self.history_window   = -1  # -1 = full history

        print(f"[Agent] {self.model.__class__.__name__} — {sum(p.numel() for p in self.model.parameters()):,} params")

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        self._hist = []

    def record_action(self, key: str) -> None:
        self._hist.append(key)

    @property
    def action_history(self) -> list[str]:
        return self._hist

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: Optional[str], model_path: str) -> dict:
        defaults = {"embed_dim": 256, "num_layers": 6, "num_heads": 8, "max_len": 256}
        if config_path and os.path.isfile(config_path):
            with open(config_path) as f:
                return {**defaults, **json.load(f)}
        # Auto-discover config.json by walking upward from checkpoint
        d = os.path.dirname(os.path.abspath(model_path))
        for _ in range(4):
            c = os.path.join(d, "config.json")
            if os.path.isfile(c):
                with open(c) as f:
                    return {**defaults, **json.load(f)}
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return defaults

    def _prepare_input(self, goal: str, state: dict) -> tuple[dict, torch.Tensor]:
        state = state.copy()
        hist  = self._hist[-self.history_window:] if self.history_window > 0 else self._hist
        state["action_history"] = hist

        goal_toks  = self.serializer.tokenize_expr(goal)
        state_toks = self.serializer.serialize(state)
        tokens     = ["[GOAL]"] + goal_toks + ["[STATE]"] + state_toks + ["[ACTION]"]
        ids        = self.tokenizer.encode(tokens)

        # Truncate from the left to fit max_len
        if len(ids) > self.max_len:
            ids = ids[-self.max_len:]

        inp = torch.tensor([ids], dtype=torch.long).to(self.device)
        return state, inp

    def _masked_logits(self, input_tensor: torch.Tensor, state: dict) -> torch.Tensor:
        """Run model and return logits at last position with valid-action mask applied."""
        with torch.no_grad():
            raw_logits = self.model(input_tensor)   # [1, T, V]
        last = raw_logits[0, -1, :].clone()         # copy — no in-place mutation

        avail = state.get("availableInteractions", [])
        if avail:
            from ariadne.core.dataset import _KEY_NORMALIZE
            ids = [
                self.tokenizer.token_to_id[_KEY_NORMALIZE.get(k, k)]
                for k in avail
                if _KEY_NORMALIZE.get(k, k) in self.tokenizer.token_to_id
            ]
            if ids:
                mask      = torch.full((len(self.tokenizer),), float("-inf"), device=self.device)
                mask[ids] = 0.0
                last      = last + mask   # additive, not in-place

        return last

    def policy_logits_and_value(self, goal: str, state: dict) -> tuple[dict, torch.Tensor, float]:
        """Return masked policy logits and critic value for a single state."""
        state_copy, inp = self._prepare_input(goal, state)
        with torch.no_grad():
            raw_logits, hidden = self.model(inp, return_hidden_states=True)
            value = float(self.value_head(hidden[:, -1, :]).squeeze(-1).item())
        last = raw_logits[0, -1, :].clone()

        avail = state_copy.get("availableInteractions", [])
        if avail:
            from ariadne.core.dataset import _KEY_NORMALIZE
            ids = [
                self.tokenizer.token_to_id[_KEY_NORMALIZE.get(k, k)]
                for k in avail
                if _KEY_NORMALIZE.get(k, k) in self.tokenizer.token_to_id
            ]
            if ids:
                mask = torch.full((len(self.tokenizer),), float("-inf"), device=self.device)
                mask[ids] = 0.0
                last = last + mask

        return state_copy, last, value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, goal: str, state: dict) -> str:
        """Greedy action prediction."""
        state, inp = self._prepare_input(goal, state)
        logits     = self._masked_logits(inp, state)
        return self.tokenizer.decode([int(logits.argmax())])[0]

    def predict_with_verification(
        self,
        goal: str,
        state: dict,
        confidence_threshold: float = 0.85,
        top_k: int = 3,
    ) -> tuple[str, float]:
        """Confidence-gated beam search (Variant C).

        Fast path when top probability ≥ threshold; otherwise verifies
        candidates via simulated-expression prefix matching.
        """
        state, inp = self._prepare_input(goal, state)
        logits     = self._masked_logits(inp, state)

        probs               = F.softmax(logits, dim=-1)
        top_probs, top_ids  = probs.topk(top_k)

        greedy_prob  = top_probs[0].item()
        greedy_token = self.tokenizer.decode([int(top_ids[0])])[0]

        if greedy_prob >= confidence_threshold:
            return greedy_token, greedy_prob

        current_expr = self._get_readout(state)
        for i in range(top_k):
            tok  = self.tokenizer.decode([int(top_ids[i])])[0]
            prob = top_probs[i].item()
            if tok in ("Enter", "=", "m", "Escape", "Backspace"):
                continue
            simulated = self._simulate(current_expr, tok)
            if self._is_prefix(simulated, goal):
                return tok, prob

        return greedy_token, greedy_prob

    # ------------------------------------------------------------------
    # Expression simulation
    # ------------------------------------------------------------------

    _NORMALIZE   = {"×": "*", "÷": "/", "⌫": "", " ": ""}

    @classmethod
    def _simulate(cls, current: str, action: str) -> str:
        return apply_key_to_readout(current, action)

    @classmethod
    def _norm(cls, expr: str) -> str:
        for a, b in cls._NORMALIZE.items():
            expr = expr.replace(a, b)
        return expr

    @classmethod
    def _is_prefix(cls, current: str, goal: str) -> bool:
        return cls._norm(goal).startswith(cls._norm(current))

    @staticmethod
    def _get_readout(state: dict) -> str:
        r = state.get("readout", "")
        return "" if r == "0" else r.replace(" ", "")
