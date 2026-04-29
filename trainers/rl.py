"""
trainers/rl.py — PPO actor-critic fine-tuning for Ariadne.

This trainer consumes rollout JSONL files where each line stores one full
episode with episode-level goal conditioning and step-level rewards/log-probs.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ariadne.core.dataset import StateSerializer
from ariadne.core.model import build_model, load_checkpoint
from ariadne.core.tokenizer import TokenMap


def _load_rollouts(rollout_dir: str) -> list[dict]:
    """Return episode dicts from RL rollout JSONL files."""
    episodes = []
    for path in glob.glob(os.path.join(rollout_dir, "**/*.jsonl"), recursive=True):
        with open(path, "r") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    episode = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                episodes.append(episode)
    return episodes


_NORMALIZE_KEYS = {
    "÷": "/",
    "×": "*",
    "⌫": "Backspace",
    "AC": "Escape",
    "=": "Enter",
}


def _valid_action_mask(
    tokenizer: TokenMap,
    available: list[str],
    device: str,
    vocab_size: int,
) -> Optional[torch.Tensor]:
    if not available:
        return None
    mask = torch.full((vocab_size,), float("-inf"), dtype=torch.float32, device=device)
    valid = 0
    for key in available:
        idx = tokenizer.token_to_id.get(_NORMALIZE_KEYS.get(key, key))
        if idx is not None:
            mask[idx] = 0.0
            valid += 1
    return mask if valid else None


def _normalize(values: list[float]) -> list[float]:
    arr = torch.tensor(values, dtype=torch.float32)
    if arr.numel() <= 1:
        return values
    std = arr.std(unbiased=False)
    if std.item() < 1e-8:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / (std + 1e-8)).tolist()


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / float(len(values)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    mid = len(arr) // 2
    if len(arr) % 2 == 0:
        return 0.5 * (arr[mid - 1] + arr[mid])
    return arr[mid]


def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k and top_k > 0 and top_k < filtered.numel():
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


@dataclass
class Transition:
    input_ids: torch.Tensor
    action_id: int
    available: list[str]
    old_log_prob: float
    old_value: float
    advantage: float
    return_target: float
    reward: float
    done: bool


class RLTrainer:
    """PPO trainer with an actor transformer and scalar value head."""

    def __init__(
        self,
        cfg: dict,
        run_dir: str,
        rollout_dir: str,
        resume_from: str,
        reference_from: str,
        tokenizer_path: str,
        iteration_index: int = 0,
        decay_step: int = 0,
        run_name: str = "rl",
        tb_log_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.rollout_dir = rollout_dir
        self.run_name = run_name
        self.iteration_index = iteration_index
        self.decay_step = decay_step

        train_cfg = cfg.get("rl", {}).get("training", {})
        self.lr = float(train_cfg.get("lr", 1e-5))
        self.lr_decay = float(train_cfg.get("lr_decay", 1.0))
        self.lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
        self.lr_min_ratio = float(train_cfg.get("lr_min_ratio", 0.05))
        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
        self.ppo_epochs = int(train_cfg.get("ppo_epochs", 4))
        self.minibatch_size = int(train_cfg.get("minibatch_size", 64))
        self.clip_ratio = float(train_cfg.get("clip_ratio", 0.2))
        self.value_loss_coef = float(train_cfg.get("value_loss_coef", 0.5))
        self.entropy_coef = float(train_cfg.get("entropy_coef", 0.001))
        self.target_kl = float(train_cfg.get("target_kl", 0.02))
        self.reference_kl_coef = float(train_cfg.get("reference_kl_coef", 0.0))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
        self.min_success_rate = float(train_cfg.get("min_success_rate", 0.0))
        self.effective_lr = self.lr * (self.lr_decay ** decay_step)
        rollout_cfg = cfg.get("rl", {}).get("rollout", {})
        self.rollout_decode = str(rollout_cfg.get("decode", "sample"))
        self.rollout_temperature = float(rollout_cfg.get("temperature", 1.0))
        self.rollout_top_k = int(rollout_cfg.get("top_k", 0))
        self.rollout_top_p = float(rollout_cfg.get("top_p", 0.95))
        self.rollout_epsilon = float(rollout_cfg.get("epsilon", 0.02))

        self.max_len = int(cfg.get("max_len", 256))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = TokenMap.load(tokenizer_path)
        self.serializer = StateSerializer(self.tokenizer)

        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_dir = tb_log_dir if tb_log_dir else os.path.join(run_dir, "logs")

        self._setup_model(resume_from)
        self._setup_reference(reference_from)
        self._save_model_config()

        print(
            f"[RLTrainer] Tokenizer loaded from {tokenizer_path} — vocab: {len(self.tokenizer)}"
        )
        print(
            f"[RLTrainer] PPO config: minibatch={self.minibatch_size}, epochs={self.ppo_epochs}, "
            f"clip={self.clip_ratio:.3f}, lr={self.effective_lr:.2e}"
        )
        print(
            f"[RLTrainer] Rollout policy for PPO ratios: decode={self.rollout_decode} "
            f"temp={self.rollout_temperature} top_k={self.rollout_top_k} "
            f"top_p={self.rollout_top_p} epsilon={self.rollout_epsilon}"
        )

    def _arc_cfg(self) -> dict:
        return {
            "vocab_size": len(self.tokenizer),
            "embed_dim": self.cfg.get("embed_dim", 256),
            "num_layers": self.cfg.get("num_layers", 6),
            "num_heads": self.cfg.get("num_heads", 8),
            "max_len": self.max_len,
            "dropout": self.cfg.get("dropout", 0.0),
        }

    def _setup_model(self, resume_from: str) -> None:
        arc_cfg = self._arc_cfg()
        self.model = build_model(arc_cfg).to(self.device)
        ckpt = load_checkpoint(self.model, resume_from, device=self.device)
        self.value_head = nn.Linear(arc_cfg["embed_dim"], 1).to(self.device)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        value_state = ckpt.get("value_head_state_dict")
        if value_state:
            try:
                self.value_head.load_state_dict(value_state, strict=True)
            except Exception:
                pass

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=self.effective_lr,
        )
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.effective_lr
            except Exception:
                pass

        print(f"[RLTrainer] Policy model loaded from {resume_from}")

    def _setup_reference(self, reference_from: str) -> None:
        self.ref_model = build_model(self._arc_cfg()).to(self.device)
        load_checkpoint(self.ref_model, reference_from, device=self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        print(f"[RLTrainer] Reference model loaded from {reference_from}")

    def _save_model_config(self) -> None:
        cfg_path = os.path.join(self.run_dir, "config.json")
        if os.path.isfile(cfg_path):
            return
        with open(cfg_path, "w") as f:
            json.dump(self._arc_cfg(), f, indent=2)

    def _build_scheduler(self, total_steps: int):
        warmup = self.lr_warmup_steps
        eta = self.lr_min_ratio

        def _lr_lambda(step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, total_steps - warmup)
            t = max(0.0, min(1.0, t))
            return eta + (1.0 - eta) * 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)

    def _encode_transition(self, goal: str, step: dict) -> Optional[tuple[torch.Tensor, int, list[str]]]:
        state = step.get("state", {})
        action_key = str(step.get("action", ""))
        action_id = self.tokenizer.token_to_id.get(action_key)
        if action_id is None:
            return None

        goal_tokens = self.serializer.tokenize_expr(goal)
        state_tokens = self.serializer.serialize(state)
        tokens = ["[GOAL]"] + goal_tokens + ["[STATE]"] + state_tokens + ["[ACTION]"]
        ids = self.tokenizer.encode(tokens)
        if len(ids) > self.max_len:
            ids = ids[-self.max_len :]

        available = list(state.get("availableInteractions", []))
        return torch.tensor(ids, dtype=torch.long), action_id, available

    def _policy_log_probs(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = logits / max(self.rollout_temperature, 1e-6)
        if self.rollout_decode in ("sample", "epsilon_greedy"):
            scaled = _top_k_top_p(scaled, self.rollout_top_k, self.rollout_top_p)
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = torch.exp(log_probs)
        return log_probs, probs

    @staticmethod
    def _finite_or_zero(values: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values))

    @classmethod
    def _safe_entropy(cls, log_probs: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        safe_probs = cls._finite_or_zero(probs)
        safe_log_probs = cls._finite_or_zero(log_probs)
        return -(safe_probs * safe_log_probs).sum()

    @classmethod
    def _safe_forward_kl(
        cls,
        current_log_probs: torch.Tensor,
        current_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        safe_probs = cls._finite_or_zero(current_probs)
        safe_current = cls._finite_or_zero(current_log_probs)
        safe_reference = cls._finite_or_zero(reference_log_probs)
        return (safe_probs * (safe_current - safe_reference)).sum()

    def _nonfinite_grad_names(self, limit: int = 5) -> list[str]:
        names = []
        for prefix, module in (("policy", self.model), ("value_head", self.value_head)):
            for name, param in module.named_parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    names.append(f"{prefix}.{name}")
                    if len(names) >= limit:
                        return names
        return names

    def _gae(self, rewards: list[float], values: list[float], dones: list[bool]) -> tuple[list[float], list[float]]:
        advantages = [0.0] * len(rewards)
        last_gae = 0.0
        next_value = 0.0

        for idx in range(len(rewards) - 1, -1, -1):
            if idx < len(rewards) - 1:
                next_value = values[idx + 1]
            nonterminal = 0.0 if dones[idx] else 1.0
            delta = rewards[idx] + self.gamma * next_value * nonterminal - values[idx]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[idx] = last_gae

        returns = [adv + value for adv, value in zip(advantages, values)]
        return advantages, returns

    def _prepare_transitions(self, episodes: list[dict]) -> tuple[list[Transition], list[float], int]:
        transitions: list[Transition] = []
        episode_returns: list[float] = []
        successes = 0
        skipped = 0

        raw_advantages: list[float] = []
        staged: list[tuple[torch.Tensor, int, list[str], float, float, float, float, bool]] = []

        for episode in episodes:
            goal = episode.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                skipped += 1
                continue

            steps = episode.get("steps", [])
            if not steps:
                skipped += 1
                continue

            rewards = [float(step.get("reward", 0.0)) for step in steps]
            values = [float(step.get("value_pred", 0.0)) for step in steps]
            dones = [
                bool(step.get("done", idx == len(steps) - 1))
                for idx, step in enumerate(steps)
            ]
            advantages, returns = self._gae(rewards, values, dones)
            episode_returns.append(float(sum(rewards)))
            successes += int(bool(episode.get("success", False)))

            for step, advantage, ret in zip(steps, advantages, returns):
                encoded = self._encode_transition(goal, step)
                if encoded is None:
                    continue
                ids, action_id, available = encoded
                old_log_prob = float(step.get("old_log_prob", 0.0))
                old_value = float(step.get("value_pred", 0.0))
                reward = float(step.get("reward", 0.0))
                done = bool(step.get("done", False))
                if not all(math.isfinite(v) for v in (old_log_prob, old_value, reward, ret)):
                    continue
                raw_advantages.append(float(advantage))
                staged.append((ids, action_id, available, old_log_prob, old_value, float(ret), reward, done))

        if skipped:
            print(f"[RLTrainer] Skipped {skipped} invalid rollout episodes.")

        norm_advantages = _normalize(raw_advantages)
        for idx, (ids, action_id, available, old_log_prob, old_value, ret, reward, done) in enumerate(staged):
            transitions.append(
                Transition(
                    input_ids=ids,
                    action_id=action_id,
                    available=available,
                    old_log_prob=old_log_prob,
                    old_value=old_value,
                    advantage=norm_advantages[idx] if idx < len(norm_advantages) else 0.0,
                    return_target=ret,
                    reward=reward,
                    done=done,
                )
            )

        return transitions, episode_returns, successes

    def _evaluate_transition(
        self,
        transition: Transition,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = transition.input_ids.unsqueeze(0).to(self.device)
        logits, hidden = self.model(ids, return_hidden_states=True)
        last_logits = logits[0, -1, :]
        mask = _valid_action_mask(self.tokenizer, transition.available, self.device, len(self.tokenizer))
        if mask is not None:
            last_logits = last_logits + mask
        log_probs, probs = self._policy_log_probs(last_logits)
        value = self.value_head(hidden[:, -1, :]).squeeze(-1)[0]
        action_log_prob = log_probs[transition.action_id]
        entropy = self._safe_entropy(log_probs, probs)
        return action_log_prob, value, entropy, probs, log_probs

    def _reference_kl(self, transition: Transition, current_log_probs: torch.Tensor, current_probs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ids = transition.input_ids.unsqueeze(0).to(self.device)
            ref_logits = self.ref_model(ids)[0, -1, :]
            mask = _valid_action_mask(self.tokenizer, transition.available, self.device, len(self.tokenizer))
            if mask is not None:
                ref_logits = ref_logits + mask
            ref_log_probs, _ = self._policy_log_probs(ref_logits)
        return self._safe_forward_kl(current_log_probs, current_probs, ref_log_probs)

    def _checkpoint_state(self, scheduler, mean_episode_return: float) -> dict:
        return {
            "model_state_dict": self.model.state_dict(),
            "value_head_state_dict": self.value_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "iteration": self.iteration_index,
            "ppo_metadata": {
                "mean_episode_return": mean_episode_return,
                "clip_ratio": self.clip_ratio,
                "gae_lambda": self.gae_lambda,
                "ppo_epochs": self.ppo_epochs,
                "minibatch_size": self.minibatch_size,
            },
        }

    def _save_checkpoint(self, name: str, scheduler, mean_episode_return: float) -> None:
        path = os.path.join(self.ckpt_dir, f"{self.run_name}_{name}.pt")
        torch.save(self._checkpoint_state(scheduler, mean_episode_return), path)
        self.tokenizer.save(os.path.join(self.run_dir, "tokenizer.json"))

    @staticmethod
    def _explained_variance(predictions: list[float], targets: list[float]) -> float:
        if len(predictions) <= 1 or len(predictions) != len(targets):
            return 0.0
        pred = torch.tensor(predictions, dtype=torch.float32)
        tgt = torch.tensor(targets, dtype=torch.float32)
        var_y = torch.var(tgt, unbiased=False)
        if var_y.item() < 1e-8:
            return 0.0
        return float(1.0 - torch.var(tgt - pred, unbiased=False) / (var_y + 1e-8))

    def train(self) -> None:
        episodes = _load_rollouts(self.rollout_dir)
        if not episodes:
            print("[RLTrainer] No valid rollout episodes found. Skipping.")
            self._save_checkpoint("latest", None, 0.0)
            self._save_checkpoint("best", None, 0.0)
            return

        n_total = len(episodes)
        n_success = sum(1 for episode in episodes if episode.get("success", False))
        success_rate = n_success / n_total if n_total else 0.0
        print(f"[RLTrainer] {n_success}/{n_total} successful episodes ({success_rate:.1%})")
        if success_rate < self.min_success_rate:
            print(
                f"[RLTrainer] Success rate {success_rate:.1%} below minimum "
                f"{self.min_success_rate:.1%}. Skipping."
            )
            self._save_checkpoint("latest", None, 0.0)
            self._save_checkpoint("best", None, 0.0)
            return

        transitions, episode_returns, prepared_successes = self._prepare_transitions(episodes)
        if not transitions:
            print("[RLTrainer] No usable transitions found. Skipping.")
            mean_episode_return = _mean(episode_returns) if episode_returns else 0.0
            self._save_checkpoint("latest", None, mean_episode_return)
            self._save_checkpoint("best", None, mean_episode_return)
            return

        episode_lengths = [int(ep.get("num_steps", len(ep.get("steps", [])))) for ep in episodes if ep.get("steps")]
        transition_rewards = [transition.reward for transition in transitions]
        advantages = [transition.advantage for transition in transitions]
        return_targets = [transition.return_target for transition in transitions]
        old_values = [transition.old_value for transition in transitions]
        mean_episode_return = sum(episode_returns) / max(1, len(episode_returns))
        writer = SummaryWriter(self.log_dir)
        batches_per_epoch = max(1, math.ceil(len(transitions) / self.minibatch_size))
        total_updates = self.ppo_epochs * batches_per_epoch
        # Keep each iteration in its own step lane inside the shared TB run.
        step_stride = max(100_000, total_updates + 1)
        global_step = self.iteration_index * step_stride
        epoch_step_offset = self.iteration_index * max(1, self.ppo_epochs)
        iteration_step = self.iteration_index + 1
        scheduler = self._build_scheduler(total_updates)

        run_success_rate = prepared_successes / max(1, len(episode_returns))
        rollout_text = (
            f"episodes={len(episode_returns)} | transitions={len(transitions)} | "
            f"success={run_success_rate:.1%} | return_mean={mean_episode_return:.3f} | "
            f"return_median={_median(episode_returns):.3f} | steps_mean={_mean(episode_lengths):.1f} | "
            f"adv_std={_std(advantages):.3f}"
        )
        print(f"[RLTrainer] Rollout batch summary: {rollout_text}")

        writer.add_text("RL/Rollout/run_summary", rollout_text, iteration_step)
        writer.add_scalar("RL/Rollout/mean_episode_return", mean_episode_return, iteration_step)
        writer.add_scalar("RL/Rollout/median_episode_return", _median(episode_returns), iteration_step)
        writer.add_scalar("RL/Rollout/min_episode_return", min(episode_returns), iteration_step)
        writer.add_scalar("RL/Rollout/max_episode_return", max(episode_returns), iteration_step)
        writer.add_scalar("RL/Rollout/success_rate", run_success_rate, iteration_step)
        writer.add_scalar("RL/Rollout/transition_count", len(transitions), iteration_step)
        writer.add_scalar("RL/Rollout/episode_count", len(episode_returns), iteration_step)
        writer.add_scalar("RL/Rollout/avg_episode_length", _mean(episode_lengths), iteration_step)
        writer.add_scalar("RL/Rollout/avg_step_reward", _mean(transition_rewards), iteration_step)
        writer.add_scalar("RL/Rollout/advantage_std", _std(advantages), iteration_step)
        writer.add_scalar("RL/Rollout/return_target_mean", _mean(return_targets), iteration_step)
        writer.add_scalar(
            "RL/Rollout/value_explained_variance_before",
            self._explained_variance(old_values, return_targets),
            iteration_step,
        )
        if transition_rewards:
            writer.add_histogram(
                "RL/Distributions/step_rewards",
                torch.tensor(transition_rewards, dtype=torch.float32),
                iteration_step,
            )
        if advantages:
            writer.add_histogram(
                "RL/Distributions/advantages",
                torch.tensor(advantages, dtype=torch.float32),
                iteration_step,
            )
        if episode_returns:
            writer.add_histogram(
                "RL/Distributions/episode_returns",
                torch.tensor(episode_returns, dtype=torch.float32),
                iteration_step,
            )

        early_stop = False
        performed_update = False
        for epoch in range(self.ppo_epochs):
            random.shuffle(transitions)
            batches = [
                transitions[start : start + self.minibatch_size]
                for start in range(0, len(transitions), self.minibatch_size)
            ]
            pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{self.ppo_epochs} [{self.run_name}]")
            epoch_policy = []
            epoch_value = []
            epoch_entropy = []
            epoch_kl = []
            epoch_clip = []
            epoch_ref_kl = []
            epoch_value_pred = []
            epoch_targets = []
            epoch_ratio = []
            epoch_grad_norm = []
            skipped_empty = 0
            skipped_nonfinite_terms = 0
            skipped_nonfinite_loss = 0
            skipped_nonfinite_grad = 0

            for batch in pbar:
                # Rollouts are collected with the actor in eval mode.
                # Keep PPO re-evaluation deterministic as well so old/new
                # log-probs are comparable and dropout does not create fake KL.
                self.model.eval()
                self.value_head.eval()
                self.optimizer.zero_grad()

                policy_losses = []
                value_losses = []
                entropies = []
                approx_kls = []
                clip_fracs = []
                ref_kls = []
                value_preds = []
                value_targets = []
                ratios = []

                for transition in batch:
                    new_log_prob, value, entropy, probs, current_log_probs = self._evaluate_transition(transition)
                    old_log_prob = torch.tensor(transition.old_log_prob, dtype=torch.float32, device=self.device)
                    advantage = torch.tensor(transition.advantage, dtype=torch.float32, device=self.device)
                    old_value = torch.tensor(transition.old_value, dtype=torch.float32, device=self.device)
                    return_target = torch.tensor(transition.return_target, dtype=torch.float32, device=self.device)

                    log_ratio = torch.clamp(new_log_prob - old_log_prob, -20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
                    policy_loss = -torch.min(unclipped, clipped)

                    value_delta = value - old_value
                    value_clipped = old_value + value_delta.clamp(-self.clip_ratio, self.clip_ratio)
                    value_loss = 0.5 * torch.max(
                        (value - return_target) ** 2,
                        (value_clipped - return_target) ** 2,
                    )

                    ref_kl = self._reference_kl(transition, current_log_probs, probs)
                    finite_terms = [
                        new_log_prob,
                        value,
                        entropy,
                        log_ratio,
                        ratio,
                        policy_loss,
                        value_loss,
                        ref_kl,
                    ]
                    if not all(torch.isfinite(term).all() for term in finite_terms):
                        skipped_nonfinite_terms += 1
                        continue

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)
                    approx_kls.append(old_log_prob.detach() - new_log_prob.detach())
                    clip_fracs.append(
                        torch.tensor(
                            float(abs(ratio.detach().item() - 1.0) > self.clip_ratio),
                            device=self.device,
                        )
                    )
                    ref_kls.append(ref_kl)
                    value_preds.append(value.detach())
                    value_targets.append(return_target.detach())
                    ratios.append(ratio.detach())

                if not policy_losses:
                    print("[RLTrainer] Skipping minibatch with no finite PPO transitions.")
                    skipped_empty += 1
                    continue

                loss = (
                    torch.stack(policy_losses).mean()
                    + self.value_loss_coef * torch.stack(value_losses).mean()
                    - self.entropy_coef * torch.stack(entropies).mean()
                )
                if self.reference_kl_coef > 0:
                    loss = loss + self.reference_kl_coef * torch.stack(ref_kls).mean()
                if not torch.isfinite(loss):
                    print("[RLTrainer] Skipping minibatch with non-finite PPO loss.")
                    skipped_nonfinite_loss += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    self.max_grad_norm,
                )
                grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                if not math.isfinite(grad_norm_value):
                    skipped_nonfinite_grad += 1
                    bad_grads = ", ".join(self._nonfinite_grad_names())
                    if bad_grads:
                        print(
                            "[RLTrainer] Skipping optimizer step due to non-finite gradient norm. "
                            f"First bad grads: {bad_grads}"
                        )
                    else:
                        print("[RLTrainer] Skipping optimizer step due to non-finite gradient norm.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
                scheduler.step()
                performed_update = True

                avg_policy = torch.stack(policy_losses).mean().item()
                avg_value = torch.stack(value_losses).mean().item()
                avg_entropy = torch.stack(entropies).mean().item()
                approx_kl = torch.stack(approx_kls).mean().item()
                clip_frac = torch.stack(clip_fracs).mean().item()
                ref_kl = torch.stack(ref_kls).mean().item()
                avg_value_pred = torch.stack(value_preds).mean().item()
                avg_value_target = torch.stack(value_targets).mean().item()
                avg_ratio = torch.stack(ratios).mean().item()

                writer.add_scalar("RL/PPO/policy_loss", avg_policy, global_step)
                writer.add_scalar("RL/PPO/value_loss", avg_value, global_step)
                writer.add_scalar("RL/PPO/avg_entropy", avg_entropy, global_step)
                writer.add_scalar("RL/PPO/approx_kl", approx_kl, global_step)
                writer.add_scalar("RL/PPO/reference_kl", ref_kl, global_step)
                writer.add_scalar("RL/PPO/clip_fraction", clip_frac, global_step)
                writer.add_scalar("RL/PPO/value_prediction_mean", avg_value_pred, global_step)
                writer.add_scalar("RL/PPO/value_target_mean", avg_value_target, global_step)
                writer.add_scalar("RL/PPO/ratio_mean", avg_ratio, global_step)
                writer.add_scalar("RL/PPO/grad_norm", grad_norm_value, global_step)
                writer.add_scalar("RL/PPO/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)

                epoch_policy.append(avg_policy)
                epoch_value.append(avg_value)
                epoch_entropy.append(avg_entropy)
                epoch_kl.append(approx_kl)
                epoch_clip.append(clip_frac)
                epoch_ref_kl.append(ref_kl)
                epoch_value_pred.extend([float(x.item()) for x in value_preds])
                epoch_targets.extend([float(x.item()) for x in value_targets])
                epoch_ratio.append(avg_ratio)
                epoch_grad_norm.append(grad_norm_value)

                pbar.set_postfix(
                    pol=f"{avg_policy:.4f}",
                    val=f"{avg_value:.4f}",
                    kl=f"{approx_kl:.4f}",
                    clip=f"{clip_frac:.2f}",
                    gnorm=f"{grad_norm_value:.2f}",
                )

                global_step += 1

                if self.target_kl > 0 and approx_kl > self.target_kl:
                    print(
                        f"[RLTrainer] Early stop at epoch {epoch + 1}: "
                        f"approx_kl {approx_kl:.4f} > target {self.target_kl:.4f}"
                    )
                    early_stop = True
                    break

            epoch_explained_var = self._explained_variance(epoch_value_pred, epoch_targets)
            epoch_step = epoch_step_offset + epoch
            print(
                f"[RLTrainer] Epoch {epoch + 1}/{self.ppo_epochs} summary | "
                f"policy={_mean(epoch_policy):.4f} | value={_mean(epoch_value):.4f} | "
                f"entropy={_mean(epoch_entropy):.4f} | approx_kl={_mean(epoch_kl):.4f} | "
                f"clip={_mean(epoch_clip):.2f} | ref_kl={_mean(epoch_ref_kl):.4f} | "
                f"value_pred={_mean(epoch_value_pred):.4f} | value_tgt={_mean(epoch_targets):.4f} | "
                f"explained_var={epoch_explained_var:.4f} | grad_norm={_mean(epoch_grad_norm):.2f} | "
                f"skips(empty/terms/loss/grad)="
                f"{skipped_empty}/{skipped_nonfinite_terms}/{skipped_nonfinite_loss}/{skipped_nonfinite_grad}"
            )
            writer.add_scalar("RL/Epoch/policy_loss", _mean(epoch_policy), epoch_step)
            writer.add_scalar("RL/Epoch/value_loss", _mean(epoch_value), epoch_step)
            writer.add_scalar("RL/Epoch/entropy", _mean(epoch_entropy), epoch_step)
            writer.add_scalar("RL/Epoch/approx_kl", _mean(epoch_kl), epoch_step)
            writer.add_scalar("RL/Epoch/clip_fraction", _mean(epoch_clip), epoch_step)
            writer.add_scalar("RL/Epoch/reference_kl", _mean(epoch_ref_kl), epoch_step)
            writer.add_scalar("RL/Epoch/value_prediction_mean", _mean(epoch_value_pred), epoch_step)
            writer.add_scalar("RL/Epoch/value_target_mean", _mean(epoch_targets), epoch_step)
            writer.add_scalar("RL/Epoch/explained_variance", epoch_explained_var, epoch_step)
            writer.add_scalar("RL/Epoch/ratio_mean", _mean(epoch_ratio), epoch_step)
            writer.add_scalar("RL/Epoch/grad_norm", _mean(epoch_grad_norm), epoch_step)
            writer.add_scalar("RL/Epoch/skipped_empty_minibatches", skipped_empty, epoch_step)
            writer.add_scalar("RL/Epoch/skipped_nonfinite_terms", skipped_nonfinite_terms, epoch_step)
            writer.add_scalar("RL/Epoch/skipped_nonfinite_loss", skipped_nonfinite_loss, epoch_step)
            writer.add_scalar("RL/Epoch/skipped_nonfinite_grad", skipped_nonfinite_grad, epoch_step)
            self._save_checkpoint("latest", scheduler, mean_episode_return)
            if early_stop:
                break

        if not performed_update:
            print("[RLTrainer] No finite PPO updates were applied; carrying forward the seed model unchanged.")
        self._save_checkpoint("best", scheduler, mean_episode_return)
        self._save_checkpoint("latest", scheduler, mean_episode_return)
        writer.close()
        print(
            f"[RLTrainer] PPO training complete. mean_episode_return={mean_episode_return:.3f} | "
            f"success_rate={run_success_rate:.1%} | transitions={len(transitions)}"
        )
