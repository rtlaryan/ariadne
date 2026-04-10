"""
trainers/rl.py — REINFORCE reinforcement-learning fine-tuning.

Algorithm:
  1. Load rollout episodes (one JSON object per episode, written by rl_agent.py)
  2. Per episode: compute discounted returns, compute advantages (vs. baseline)
  3. Policy gradient update with entropy bonus + KL divergence penalty
  4. Optionally mix in supervised loss computed over expert data

All hyperparameters come from the experiment YAML (rl.training section).
"""

import glob
import json
import math
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ariadne.core.dataset import SoftwareTrajectoryDataset, StateSerializer
from ariadne.core.model import (
    AgentTransformer,
    PackedCollator,
    build_model,
    load_checkpoint,
    masked_ce_loss,
)
from ariadne.core.tokenizer import TokenMap


# ---------------------------------------------------------------------------
# Rollout loading
# ---------------------------------------------------------------------------

def _load_rollouts(rollout_dir: str, success_only: bool = True) -> list[dict]:
    """Return a list of episode dicts from rl_agent JSONL rollout files."""
    episodes = []
    for path in glob.glob(os.path.join(rollout_dir, "**/*.jsonl"), recursive=True):
        with open(path, "r") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ep = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if success_only and not ep.get("success", False):
                    continue
                episodes.append(ep)
    return episodes


# ---------------------------------------------------------------------------
# Reward / return computation
# ---------------------------------------------------------------------------

def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def _normalize(values: list[float]) -> list[float]:
    arr = torch.tensor(values, dtype=torch.float32)
    if arr.numel() <= 1:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / (arr.std() + 1e-8)).tolist()


# ---------------------------------------------------------------------------
# Valid-action masking
# ---------------------------------------------------------------------------

_NORMALIZE_KEYS = {
    "÷": "/", "×": "*", "⌫": "Backspace", "AC": "Escape", "=": "Enter"
}


def _valid_action_mask(
    tokenizer: TokenMap,
    available: list[str],
    device: str,
    vocab_size: int,
) -> torch.Tensor:
    """Return additive logit mask (0 = allowed, -1e9 = blocked)."""
    mask = torch.full((vocab_size,), -1e9, dtype=torch.float32, device=device)
    for k in available:
        k = _NORMALIZE_KEYS.get(k, k)
        idx = tokenizer.token_to_id.get(k)
        if idx is not None:
            mask[idx] = 0.0
    return mask


# ---------------------------------------------------------------------------
# RL Trainer
# ---------------------------------------------------------------------------

class RLTrainer:
    """REINFORCE trainer with KL penalty and optional supervised anchor.

    Parameters (cfg keys under rl.training):
      lr                float – learning rate
      lr_decay          float – multiplicative LR decay per iteration
      gamma             float – discount factor
      entropy_bonus     float – entropy regularization coefficient
      entropy_decay     float – multiplicative decay for entropy_bonus per iteration
      kl_penalty        float – KL divergence penalty coefficient
      kl_decay          float – per-iteration decay for kl_penalty
      max_grad_norm     float – gradient clipping
      baseline_alpha    float – EMA coefficient for return baseline
      rl_epochs         int   – gradient passes per iteration
      rl_batch_size     int   – episodes per gradient step
      success_only      bool  – only train on successful episodes
      supervised_weight float – fraction of supervised loss added to RL loss
    """

    def __init__(
        self,
        cfg:            dict,
        run_dir:        str,
        rollout_dir:    str,
        resume_from:    str,
        reference_from: str,
        tokenizer_path: str,
        iteration_index: int  = 0,
        decay_step:     int   = 0,
        run_name:       str   = "rl",
        tb_log_dir:     Optional[str] = None,
    ) -> None:
        self.cfg             = cfg
        self.run_dir         = run_dir
        self.rollout_dir     = rollout_dir
        self.run_name        = run_name
        self.iteration_index = iteration_index

        train_cfg = cfg.get("rl", {}).get("training", {})

        self.lr               = float(train_cfg.get("lr",               1e-5))
        self.lr_decay         = float(train_cfg.get("lr_decay",         1.0))
        self.gamma            = float(train_cfg.get("gamma",            0.99))
        self.entropy_bonus    = float(train_cfg.get("entropy_bonus",    0.01))  * (float(train_cfg.get("entropy_decay", 1.0)) ** decay_step)
        self.kl_penalty       = float(train_cfg.get("kl_penalty",       0.1))   * (float(train_cfg.get("kl_decay",      1.0)) ** decay_step)
        self.max_grad_norm    = float(train_cfg.get("max_grad_norm",    0.5))
        self.baseline_alpha   = float(train_cfg.get("baseline_alpha",   0.99))
        self.rl_epochs        = int(train_cfg.get("rl_epochs",        1))
        self.rl_batch_size    = int(train_cfg.get("rl_batch_size",    16))
        self.success_only     = bool(train_cfg.get("success_only",     True))
        self.sup_weight       = float(train_cfg.get("supervised_weight", 0.0))
        self.min_success_rate = float(train_cfg.get("min_success_rate", 0.0))
        self.effective_lr     = self.lr * (self.lr_decay ** decay_step)

        self.lr_warmup_steps = int(train_cfg.get("lr_warmup_steps",   0))
        self.lr_min_ratio     = float(train_cfg.get("lr_min_ratio",    0.05))
        self.use_packing  = cfg.get("use_packing",  False)
        self.block_size   = cfg.get("block_size",   256)
        self.max_len      = cfg.get("max_len",      256)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenizer
        self.tokenizer = TokenMap.load(tokenizer_path)
        self.serializer = StateSerializer(self.tokenizer)
        print(f"[RLTrainer] Tokenizer loaded from {tokenizer_path} — vocab: {len(self.tokenizer)}")

        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        # Use shared log dir if provided (so all RL iters are one TB run);
        # fall back to per-iteration dir.
        self.log_dir  = tb_log_dir if tb_log_dir else os.path.join(run_dir, "logs")

        self._setup_model(resume_from)
        self._setup_reference(reference_from)
        self.baseline = 0.0

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _arc_cfg(self) -> dict:
        return {
            "vocab_size": len(self.tokenizer),
            "embed_dim":  self.cfg.get("embed_dim",  256),
            "num_layers": self.cfg.get("num_layers", 6),
            "num_heads":  self.cfg.get("num_heads",  8),
            "max_len":    self.max_len,
            "dropout":    self.cfg.get("dropout",    0.0),   # no dropout during RL
        }

    def _setup_model(self, resume_from: str) -> None:
        self.model = build_model(self._arc_cfg()).to(self.device)
        ckpt = load_checkpoint(self.model, resume_from, device=self.device)
        # Restore baseline EMA if saved
        self.baseline = ckpt.get("rl_baseline", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.effective_lr
        )
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
                # Override LR with current effective LR
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.effective_lr
            except Exception:
                pass
        print(
            f"[RLTrainer] Policy model loaded from {resume_from}. "
            f"Effective LR: {self.effective_lr:.2e}"
        )

        # Save model config so inference agents can reconstruct the architecture correctly
        cfg_path = os.path.join(self.run_dir, "config.json")
        if not os.path.isfile(cfg_path):
            import json
            with open(cfg_path, "w") as f:
                json.dump(self._arc_cfg(), f, indent=2)

    def _build_scheduler(self, total_steps: int):
        """Build a warmup + cosine decay LR scheduler for *total_steps* steps.

        The scheduler is intended to be stepped every gradient update.
        If total_steps <= 0 the LR stays constant.
        """
        warmup = self.lr_warmup_steps
        eta    = self.lr_min_ratio

        def _lr_lambda(step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, total_steps - warmup)
            return eta + (1.0 - eta) * 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)

    def _setup_reference(self, reference_from: str) -> None:
        self.ref_model = build_model(self._arc_cfg()).to(self.device)
        load_checkpoint(self.ref_model, reference_from, device=self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        print(f"[RLTrainer] Reference model loaded from {reference_from}")

    # ------------------------------------------------------------------
    # Sequence encoding
    # ------------------------------------------------------------------

    def _encode_step(self, step: dict) -> Optional[tuple[torch.Tensor, int]]:
        """Encode a single rollout step into (input_ids, action_id).

        Returns None if the step cannot be encoded (missing action token etc.).
        """
        state     = step.get("state", {})
        key       = step.get("action", "")
        action_id = self.tokenizer.token_to_id.get(key)
        if action_id is None:
            return None

        # Reconstruct the token sequence exactly as the agent does
        task     = state.get("_goal_override") or state.get("task") or ""
        goal_toks = self.serializer.tokenize_expr(task)
        st_toks   = self.serializer.serialize(state)
        full      = ["[GOAL]"] + goal_toks + ["[STATE]"] + st_toks + ["[ACTION]"]
        ids       = self.tokenizer.encode(full)
        if len(ids) > self.max_len - 1:
            ids = ids[-(self.max_len - 1):]
        return torch.tensor(ids, dtype=torch.long), action_id

    def _get_logits(self, model: AgentTransformer, ids: torch.Tensor) -> torch.Tensor:
        """Run *model* and return logits at the last position."""
        inp = ids.unsqueeze(0).to(self.device)
        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            out = model(inp)
        return out[0, -1, :]   # [V]

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _episode_loss(
        self,
        episode: dict,
        advantages: list[float],
    ) -> tuple[Optional[torch.Tensor], dict]:
        policy_terms = []
        entropy_terms = []
        kl_terms = []
        valid_steps = 0

        steps = episode.get("steps", [])
        assert len(steps) == len(advantages)

        for step, adv in zip(steps, advantages):
            encoded = self._encode_step(step)
            if encoded is None:
                continue
            ids, action_id = encoded

            # Available interactions → valid-action mask
            avail = step.get("state", {}).get("availableInteractions", [])
            va_mask = _valid_action_mask(
                self.tokenizer, avail, self.device, len(self.tokenizer)
            ) if avail else None

            # Policy logits (with grad)
            policy_logits = self._get_logits(self.model, ids)
            if va_mask is not None:
                masked_pol = policy_logits + va_mask
            else:
                masked_pol = policy_logits
            log_probs = F.log_softmax(masked_pol, dim=-1)
            log_p     = log_probs[action_id]

            # Skip steps with degenerate log prob
            if log_p.item() < -30.0:
                continue

            # Reference log prob (no grad)
            with torch.no_grad():
                ref_logits = self._get_logits(self.ref_model, ids)
                if va_mask is not None:
                    masked_ref = ref_logits + va_mask
                else:
                    masked_ref = ref_logits
                ref_log_probs = F.log_softmax(masked_ref, dim=-1)

            probs = log_probs.detach().exp()
            entropy = -(probs * log_probs.detach()).sum()

            # KL divergence (forward) per token
            kl = F.kl_div(ref_log_probs, probs, reduction="sum")

            policy_terms.append(log_p * adv)
            entropy_terms.append(entropy)
            kl_terms.append(kl)
            valid_steps += 1

        if not policy_terms:
            return None, {}

        policy_loss  = -torch.stack(policy_terms).mean()
        entropy_loss = -self.entropy_bonus * torch.stack(entropy_terms).mean()
        kl_loss      =  self.kl_penalty    * torch.stack(kl_terms).mean()
        total        = policy_loss + entropy_loss + kl_loss

        stats = {
            "policy_loss":  policy_loss.item(),
            "entropy":      (-entropy_loss / (self.entropy_bonus + 1e-9)).item(),
            "kl":           (kl_loss       / (self.kl_penalty    + 1e-9)).item(),
            "valid_steps":  valid_steps,
        }
        return total, stats

    # ------------------------------------------------------------------
    # Supervised anchor
    # ------------------------------------------------------------------

    def _supervised_loss(
        self,
        sup_files: list[str],
    ) -> Optional[torch.Tensor]:
        if not sup_files or self.sup_weight <= 0:
            return None
        ds = SoftwareTrajectoryDataset(
            data_files    = sup_files,
            tokenizer     = self.tokenizer,
            max_len       = self.max_len,
            max_episodes  = max(32, self.rl_batch_size * 4),
            use_packing   = self.use_packing,
        )
        collate = (
            PackedCollator(self.block_size, self.tokenizer.token_to_id.get("[PAD]", 0), return_attn_mask=True)
            if self.use_packing else None
        )
        loader = DataLoader(ds, batch_size=self.rl_batch_size, collate_fn=collate, num_workers=0)
        losses = []
        self.model.train()
        for batch in loader:
            if self.use_packing:
                inp  = batch.input_ids.to(self.device)
                lbl  = batch.labels.to(self.device)
                mask = batch.loss_mask.to(self.device)
                atm  = batch.attn_mask.to(self.device) if batch.attn_mask is not None else None
                pid  = batch.position_ids.to(self.device) if getattr(batch, "position_ids", None) is not None else None
                logits = self.model(inp, attn_mask=atm, position_ids=pid)
                losses.append(masked_ce_loss(logits, lbl, mask))
            else:
                inp  = batch["input_ids"].to(self.device)
                lbl  = batch["labels"].to(self.device)
                logits = self.model(inp[:, :-1])
                mask   = lbl[:, 1:] != -100
                losses.append(masked_ce_loss(logits, lbl[:, 1:], mask))
            if len(losses) >= 4:
                break
        if not losses:
            return None
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save(self, tag: str) -> None:
        state = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rl_baseline":          self.baseline,
            "iteration":            self.iteration_index,
        }
        for fname in [f"{self.run_name}_{tag}.pt", f"{self.run_name}_best.pt"]:
            torch.save(state, os.path.join(self.ckpt_dir, fname))
        self.tokenizer.save(os.path.join(self.run_dir, "tokenizer.json"))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, sup_files: Optional[list[str]] = None) -> None:
        all_episodes = _load_rollouts(self.rollout_dir, success_only=False)
        if not all_episodes:
            print("[RLTrainer] No valid rollout episodes found. Skipping.")
            return

        n_total  = len(all_episodes)
        n_suc    = sum(1 for e in all_episodes if e.get("success", False))
        rate     = n_suc / n_total if n_total else 0.0
        print(f"[RLTrainer] {n_suc}/{n_total} successful episodes ({rate:.1%})")

        if rate < self.min_success_rate:
            print(f"[RLTrainer] Success rate {rate:.1%} below minimum {self.min_success_rate:.1%}. Skipping.")
            return

        if self.success_only:
            episodes = [e for e in all_episodes if e.get("success", False)]
        else:
            episodes = all_episodes

        if not episodes:
            print("[RLTrainer] No successful rollout episodes found for training. Skipping.")
            return

        writer      = SummaryWriter(self.log_dir)
        # Offset so every RL iteration continues from where the previous left
        # off on the shared TensorBoard x-axis instead of restarting from 0.
        ep_per_iter  = len(episodes)
        batches_per_epoch = max(1, math.ceil(ep_per_iter / self.rl_batch_size))
        global_step = self.iteration_index * self.rl_epochs * batches_per_epoch
        best_pol_loss = float("inf")

        # Build the within-iteration LR scheduler.
        # total_steps = gradient updates across all rl_epochs for this iteration.
        total_steps = self.rl_epochs * batches_per_epoch
        scheduler   = self._build_scheduler(total_steps)
        for epoch in range(self.rl_epochs):
            random.shuffle(episodes)
            batches = [
                episodes[i : i + self.rl_batch_size]
                for i in range(0, len(episodes), self.rl_batch_size)
            ]

            pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{self.rl_epochs} [{self.run_name}]")
            for batch in pbar:
                self.model.train()
                self.optimizer.zero_grad()

                batch_pol_losses  = []
                batch_entropies   = []
                batch_kls         = []
                batch_valid_steps = []
                batch_advantages  = []
                batch_successes   = 0

                batch_returns = []
                batch_episodes = []

                for episode in batch:
                    rewards   = episode.get("step_rewards", [1.0] if episode.get("success") else [0.0])
                    returns   = _discounted_returns(rewards, self.gamma)

                    # Update baseline with weighted return
                    ep_return = sum(returns)
                    self.baseline = (
                        self.baseline_alpha * self.baseline
                        + (1 - self.baseline_alpha) * ep_return
                    )

                    batch_returns.append(returns)
                    batch_episodes.append(episode)

                # Flatten all returns to compute batch-level advantages
                flat_returns = [r for returns in batch_returns for r in returns]
                
                # Baseline subtraction first
                flat_advantages = [r - self.baseline for r in flat_returns]

                # Then normalize across the whole batch
                flat_advantages = _normalize(flat_advantages)
                
                # Unflatten advantages back to episodes
                adv_idx = 0
                for episode, returns in zip(batch_episodes, batch_returns):
                    ep_len = len(returns)
                    advantages = flat_advantages[adv_idx : adv_idx + ep_len]
                    adv_idx += ep_len

                    loss, stats = self._episode_loss(episode, advantages)
                    if loss is None:
                        continue
                    
                    loss.backward()
                    batch_pol_losses.append(stats["policy_loss"])
                    batch_entropies.append(stats["entropy"])
                    batch_kls.append(stats["kl"])
                    batch_valid_steps.append(stats["valid_steps"])
                    batch_advantages.extend(advantages)
                    if episode.get("success"):
                        batch_successes += 1

                # Supervised anchor
                if sup_files and self.sup_weight > 0:
                    sup_loss = self._supervised_loss(sup_files)
                    if sup_loss is not None:
                        (self.sup_weight * sup_loss).backward()

                n_valid = len(batch_pol_losses)
                if n_valid > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    avg_pol = sum(batch_pol_losses)  / n_valid
                    avg_ent = sum(batch_entropies)   / n_valid
                    avg_kl  = sum(batch_kls)         / n_valid
                    avg_adv = sum(batch_advantages)  / max(len(batch_advantages), 1)

                    writer.add_scalar("RL_Progress/policy_loss", avg_pol, global_step)
                    writer.add_scalar("RL_Progress/avg_entropy", avg_ent, global_step)
                    writer.add_scalar("RL_Progress/avg_kl_divergence", avg_kl, global_step)
                    writer.add_scalar("RL_Progress/baseline", self.baseline, global_step)
                    writer.add_scalar("RL_Progress/success_rate", rate, global_step)

                    writer.add_scalar("RL/avg_advantage", avg_adv, global_step)
                    writer.add_scalar("RL/avg_valid_steps", sum(batch_valid_steps) / n_valid, global_step)
                    writer.add_scalar("RL/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)

                    scheduler.step()

                    pbar.set_postfix({
                        "pol": f"{avg_pol:.4f}",
                        "ent": f"{avg_ent:.4f}",
                        "kl": f"{avg_kl:.4f}"
                    })

                    global_step += 1

                    if avg_pol < best_pol_loss:
                        best_pol_loss = avg_pol
                        self._save("best")

            # Epoch checkpoint
            self._save("latest")
            print(
                f"[RLTrainer] Epoch {epoch+1}/{self.rl_epochs} done. "
                f"Best policy loss: {best_pol_loss:.4f}"
            )

        writer.close()
        print("[RLTrainer] RL training complete.")
