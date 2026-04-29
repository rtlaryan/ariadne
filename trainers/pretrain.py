"""
trainers/pretrain.py — Supervised pre-training (and DAgger fine-tuning).

Usage
-----
    from ariadne.trainers.pretrain import Trainer
    trainer = Trainer(cfg)
    trainer.train()

The *cfg* dict comes from the experiment YAML.  All path resolution is
done relative to the experiment directory supplied inside *cfg* or as
*run_dir*.
"""

import glob
import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ariadne.core.dataset import SoftwareTrajectoryDataset, _collect_jsonl, _iter_episodes
from ariadne.core.model import (
    AgentTransformer,
    PackedCollator,
    build_model,
    load_checkpoint,
    masked_ce_loss,
)
from ariadne.core.tokenizer import TokenMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_checkpoint(run_dir: str) -> Optional[str]:
    """Return the most recently modified .pt checkpoint in *run_dir*."""
    files = glob.glob(os.path.join(run_dir, "checkpoints", "*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def _resize_vocab(model: AgentTransformer, new_vocab_size: int) -> None:
    """Expand embedding + head matrices when vocabulary grows."""
    old = model.vocab_size
    if new_vocab_size <= old:
        return
    print(f"[Trainer] Resizing vocab {old} → {new_vocab_size}")
    # Token embedding
    old_w = model.tok_emb.weight.data
    model.tok_emb = nn.Embedding(new_vocab_size, model.embed_dim, padding_idx=0).to(old_w.device)
    model.tok_emb.weight.data[:old] = old_w
    # Head is weight-tied — retie
    model.head.weight = model.tok_emb.weight
    model.vocab_size  = new_vocab_size


def _load_dagger_collection_metrics(data_dirs: list[str]) -> Optional[dict]:
    """Load aggregate episode metrics for generated DAgger data when available."""
    summary_files: list[str] = []
    for root in data_dirs:
        if os.path.isdir(root):
            summary_files.extend(
                glob.glob(os.path.join(root, "**", "dagger_episodes_*.ndjson"), recursive=True)
            )

    total = success = 0
    total_steps = total_corrections = 0
    total_policy_mistakes = total_override_steps = total_recovery_steps = 0

    for path in summary_files:
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
                    total_corrections += int(ep.get("corrections", 0))
                    total_policy_mistakes += int(ep.get("policy_mistakes", 0))
                    total_override_steps += int(ep.get("override_steps", 0))
                    total_recovery_steps += int(ep.get("recovery_steps", 0))
        except OSError:
            continue

    if total > 0:
        denom = float(total)
        return {
            "source": "episode_summaries",
            "episodes": total,
            "successes": success,
            "success_rate": success / denom,
            "avg_steps": total_steps / denom,
            "avg_corrections": total_corrections / denom,
            "avg_policy_mistakes": total_policy_mistakes / denom,
            "avg_override_steps": total_override_steps / denom,
            "avg_recovery_steps": total_recovery_steps / denom,
        }

    files = [p for p in _collect_jsonl(data_dirs) if "dagger" in os.path.basename(p)]
    if not files:
        return None

    inferred_total = inferred_success = 0
    for episode in _iter_episodes(files):
        if not episode:
            continue
        inferred_total += 1
        action = episode[-1].get("action", {})
        key = action.get("key") if isinstance(action, dict) else str(action)
        if key == "Enter":
            inferred_success += 1

    if inferred_total == 0:
        return None

    return {
        "source": "step_labels_inferred",
        "episodes": inferred_total,
        "successes": inferred_success,
        "success_rate": inferred_success / float(inferred_total),
    }


def _batch_episode_success_stats(batch) -> tuple[float, int]:
    """Return (success_sum, episode_count) for the current training batch."""
    if isinstance(batch, dict):
        values = batch.get("episode_success")
        if values is None:
            return 0.0, 0
        if torch.is_tensor(values):
            if values.numel() == 0:
                return 0.0, 0
            return values.float().sum().item(), int(values.numel())
        values = list(values)
        return sum(float(v) for v in values), len(values)

    values = getattr(batch, "episode_successes", None)
    if not values:
        return 0.0, 0
    return sum(float(v) for v in values), len(values)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Supervised / DAgger trainer.

    Parameters (relevant *cfg* keys under the 'pretrain' or 'dagger' section):
      data_dir     (str|list)  – path(s) to the dataset directory/directories
      run_dir      (str)       – directory to save checkpoints / logs / tokenizer
      run_name     (str)       – TensorBoard run name (also used for checkpoint names)
      epochs       (int)       – number of training epochs
      batch_size   (int)
      lr           (float)
      lr_decay     (float)     – multiplicative LR decay across DAgger iterations
      resume_from  (str|None)  – path to a previous checkpoint to load weights from
      reset_scheduler (bool)   – restart LR scheduler when resuming

    Packing / architecture keys live at the top level of *cfg*
    (embed_dim, num_layers, num_heads, max_len, block_size, use_packing,
    model_version, dropout).
    """

    def __init__(
        self,
        cfg: dict,
        run_dir: str,
        use_dagger: bool = False,
        tb_log_dir: Optional[str] = None,
    ) -> None:
        self.cfg        = cfg
        self.run_dir    = run_dir
        self.use_dagger = use_dagger

        train_cfg = cfg.get("dagger", {}).get("training", {}) if use_dagger else cfg.get("pretrain", {})

        # Resolve paths
        data_dirs = train_cfg.get("data_dir", cfg.get("pretrain", {}).get("data_dir", "dataset"))
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs

        self.run_name    = train_cfg.get("run_name", "run")
        self.epochs      = int(train_cfg.get("epochs", 10))
        self.batch_size  = int(train_cfg.get("batch_size", 512))
        self.lr          = float(train_cfg.get("lr", 3e-4))
        self.lr_decay    = float(train_cfg.get("lr_decay", 1.0))
        # steps_per_epoch is used to size the LR scheduler for IterableDatasets
        # (which have no len()).  If not set, defaults to 1000.
        self.steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1000))
        self.resume_from: Optional[str] = train_cfg.get("resume_from")
        self.reset_sched: bool          = cfg.get("reset_scheduler", False)

        # DAgger mix params
        self.expert_multiplier = float(train_cfg.get("expert_multiplier", 1.0))
        self.decay_factor      = float(train_cfg.get("decay_factor", 1.0))
        self.current_iteration = int(train_cfg.get("current_iteration", 1))
        self.max_episodes      = train_cfg.get("max_episodes")
        self.effective_lr      = self._effective_base_lr()
        self.iteration_index   = max(0, self.current_iteration - 1) if self.use_dagger else 0

        # Architecture
        self.embed_dim   = cfg.get("embed_dim",   256)
        self.num_layers  = cfg.get("num_layers",  6)
        self.num_heads   = cfg.get("num_heads",   8)
        self.max_len     = cfg.get("max_len",     256)
        self.block_size  = cfg.get("block_size",  self.max_len)
        self.use_packing = cfg.get("use_packing", False)
        self.dropout     = cfg.get("dropout",     0.1)
        self.num_workers = train_cfg.get("num_workers", 2)
        self.use_amp     = bool(train_cfg.get("use_amp", False))

        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.ckpt_dir   = os.path.join(run_dir, "checkpoints")
        self.log_dir    = tb_log_dir if tb_log_dir else os.path.join(run_dir, "logs")
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        self._setup_tokenizer()
        self._setup_model()
        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _tokenizer_path(self) -> str:
        return os.path.join(self.run_dir, "tokenizer.json")

    def _effective_base_lr(self) -> float:
        if not self.use_dagger:
            return self.lr
        decay_step = max(0, self.current_iteration - 1)
        return self.lr * (self.lr_decay ** decay_step)

    def _phase_prefix(self) -> str:
        return "DAgger" if self.use_dagger else "Pretrain"

    def _iteration_step(self) -> int:
        return self.current_iteration if self.use_dagger else 0

    def _global_step_offset(self) -> int:
        return self.iteration_index * self.epochs * max(1, self.steps_per_epoch)

    def _epoch_step_offset(self) -> int:
        return self.iteration_index * self.epochs

    def _setup_tokenizer(self) -> None:
        tok_path = self._tokenizer_path()
        # Try to load existing tokenizer in run_dir
        if os.path.isfile(tok_path):
            self.tokenizer = TokenMap.load(tok_path)
            print(f"[Trainer] Loaded tokenizer from {tok_path} — vocab: {len(self.tokenizer)}")
        else:
            # Load base tokenizer from configs
            base = os.path.join(os.path.dirname(__file__), "..", "configs", "tokenizer.json")
            base = os.path.abspath(base)
            if os.path.isfile(base):
                self.tokenizer = TokenMap.load(base)
            else:
                self.tokenizer = TokenMap()
            print(f"[Trainer] Fresh tokenizer — vocab: {len(self.tokenizer)}")

    def _build_dataset(self) -> SoftwareTrajectoryDataset:
        files = _collect_jsonl(self.data_dirs)
        if not files:
            raise RuntimeError(f"No .jsonl files found in {self.data_dirs}")
        # First pass: build vocab from data
        from ariadne.core.dataset import _iter_episodes
        def episodes():
            for ep in _iter_episodes(files):
                yield ep
        self.tokenizer.build_from_data(episodes())
        # Save updated tokenizer
        self.tokenizer.save(self._tokenizer_path())

        return SoftwareTrajectoryDataset(
            data_files        = files,
            tokenizer         = self.tokenizer,
            max_len           = self.max_len,
            use_dagger        = self.use_dagger,
            expert_multiplier = self.expert_multiplier,
            decay_factor      = self.decay_factor,
            current_iteration = self.current_iteration,
            max_episodes      = self.max_episodes,
            use_packing       = self.use_packing,
        )

    def _arc_cfg(self) -> dict:
        return {
            "vocab_size":  len(self.tokenizer),
            "embed_dim":   self.embed_dim,
            "num_layers":  self.num_layers,
            "num_heads":   self.num_heads,
            "max_len":     self.max_len,
            "dropout":     self.dropout,
        }

    def _setup_model(self) -> None:
        self.model = build_model(self._arc_cfg()).to(self.device)
        print(f"[Trainer] Model — params: {sum(p.numel() for p in self.model.parameters()):,}")

        self._start_epoch = 0
        self._optimizer_state = None

        ckpt_path = self.resume_from or _find_best_checkpoint(self.run_dir)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[Trainer] Resuming from {ckpt_path}")
            ckpt = load_checkpoint(self.model, ckpt_path, device=self.device)
            if not self.reset_sched:
                self._start_epoch     = ckpt.get("epoch", 0)
                self._optimizer_state = ckpt.get("optimizer_state_dict")
        else:
            print("[Trainer] Starting from scratch.")

        # Save model config
        cfg_path = os.path.join(self.run_dir, "config.json")
        if not os.path.isfile(cfg_path):
            with open(cfg_path, "w") as f:
                json.dump(self._arc_cfg(), f, indent=2)

    def _setup_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.effective_lr, weight_decay=0.01
        )
        if self._optimizer_state and not self.reset_sched:
            try:
                self.optimizer.load_state_dict(self._optimizer_state)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.effective_lr
            except Exception as e:
                print(f"[Trainer] Could not restore optimizer state: {e}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save(self, epoch: int, tag: str) -> None:
        # Resize model to current vocab before saving (handles DAgger vocab growth)
        _resize_vocab(self.model, len(self.tokenizer))
        state = {
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        fname = f"{self.run_name}_{tag}.pt"
        path  = os.path.join(self.ckpt_dir, fname)
        torch.save(state, path)
        # Also always keep "best"
        best = os.path.join(self.ckpt_dir, f"{self.run_name}_best.pt")
        torch.save(state, best)
        self.tokenizer.save(self._tokenizer_path())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        dataset = self._build_dataset()

        # Resize model if vocab grew
        _resize_vocab(self.model, len(self.tokenizer))

        collate = None
        if self.use_packing:
            collate = PackedCollator(
                block_size       = self.block_size,
                pad_token_id     = self.tokenizer.token_to_id.get("[PAD]", 0),
                return_attn_mask = True,
            )

        loader = DataLoader(
            dataset,
            batch_size       = self.batch_size,
            num_workers      = self.num_workers,
            pin_memory       = True,
            collate_fn       = collate,
            persistent_workers = self.num_workers > 0,
        )

        # ------------------------------------------------------------------
        # LR Scheduler: linear warm-up (1 epoch) + cosine decay to 5 % of
        # peak LR, anchored to the fixed `epochs` config value.
        #
        # Using epochs as the unit (instead of steps) means the curve shape
        # is identical regardless of dataset size, batch size, or how many
        # DAgger iterations have been run.  The scheduler is stepped once per
        # epoch, not per batch.
        # ------------------------------------------------------------------
        warmup_epochs = 1 if self.use_dagger else 0
        total_epochs  = max(1, self.epochs)
        eta_ratio     = 0.05  # final LR = lr * eta_ratio

        def _lr_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                # Linear ramp from 0 → 1 over warmup_epochs
                return (epoch_idx + 1) / warmup_epochs
            progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            # Cosine from 1 → eta_ratio
            return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        # Reset optimizer LR to the effective base LR so the lambda multipliers
        # start from the right value when DAgger iteration-level decay is used.
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.effective_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)

        print(
            f"[Trainer] Base LR: {self.effective_lr:.2e}"
            + (
                f" (decayed from {self.lr:.2e} for DAgger iter {self.current_iteration})"
                if self.use_dagger and self.current_iteration > 1
                else ""
            )
        )

        writer = SummaryWriter(self.log_dir)
        phase_prefix = self._phase_prefix()
        global_step = self._global_step_offset()
        epoch_offset = self._epoch_step_offset()
        best_loss   = float("inf")
        dagger_metrics = _load_dagger_collection_metrics(self.data_dirs) if self.use_dagger else None
        if dagger_metrics:
            msg = (
                f"[Trainer] DAgger collection success {dagger_metrics['successes']}/"
                f"{dagger_metrics['episodes']} ({dagger_metrics['success_rate']:.1%}) "
                f"[source={dagger_metrics['source']}]"
            )
            if "avg_steps" in dagger_metrics:
                msg += (
                    f" | avg_steps={dagger_metrics['avg_steps']:.2f}"
                    f" avg_recovery_steps={dagger_metrics['avg_recovery_steps']:.2f}"
                )
            print(msg)
            writer.add_scalar(
                "DAgger/Collection/success_rate",
                dagger_metrics["success_rate"],
                self._iteration_step(),
            )
            writer.add_scalar(
                "DAgger/Collection/episodes",
                dagger_metrics["episodes"],
                self._iteration_step(),
            )
            if "avg_steps" in dagger_metrics:
                writer.add_scalar(
                    "DAgger/Collection/avg_steps",
                    dagger_metrics["avg_steps"],
                    self._iteration_step(),
                )
                writer.add_scalar(
                    "DAgger/Collection/avg_corrections",
                    dagger_metrics["avg_corrections"],
                    self._iteration_step(),
                )
                writer.add_scalar(
                    "DAgger/Collection/avg_policy_mistakes",
                    dagger_metrics["avg_policy_mistakes"],
                    self._iteration_step(),
                )
                writer.add_scalar(
                    "DAgger/Collection/avg_override_steps",
                    dagger_metrics["avg_override_steps"],
                    self._iteration_step(),
                )
                writer.add_scalar(
                    "DAgger/Collection/avg_recovery_steps",
                    dagger_metrics["avg_recovery_steps"],
                    self._iteration_step(),
                )

        for epoch in range(self._start_epoch, self._start_epoch + self.epochs):
            self.model.train()
            epoch_loss = epoch_acc = epoch_episode_success = epoch_n = 0
            epoch_episode_count = 0
            epoch_step = epoch_offset + epoch

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self._start_epoch + self.epochs} [{self.run_name}]")
            for batch in pbar:
                batch_episode_success_sum, batch_episode_count = _batch_episode_success_stats(batch)
                batch_episode_success = (
                    batch_episode_success_sum / batch_episode_count
                    if batch_episode_count > 0 else 0.0
                )
                # Handle packed vs. standard batch
                if self.use_packing:
                    input_ids = batch.input_ids.to(self.device, non_blocking=True)
                    labels    = batch.labels.to(self.device, non_blocking=True)
                    loss_mask = batch.loss_mask.to(self.device, non_blocking=True)
                    attn_mask = batch.attn_mask.to(self.device, non_blocking=True) if batch.attn_mask is not None else None
                    pos_ids   = batch.position_ids.to(self.device, non_blocking=True) if getattr(batch, "position_ids", None) is not None else None

                    if input_ids.shape[0] == 0:
                        continue

                    # Forward pass
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                        logits     = self.model(input_ids, attn_mask=attn_mask, position_ids=pos_ids)
                        loss       = masked_ce_loss(logits, labels, loss_mask)
                    # Accuracy over masked positions
                    preds      = logits.detach().argmax(-1)
                    acc        = (preds[loss_mask] == labels[loss_mask]).float().mean().item() if loss_mask.any() else 0.0
                else:
                    input_ids  = batch["input_ids"].to(self.device)
                    labels     = batch["labels"].to(self.device)
                    # Causal LM: predict token i+1 from token i.
                    # input_ids is [B, L] (already padded to max_len).
                    # labels    is [B, L] with -100 on non-action positions.
                    # We feed input_ids[:,:-1] and predict labels[:,1:].
                    inp    = input_ids[:, :-1]   # [B, L-1]
                    tgt    = labels[:, 1:]        # [B, L-1]
                    logits = self.model(inp)      # [B, L-1, V]
                    mask   = tgt != -100
                    loss   = masked_ce_loss(logits, tgt, mask)
                    preds  = logits.detach().argmax(-1)
                    acc    = (preds[mask] == tgt[mask]).float().mean().item() if mask.any() else 0.0

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                l = loss.item()
                epoch_loss += l
                epoch_acc  += acc
                epoch_episode_success += batch_episode_success_sum
                epoch_episode_count += batch_episode_count
                epoch_n    += 1
                global_step += 1

                writer.add_scalar(f"{phase_prefix}/Train/loss_step", l, global_step)
                writer.add_scalar(f"{phase_prefix}/Train/accuracy_step", acc, global_step)
                writer.add_scalar(
                    f"{phase_prefix}/Train/episode_success_step",
                    batch_episode_success,
                    global_step,
                )

                pbar.set_postfix(
                    {
                        "loss": f"{l:.4f}",
                        "acc": f"{acc:.4f}",
                        "ep_succ": f"{batch_episode_success:.4f}",
                    }
                )

                # Live control.json override
                ctrl = os.path.join(self.run_dir, "control.json")
                if os.path.isfile(ctrl):
                    try:
                        with open(ctrl) as f:
                            ctrl_data = json.load(f)
                        new_lr = ctrl_data.get("lr")
                        if new_lr:
                            for pg in self.optimizer.param_groups:
                                pg["lr"] = float(new_lr)
                    except Exception:
                        pass

            avg_loss = epoch_loss / max(1, epoch_n)
            avg_acc  = epoch_acc  / max(1, epoch_n)
            avg_episode_success = epoch_episode_success / max(1, epoch_episode_count)
            writer.add_scalar(f"{phase_prefix}/Train/loss_epoch", avg_loss, epoch_step)
            writer.add_scalar(f"{phase_prefix}/Train/accuracy_epoch", avg_acc, epoch_step)
            writer.add_scalar(
                f"{phase_prefix}/Train/episode_success_epoch",
                avg_episode_success,
                epoch_step,
            )

            # Advance the epoch-level LR scheduler
            scheduler.step()
            writer.add_scalar(
                f"{phase_prefix}/Train/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                epoch_step,
            )

            print(
                f"[Trainer] Epoch {epoch+1} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f} "
                f"| EpisodeSuccess {avg_episode_success:.4f} "
                f"| LR {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            self._save(epoch + 1, "latest")
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save(epoch + 1, "best")

        writer.close()
        print(f"[Trainer] Training complete. Best loss: {best_loss:.4f}")
