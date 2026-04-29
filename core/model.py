"""
core/model.py — Causal transformer model for the Ariadne agent.

A single modern architecture (V2Lite):
  - Learned absolute position embeddings
  - RMSNorm + SwiGLU FFN
  - Flash-SDPA causal attention
  - Tied input/output embeddings

Also contains sequence-packing utilities (PackedBatch, pack_to_blocks,
PackedCollator) and the masked cross-entropy loss helper.

Factory
-------
    cfg = {"vocab_size": 64, "embed_dim": 256, "num_layers": 6,
           "num_heads": 8, "max_len": 512, "dropout": 0.1}
    model = build_model(cfg)
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms.to(x.dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_out  = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.dropout_p  = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        dp = self.dropout_p if self.training else 0.0

        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1), dropout_p=dp)
        else:
            # attn_mask: bool [B, T, T]  (True = allow)  →  add head dim
            mask = attn_mask[:, None, :, :].to(torch.bool)  # [B, 1, T, T]
            if T > 1:
                causal = torch.tril(
                    torch.ones(T, T, dtype=torch.bool, device=x.device)
                ).view(1, 1, T, T)
                mask = mask & causal
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.1, ffn_mult: float = 8 / 3
    ) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = RMSNorm(embed_dim)
        hidden = ((int(ffn_mult * embed_dim) + 63) // 64) * 64
        self.ffn  = SwiGLU(embed_dim, hidden, dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AgentTransformer(nn.Module):
    """Causal transformer for agent action prediction.

    Input  : token ids  [B, T]
    Output : logits     [B, T, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int  = 256,
        num_layers: int = 6,
        num_heads: int  = 8,
        max_len: int    = 512,
        dropout: float  = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.max_len      = max_len
        self.pad_token_id = pad_token_id

        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop    = nn.Dropout(dropout)
        self.layers  = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm    = RMSNorm(embed_dim)
        self.head    = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()
        # Weight tying
        self.head.weight = self.tok_emb.weight

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        if position_ids is None:
            if T > self.max_len:
                raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}.")
            pos = torch.arange(T, device=input_ids.device)
        else:
            pos = position_ids
            
        x   = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.layers:
            x = block(x, attn_mask=attn_mask)
        hidden = self.norm(x)
        logits = self.head(hidden)
        if return_hidden_states:
            return logits, hidden
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict, compile: bool = False) -> AgentTransformer:
    """Create an AgentTransformer from a config dict.

    Required keys: vocab_size
    Optional keys: embed_dim, num_layers, num_heads, max_len, dropout, pad_token_id
    """
    model = AgentTransformer(
        vocab_size   = cfg["vocab_size"],
        embed_dim    = cfg.get("embed_dim",    256),
        num_layers   = cfg.get("num_layers",   6),
        num_heads    = cfg.get("num_heads",    8),
        max_len      = cfg.get("max_len",      512),
        dropout      = cfg.get("dropout",      0.1),
        pad_token_id = cfg.get("pad_token_id", 0),
    )
    if compile:
        model = torch.compile(model)
    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: AgentTransformer,
    path: str,
    device: str = "cpu",
    strict: bool = False,
) -> dict:
    """Load a checkpoint, handling vocab-size mismatches gracefully.

    Returns the full checkpoint dict (contains 'optimizer_state_dict' etc.).
    """
    ckpt = torch.load(path, map_location=device)
    src  = ckpt["model_state_dict"]
    dst  = model.state_dict()

    new_state: dict = {}
    for k, v in src.items():
        if k not in dst:
            continue
        if dst[k].shape == v.shape:
            new_state[k] = v
        elif "tok_emb" in k or "head" in k:
            # Vocab size mismatch: copy the overlapping slice
            min_v = min(v.shape[0], dst[k].shape[0])
            if v.dim() == 2:
                dst[k][:min_v, :] = v[:min_v, :]
            else:
                dst[k][:min_v] = v[:min_v]
        # else: skip silently

    dst.update(new_state)
    model.load_state_dict(dst, strict=False)
    return ckpt


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------

@dataclass
class PackedBatch:
    input_ids:      torch.Tensor              # [B, T]
    labels:         torch.Tensor              # [B, T]
    loss_mask:      torch.Tensor              # [B, T] bool
    attn_mask:      Optional[torch.Tensor]    # [B, T, T] bool  (True=allow)
    position_ids:   Optional[torch.Tensor]    # [B, T]
    sources:        Optional[List[int]]       # source ids per sequence (pre-pack)
    episode_successes: Optional[List[float]]  # success flags per sequence (pre-pack)


def pack_to_blocks(
    sequences:        List[torch.Tensor],
    label_sequences:  List[torch.Tensor],
    block_size:       int,
    pad_token_id:     int,
    sep_token_id:     Optional[int] = None,
    drop_last:        bool = True,
    return_attn_mask: bool = False,
    sources:          Optional[List[int]] = None,
    episode_successes: Optional[List[float]] = None,
) -> PackedBatch:
    """Pack variable-length sequences into fixed-size blocks.

    Each block is block_size tokens long (input = block[:-1], label = block[1:]).
    Sequences are separated by sep_token_id if provided.
    When return_attn_mask=True, the mask prevents cross-sequence attention.
    """
    assert block_size >= 2

    flat: List[torch.Tensor] = []
    flat_labels: List[torch.Tensor] = []
    seq_ids: List[torch.Tensor] = []
    pos_ids: List[torch.Tensor] = []

    for i, (s, l) in enumerate(zip(sequences, label_sequences)):
        if s.numel() == 0:
            continue
        flat.append(s)
        flat_labels.append(l)
        seq_ids.append(torch.full((s.numel(),), i, dtype=torch.long))
        pos_ids.append(torch.arange(s.numel(), dtype=torch.long))
        if sep_token_id is not None and i < len(sequences) - 1:
            flat.append(torch.tensor([sep_token_id], dtype=torch.long))
            flat_labels.append(torch.tensor([-100], dtype=torch.long))
            seq_ids.append(torch.tensor([i], dtype=torch.long))
            pos_ids.append(torch.tensor([s.numel()], dtype=torch.long))

    if not flat:
        empty_ids  = torch.full((1, block_size), pad_token_id, dtype=torch.long)
        empty_lab  = torch.full((1, block_size), -100, dtype=torch.long)
        empty_mask = torch.zeros((1, block_size), dtype=torch.bool)
        attn = torch.zeros((1, block_size, block_size), dtype=torch.bool) if return_attn_mask else None
        p_ids = torch.zeros((1, block_size), dtype=torch.long)
        return PackedBatch(empty_ids, empty_lab, empty_mask, attn, p_ids, sources, episode_successes)

    stream      = torch.cat(flat,       dim=0)
    stream_lab  = torch.cat(flat_labels, dim=0)
    stream_ids  = torch.cat(seq_ids,    dim=0)
    stream_pos  = torch.cat(pos_ids,    dim=0)

    span  = block_size + 1
    n_blk = stream.numel() // span if drop_last else math.ceil(stream.numel() / span)
    total = n_blk * span

    if stream.numel() < total:
        pad = total - stream.numel()
        stream     = torch.cat([stream,     torch.full((pad,), pad_token_id, dtype=torch.long)])
        stream_lab = torch.cat([stream_lab, torch.full((pad,), -100,         dtype=torch.long)])
        stream_ids = torch.cat([stream_ids, torch.full((pad,), -1,           dtype=torch.long)])
        stream_pos = torch.cat([stream_pos, torch.zeros((pad,),              dtype=torch.long)])
    else:
        stream     = stream[:total]
        stream_lab = stream_lab[:total]
        stream_ids = stream_ids[:total]
        stream_pos = stream_pos[:total]

    blocks     = stream.view(n_blk, span)
    blocks_lab = stream_lab.view(n_blk, span)
    blocks_ids = stream_ids.view(n_blk, span)
    blocks_pos = stream_pos.view(n_blk, span)

    input_ids = blocks[:, :-1].contiguous()
    labels    = blocks_lab[:, 1:].contiguous()
    in_ids    = blocks_ids[:, :-1].contiguous()
    position_ids = blocks_pos[:, :-1].contiguous()

    loss_mask = labels != -100

    attn_mask = None
    if return_attn_mask:
        same_seq = in_ids.unsqueeze(-1) == in_ids.unsqueeze(1)
        not_pad  = (in_ids.unsqueeze(-1) != -1) & (in_ids.unsqueeze(1) != -1)
        attn_mask = same_seq & not_pad

    return PackedBatch(
        input_ids,
        labels,
        loss_mask,
        attn_mask,
        position_ids,
        sources,
        episode_successes,
    )


class PackedCollator:
    """DataLoader collate_fn that packs a batch of sequences into blocks."""

    def __init__(
        self,
        block_size:       int,
        pad_token_id:     int = 0,
        sep_token_id:     Optional[int] = None,
        drop_last:        bool = True,
        return_attn_mask: bool = False,
    ) -> None:
        self.block_size       = block_size
        self.pad_token_id     = pad_token_id
        self.sep_token_id     = sep_token_id
        self.drop_last        = drop_last
        self.return_attn_mask = return_attn_mask

    def __call__(self, batch) -> PackedBatch:
        seqs:              List[torch.Tensor] = []
        labels:            List[torch.Tensor] = []
        sources:           List[int]          = []
        episode_successes: List[float]        = []

        for item in batch:
            if isinstance(item, dict):
                s = item["input_ids"]
                l = item.get("labels", s)
                if "src" in item:
                    v = item["src"]
                    sources.append(int(v.item() if torch.is_tensor(v) else v))
                if "episode_success" in item:
                    v = item["episode_success"]
                    episode_successes.append(float(v.item() if torch.is_tensor(v) else v))
            else:
                s, l = item, item

            if not torch.is_tensor(s):
                s = torch.tensor(s, dtype=torch.long)
            if not torch.is_tensor(l):
                l = torch.tensor(l, dtype=torch.long)
            seqs.append(s.long())
            labels.append(l.long())

        return pack_to_blocks(
            sequences        = seqs,
            label_sequences  = labels,
            block_size       = self.block_size,
            pad_token_id     = self.pad_token_id,
            sep_token_id     = self.sep_token_id,
            drop_last        = self.drop_last,
            return_attn_mask = self.return_attn_mask,
            sources          = sources or None,
            episode_successes = episode_successes or None,
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_ce_loss(
    logits:    torch.Tensor,
    labels:    torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss over positions where loss_mask is True.

    Returns zero if the batch is empty (B=0) or no mask positions are active.
    """
    if logits.shape[0] == 0:
        return logits.sum() * 0.0   # zero, but keeps grad_fn
    B, T, V = logits.shape
    per_tok  = F.cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T), reduction="none")
    per_tok  = per_tok.view(B, T)
    return per_tok[loss_mask].mean() if loss_mask.any() else per_tok.mean()
