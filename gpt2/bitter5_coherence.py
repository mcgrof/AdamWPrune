"""
bitter5_coherence.py

Summary
-------
This module defines a pruning score called **bitter5_coherence** for the AdamWPrune project.
It extends the "bitter3" idea (|w| * sqrt(|m|)) by adding temporal-coherence features:

1) Sign stability (few sign flips in parameter updates).
2) Directional consistency of successive updates (cosine similarity of Δw_t and Δw_{t-1}).
3) Momentum–weight alignment (cosine similarity of Adam m and current w).
4) Jitter penalty (variance of update magnitudes).

Intuition
---------
Keep parameters that are:
- strong (|w|), active (|m|),
- stable in direction (few sign flips),
- coherent over time (successive updates point roughly the same way),
- aligned with their optimizer memory (m ~ w),
- and low-jitter (updates are not wildly varying).

Score (keep-high, prune-low)
----------------------------
Let:
  w  := parameter tensor
  m  := Adam first moment (exp_avg) for w (same shape)
  Δw := step-to-step parameter update (w_t - w_{t-1})

We compute a per-parameter score S roughly as:

  S = alpha * |w| * sqrt(|m| + eps)
      * (1 - flip_rate_ema)
      * cos_dir_ema
      * cos_mw
      / (1 + jitter_ema)

Where:
- flip_rate_ema      \in [0,1]: EMA of update sign flips (lower is better)
- cos_dir_ema        \in [-1,1]: EMA of cos(Δw_t, Δw_{t-1}) (higher is better)
- cos_mw             \in [-1,1]: cos(m, w) at the current step (higher is better)
- jitter_ema         >= 0      : EMA of update magnitude variance (lower is better)

Implementation Notes
--------------------
- We maintain a lightweight "tracker" state per parameter tensor:
    prev_w,  ema_dw, ema_dw2, ema_cos_dir, ema_flip_rate
  These are stored in a side dict keyed by the param object id, so no nn.Module edits.
- We rely on Adam/AdamW to supply exp_avg (first moment) via optimizer.state[param]['exp_avg'].
- For efficiency, we do not store full histories; only EMAs and prev_w.

Integration Points
------------------
1) Call `tracker.update(param)` **after** optimizer.step() each iteration to feed Δw.
2) When it’s time to prune, call `bitter5_coherence_scores(params, optimizer, tracker, cfg)`
   to get a concatenated score tensor you can percentile-cut for global sparsity.
3) Apply masks the same way AdamWPrune does today.

This is a *template*; adjust hyperparameters and numerical stabilizers as needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional
import math
import torch
import torch.nn as nn

# ---------------------------
# Config & small utilities
# ---------------------------

@dataclass
class Bitter5Config:
    alpha: float = 1.0        # weight on |w| * sqrt(|m|)
    beta_flip: float = 1.0    # weight multiplier inside (1 - flip_rate_ema) ** beta_flip
    beta_dir: float = 1.0     # weight multiplier on cos_dir_ema ** beta_dir
    beta_mw: float = 1.0      # weight multiplier on cos_mw ** beta_mw
    beta_jitter: float = 1.0  # weight multiplier on 1/(1 + beta_jitter * jitter_ema)
    # EMA coefficients
    ema_dw: float = 0.9       # EMA for Δw vector
    ema_stats: float = 0.9    # EMA for scalar stats (flip rate, cos dir, jitter)
    # numerical epsilons
    eps_m: float = 1e-12
    eps_norm: float = 1e-12
    # which tensors to score
    include_bias: bool = False
    include_norm: bool = False  # e.g., LayerNorm/GN/BN
    # reduce mode for tensor -> scalar modifiers (sign flips, cos, jitter)
    reduce: str = "mean"  # {"mean","median"}

def _reduce(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    if mode == "median":
        return x.flatten().median()
    return x.flatten().mean()

def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    # elementwise cosine between vectors a and b (same shape)
    num = a * b
    den = (a.abs() + eps) * (b.abs() + eps)  # L1-based surrogate to be stable per-weight
    cos_elem = num / den
    return cos_elem.clamp(-1.0, 1.0)

# ---------------------------
# Coherence Tracker
# ---------------------------

class CoherenceTracker:
    """
    Tracks lightweight temporal stats per-parameter:
      - prev_w: last parameter value (to compute Δw)
      - ema_dw: EMA of Δw (vector)
      - ema_dw2: EMA of (Δw^2) to estimate variance
      - ema_cos_dir: EMA of cos(Δw_t, Δw_{t-1}) (reduced scalar)
      - ema_flip_rate: EMA of sign-flip indicator for Δw
      - jitter_ema: EMA of variance proxy for |Δw| over time

    Usage:
      tracker = CoherenceTracker(cfg)
      # after optimizer.step():
      for p in model.parameters(): tracker.update(p)
    """
    def __init__(self, cfg: Bitter5Config):
        self.cfg = cfg
        self._state: Dict[int, Dict[str, Any]] = {}

    def reset(self):
        self._state.clear()

    def update(self, p: nn.Parameter):
        if p.grad is None:
            return
        pid = id(p)
        s = self._state.get(pid)
        with torch.no_grad():
            w = p.data
            if s is None:
                self._state[pid] = {
                    "prev_w": w.clone(),
                    "ema_dw": torch.zeros_like(w),
                    "ema_dw2": torch.zeros_like(w),
                    "ema_cos_dir": torch.tensor(0.0, device=w.device),
                    "ema_flip_rate": torch.tensor(0.0, device=w.device),
                    "jitter_ema": torch.tensor(0.0, device=w.device),
                    "inited": False,
                }
                return

            dw = w - s["prev_w"]                # Δw_t
            prev_ema_dw = s["ema_dw"]           # proxy for Δw_{t-1}
            # EMAs for vector stats
            s["ema_dw"].mul_(self.cfg.ema_dw).add_(dw, alpha=1.0 - self.cfg.ema_dw)
            s["ema_dw2"].mul_(self.cfg.ema_dw).add_(dw * dw, alpha=1.0 - self.cfg.ema_dw)

            # Directional consistency: cos(Δw_t, Δw_{t-1})
            cos_dir_elem = _safe_cos(dw, prev_ema_dw, eps=self.cfg.eps_norm)
            cos_dir = _reduce(cos_dir_elem, self.cfg.reduce)
            s["ema_cos_dir"] = s["ema_cos_dir"] * self.cfg.ema_stats + cos_dir * (1 - self.cfg.ema_stats)

            # Sign flip rate (per-element)
            flip = (torch.sign(dw) != torch.sign(prev_ema_dw)).float()
            flip_rate = _reduce(flip, self.cfg.reduce)
            s["ema_flip_rate"] = s["ema_flip_rate"] * self.cfg.ema_stats + flip_rate * (1 - self.cfg.ema_stats)

            # Jitter: variance proxy for |Δw|
            abs_dw = dw.abs()
            mean_abs = _reduce(abs_dw, self.cfg.reduce)
            mean_abs2 = _reduce(abs_dw * abs_dw, self.cfg.reduce)
            var_proxy = torch.clamp(mean_abs2 - mean_abs * mean_abs, min=0.0)
            s["jitter_ema"] = s["jitter_ema"] * self.cfg.ema_stats + var_proxy * (1 - self.cfg.ema_stats)

            # finalize
            s["prev_w"].copy_(w)
            s["inited"] = True

    def get_state(self, p: nn.Parameter) -> Optional[Dict[str, Any]]:
        return self._state.get(id(p), None)

# ---------------------------
# Bitter5 score computation
# ---------------------------

def bitter5_coherence_scores(params: Iterable[nn.Parameter],
                             optimizer: torch.optim.Optimizer,
                             tracker: CoherenceTracker,
                             cfg: Bitter5Config) -> torch.Tensor:
    """
    Compute bitter5 scores for all provided params (concatenated).
    Requires optimizer.state[p]['exp_avg'] (Adam/AdamW first moment).

    Returns
    -------
    scores_flat : torch.Tensor
        1-D tensor of per-weight scores (keep-high).
    """
    score_chunks = []
    for p in params:
        if p.grad is None:
            continue
        # Filter param types if desired
        is_bias = p.ndim == 1
        is_norm = hasattr(p, "is_norm") or ("norm" in (getattr(p, "name", "") or "").lower())
        if (is_bias and not cfg.include_bias) or (is_norm and not cfg.include_norm):
            continue

        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", None)
        if m is None:
            # Fallback: if no Adam state yet, treat m as zeros
            m = torch.zeros_like(p.data)

        s_track = tracker.get_state(p)
        # Base bitter3-like core: |w| * sqrt(|m|)
        base = p.data.abs() * torch.sqrt(m.abs() + cfg.eps_m)

        # If we lack history, fall back to base only
        if not s_track or not s_track.get("inited", False):
            score_chunks.append(cfg.alpha * base.flatten())
            continue

        # Coherence modifiers
        flip_rate = s_track["ema_flip_rate"].clamp(0.0, 1.0)
        flip_mod  = (1.0 - flip_rate).pow(cfg.beta_flip)  # higher if fewer flips

        cos_dir = s_track["ema_cos_dir"].clamp(-1.0, 1.0)
        cos_dir_mod = torch.pow(torch.clamp(cos_dir, min=0.0), cfg.beta_dir)  # clamp negatives to 0

        # Momentum–weight alignment (elementwise cosine surrogate)
        cos_mw_elem = _safe_cos(m, p.data, eps=cfg.eps_norm)
        cos_mw = _reduce(cos_mw_elem, cfg.reduce).clamp(-1.0, 1.0)
        cos_mw_mod = torch.pow(torch.clamp(cos_mw, min=0.0), cfg.beta_mw)

        jitter = torch.clamp(s_track["jitter_ema"], min=0.0)
        jitter_mod = 1.0 / (1.0 + cfg.beta_jitter * jitter + 1e-12)

        # Combine
        # Broadcast scalar modifiers to tensor shape
        mod = flip_mod * cos_dir_mod * cos_mw_mod * jitter_mod
        mod = mod.to(p.data.device).reshape(1)  # scalars
        score = cfg.alpha * base * mod
        score_chunks.append(score.flatten())

    if not score_chunks:
        return torch.tensor([], dtype=torch.float32)

    return torch.cat(score_chunks, dim=0)


# ---------------------------
# Example integration sketch
# ---------------------------

def example_training_loop(model: nn.Module,
                          opt: torch.optim.AdamW,
                          data_loader,
                          prune_every_steps: int = 1000,
                          target_sparsity: float = 0.5):
    """
    Sketch only: shows where to plug the tracker and scorer.
    Replace pruning_impl() with your AdamWPrune masking utilities.
    """
    cfg = Bitter5Config()
    tracker = CoherenceTracker(cfg)

    step = 0
    for xb, yb in data_loader:
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()

        # update coherence stats AFTER params changed
        for p in model.parameters():
            tracker.update(p)

        step += 1
        if step % prune_every_steps == 0:
            params = [p for p in model.parameters() if p.grad is not None]
            scores = bitter5_coherence_scores(params, opt, tracker, cfg)

            # global threshold for desired sparsity
            k = int(scores.numel() * target_sparsity)
            if k > 0:
                thresh = torch.topk(scores, k, largest=False).values.max()
                # build mask per param using same score logic (recompute per-tensor, compare to thresh)
                # In real code, integrate with your existing pruning mask infra.
                apply_global_pruning_masks(params, opt, tracker, cfg, thresh)


def apply_global_pruning_masks(params, opt, tracker, cfg, threshold):
    """
    Minimal illustration: build and apply binary masks based on bitter5 scores.
    Replace with AdamWPrune's existing mask pipeline.
    """
    idx = 0
    for p in params:
        if p.grad is None:
            continue
        # recompute per-tensor scores
        s = bitter5_coherence_scores([p], opt, tracker, cfg)
        if s.numel() == 0:
            continue
        mask = (s > threshold).reshape_as(p.data).to(p.data.dtype)
        p.data.mul_(mask)  # hard prune-in-place (illustrative)
        # You likely want persistent masks and to zero-out grads, etc.


# ---------------------------
# Quick test stub (optional)
# ---------------------------

if __name__ == "__main__":
    # Tiny smoke test with a linear layer
    torch.manual_seed(0)
    lin = nn.Linear(16, 16)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)

    # Fake single batch loop
    tracker = CoherenceTracker(Bitter5Config())
    for _ in range(5):
        x = torch.randn(8, 16)
        y = torch.randint(0, 16, (8,))
        opt.zero_grad(set_to_none=True)
        out = lin(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
        for p in lin.parameters():
            tracker.update(p)

    scores = bitter5_coherence_scores(lin.parameters(), opt, tracker, Bitter5Config())
    print("bitter5_coherence score shape:", scores.shape, "min/max:", scores.min().item(), scores.max().item())
