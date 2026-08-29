"""
dynamic_weights.py
==================
Adaptive per-class loss weighting for multi-class classification.

Instead of manually searching over hundreds of fixed weight combinations,
this module monitors per-class F1 on a held-out slice of *training* data
(NOT validation or test) and multiplicatively adjusts the CrossEntropyLoss
weights so that under-performing classes receive higher weight.

Key design choices
------------------
- **F1 over raw recall**: optimising recall alone lets the model game the
  metric by predicting everything as one class.  F1 penalises that.
- **EMA smoothing**: avoids reacting to noisy single-epoch measurements.
- **Multiplicative updates**: weight_c *= exp(η · error_c) keeps the
  updates proportional to the current weight rather than additive.
- **Max-change clamp**: each update can change a weight by at most
  `max_change_per_update` (default ×1.3 / ÷1.3).
- **Global clipping + normalisation**: weights stay in [w_min, w_max]
  and are re-normalised so their mean ≈ 1.0.
- **Warmup**: no updates until the model has trained for `warmup_epochs`.
- **No gradient leakage**: all weight updates happen inside
  `torch.no_grad()`.
- **No val/test contamination**: monitoring uses training data only.

Usage
-----
See eeg_tcn.py's `--class_weighting dynamic` path for the full
integration.  Standalone:

    from dynamic_weights import DynamicClassWeights, DynamicWeightConfig

    dw_cfg = DynamicWeightConfig()  # conservative defaults
    dw = DynamicClassWeights(dw_cfg, device=torch.device("cuda"))
    criterion = dw.get_criterion()

    for epoch in range(1, n_epochs + 1):
        # ... train one epoch with `criterion` ...
        criterion = dw.maybe_update(epoch, model, monitor_loader, device)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class DynamicWeightConfig:
    """All hyperparameters for dynamic class weighting."""

    num_classes: int = 4

    # --- Targets (per-class F1 the algorithm tries to reach) ---
    # Asymmetric: Ictal > Preictal > Interictal > Postictal
    target_f1: List[float] = field(
        default_factory=lambda: [0.70, 0.75, 0.80, 0.65]
    )

    # --- Update schedule ---
    update_interval: int = 3       # update every N epochs
    warmup_epochs: int = 5         # no updates before this epoch

    # --- Learning rate & smoothing ---
    eta: float = 0.3               # step size for exp(η · error)
    ema_alpha: float = 0.3         # EMA decay: new = α·measured + (1-α)·old

    # --- Safeguards ---
    max_change_per_update: float = 1.3   # max multiplicative change per update
    w_min: float = 0.3             # absolute weight floor
    w_max: float = 5.0             # absolute weight ceiling
    min_samples_per_class: int = 30  # skip update if any class has fewer

    # --- Initial weights ---
    initial_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )

    class_names: List[str] = field(
        default_factory=lambda: ["Interictal", "Preictal", "Ictal", "Postictal"]
    )


# --------------------------------------------------------------------------- #
# Dynamic weight manager
# --------------------------------------------------------------------------- #

class DynamicClassWeights:
    """
    Manages adaptive per-class CrossEntropyLoss weights.

    Call `maybe_update()` after each epoch — it will only actually
    recompute weights every `update_interval` epochs (after warmup).

    The returned criterion always reflects the current weights.
    """

    def __init__(self, cfg: DynamicWeightConfig,
                 device: torch.device = torch.device("cpu")):
        self.cfg = cfg
        self.device = device

        # Current weights (on device, detached — never part of the graph)
        self.weights = torch.tensor(
            cfg.initial_weights, dtype=torch.float32, device=device
        )

        # EMA of per-class F1 — initialised to None (no history yet)
        self._ema_f1: Optional[np.ndarray] = None

        # Track how many updates have been applied
        self._n_updates = 0

        # Build initial criterion
        self._criterion = nn.CrossEntropyLoss(weight=self.weights.clone())

        print(f"\n{'─'*60}")
        print("  Dynamic Class Weighting initialised")
        print(f"{'─'*60}")
        print(f"  Initial weights:      {self._fmt_weights()}")
        print(f"  Target F1:            {cfg.target_f1}")
        print(f"  Update interval:      every {cfg.update_interval} epochs")
        print(f"  Warmup:               {cfg.warmup_epochs} epochs")
        print(f"  η (step size):        {cfg.eta}")
        print(f"  EMA α:                {cfg.ema_alpha}")
        print(f"  Max change/update:    ×{cfg.max_change_per_update:.2f}")
        print(f"  Weight range:         [{cfg.w_min}, {cfg.w_max}]")
        print(f"  Min samples/class:    {cfg.min_samples_per_class}")
        print(f"{'─'*60}\n")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_criterion(self) -> nn.CrossEntropyLoss:
        """Return the current weighted CrossEntropyLoss."""
        return self._criterion

    @torch.no_grad()
    def maybe_update(self, epoch: int, model: nn.Module,
                     monitor_loader, device: torch.device
                     ) -> nn.CrossEntropyLoss:
        """
        Conditionally update weights based on monitoring metrics.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-indexed).
        model : nn.Module
            The model (set to eval mode internally, restored after).
        monitor_loader : DataLoader
            DataLoader over the training-internal monitoring split.
        device : torch.device

        Returns
        -------
        nn.CrossEntropyLoss
            The (possibly updated) criterion.
        """
        # --- Guard: warmup ---
        if epoch < self.cfg.warmup_epochs:
            return self._criterion

        # --- Guard: update interval ---
        if epoch % self.cfg.update_interval != 0:
            return self._criterion

        # --- Compute per-class F1 on the monitor split ---
        f1_per_class, support = self._compute_monitor_metrics(
            model, monitor_loader, device
        )

        if f1_per_class is None:
            return self._criterion  # not enough data

        # --- EMA smoothing ---
        if self._ema_f1 is None:
            self._ema_f1 = f1_per_class.copy()
        else:
            alpha = self.cfg.ema_alpha
            self._ema_f1 = alpha * f1_per_class + (1.0 - alpha) * self._ema_f1

        # --- Multiplicative weight update ---
        target = np.array(self.cfg.target_f1)
        error = target - self._ema_f1  # positive = under-performing

        # Compute raw multipliers: exp(η · error)
        raw_mult = np.exp(self.cfg.eta * error)

        # Clamp per-update change
        max_c = self.cfg.max_change_per_update
        clamped_mult = np.clip(raw_mult, 1.0 / max_c, max_c)

        # Apply to current weights (numpy for arithmetic, then back to torch)
        w = self.weights.cpu().numpy()
        w_new = w * clamped_mult

        # Global clip
        w_new = np.clip(w_new, self.cfg.w_min, self.cfg.w_max)

        # Normalise so mean ≈ 1.0
        w_new = w_new / w_new.mean()

        # Update stored weights
        self.weights = torch.tensor(
            w_new, dtype=torch.float32, device=self.device
        )
        self._criterion = nn.CrossEntropyLoss(weight=self.weights.clone())
        self._n_updates += 1

        # --- Print report ---
        self._print_update_report(epoch, f1_per_class, support,
                                  clamped_mult, w, w_new)

        return self._criterion

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_monitor_metrics(self, model, loader, device):
        """
        Run inference on the monitor split and return per-class F1
        and support.  Returns (None, None) if any class has fewer
        than `min_samples_per_class` true samples.
        """
        was_training = model.training
        model.eval()

        all_preds, all_targets = [], []
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())

        if was_training:
            model.train()

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Check minimum samples
        counts = np.bincount(all_targets, minlength=self.cfg.num_classes)
        if np.any(counts < self.cfg.min_samples_per_class):
            low = {self.cfg.class_names[i]: int(counts[i])
                   for i in range(self.cfg.num_classes)
                   if counts[i] < self.cfg.min_samples_per_class}
            print(f"  [DynWeights] Skipping update: insufficient monitor "
                  f"samples for {low}")
            return None, None

        _, _, f1_per_class, support = precision_recall_fscore_support(
            all_targets, all_preds,
            labels=list(range(self.cfg.num_classes)),
            zero_division=0,
        )

        return f1_per_class, support

    def _print_update_report(self, epoch, measured_f1, support,
                              multipliers, old_w, new_w):
        """Print a concise report after each weight update."""
        print(f"\n{'─'*60}")
        print(f"  Dynamic Weight Update #{self._n_updates} (epoch {epoch})")
        print(f"{'─'*60}")

        header = (f"  {'Class':<12s} {'EMA_F1':>7s} {'Target':>7s} "
                  f"{'Error':>7s} {'Mult':>6s} {'Old_W':>6s} {'New_W':>6s} "
                  f"{'Support':>8s}")
        print(header)

        target = np.array(self.cfg.target_f1)
        for i in range(self.cfg.num_classes):
            err = target[i] - self._ema_f1[i]
            print(f"  {self.cfg.class_names[i]:<12s} "
                  f"{self._ema_f1[i]:7.4f} {target[i]:7.2f} "
                  f"{err:+7.4f} {multipliers[i]:6.3f} "
                  f"{old_w[i]:6.3f} {new_w[i]:6.3f} "
                  f"{int(support[i]):8d}")

        print(f"  Weight mean: {new_w.mean():.4f}  "
              f"(should be ≈1.0 after normalisation)")
        print(f"{'─'*60}\n")

    def _fmt_weights(self) -> str:
        """Format current weights as a readable string."""
        return "[" + ", ".join(f"{w:.4f}" for w in self.weights.tolist()) + "]"