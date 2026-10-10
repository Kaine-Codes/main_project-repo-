"""
eeg_tcn.py
==========
Temporal Convolutional Network (TCN) for 4-state EEG classification
(Interictal / Preictal / Ictal / Postictal), built for edge deployment
on a Raspberry Pi 5.

Architecture:
    - Configurable N Residual Blocks (default 6), each using Depthwise
      Separable Causal Dilated Convolutions (DSC) to keep the parameter
      count (and therefore inference latency) low.
    - Default dilation schedule: 1, 2, 4, 8, 16, 32 (receptive field = 253).
    - Global average pooling + a single Linear layer maps the final
      temporal features to the 4 clinical states.

Data:
    - Expects preprocessed .npz files such as chb10_train.npz /
      chb08_test.npz, each containing:
        X : float array, shape (num_windows, sequence_length, 25)
        y : int array,   shape (num_windows,)   values in {0,1,2,3}
    - The loader also tolerates a few common alternate key names
      ("features"/"labels", "data"/"target") so you don't have to
      re-save your files if they were exported with different keys.

Usage:
    python eeg_tcn.py \
        --train_npz chb10_train.npz \
        --val_npz   chb08_test.npz \
        --epochs 50 --batch_size 64 --lr 1e-3 \
        --out_path eeg_tcn_chb10.pth

Author note on Softmax + CrossEntropyLoss:
    The spec asks for an FC layer mapped to 4 classes "using a Softmax
    activation." nn.CrossEntropyLoss internally applies log-softmax to
    raw logits, so feeding it already-softmaxed probabilities would
    apply softmax twice, flattening gradients and hurting convergence.
    To respect both requirements correctly, the model exposes:
        - forward(x)      -> raw logits (used for training/loss)
        - predict_proba(x)-> softmax probabilities (used for inference/
                              deployment, where you actually want a
                              probability distribution over the 4 states)
    This is the numerically correct way to get a "softmax output layer"
    without breaking Cross-Entropy training.
"""

import argparse
import glob
import json
import os
import pickle
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                              balanced_accuracy_score, f1_score,
                              precision_recall_fscore_support,
                              accuracy_score)


# --------------------------------------------------------------------------- #
# 0. Experiment numbering
# --------------------------------------------------------------------------- #

def get_next_experiment_id(results_dir: str = "results") -> str:
    """Scan results_dir for existing exp_XXX directories and return the next ID."""
    os.makedirs(results_dir, exist_ok=True)
    existing = glob.glob(os.path.join(results_dir, "exp_*"))
    max_id = 0
    for path in existing:
        basename = os.path.basename(path)
        m = re.match(r'exp_(\d+)', basename)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return f"exp_{max_id + 1:03d}"


def create_experiment_dir(results_dir: str = "results") -> str:
    """Create and return the next experiment directory. Aborts if it exists."""
    exp_id = get_next_experiment_id(results_dir)
    exp_dir = os.path.join(results_dir, exp_id)
    if os.path.exists(exp_dir):
        raise RuntimeError(
            f"Experiment directory {exp_dir} already exists! "
            f"This should not happen. Aborting to prevent overwrite."
        )
    os.makedirs(exp_dir)
    return exp_id, exp_dir


# --------------------------------------------------------------------------- #
# 1. Dataset / DataLoader
# --------------------------------------------------------------------------- #

class EEGWindowDataset(Dataset):
    """
    Loads a preprocessed .npz file of EEG features.

    Handles TWO on-disk formats:
        - 3D: X = (N, seq_len, F), y = (N,)   -- already windowed
        - 2D: X = (N, F), y = (N,)             -- flat feature vectors from
              preprocess_chbmit.py.  In this case the dataset creates sliding
              windows of `seq_len` consecutive feature vectors automatically,
              producing (N - seq_len + 1, seq_len, F).  The label for each
              window is taken from the LAST timestep in the window (i.e. we
              only predict after seeing the full context).

    F = number of features per timestep (25 for feature sets A/B, 30 for C/D).
    Internally X is permuted to (N, F, seq_len) because Conv1d expects
    (batch, channels, length).
    """

    _X_KEYS = ("X", "x", "features", "data", "windows")
    _Y_KEYS = ("y", "Y", "labels", "target")

    def __init__(self, npz_path: str, seq_len: int = 64, expected_features: int = None):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        archive = np.load(npz_path)
        X = self._find_array(archive, self._X_KEYS, npz_path, "features")
        y = self._find_array(archive, self._Y_KEYS, npz_path, "labels")

        # Auto-detect feature count from data if not specified
        actual_features = X.shape[-1]
        if expected_features is not None and actual_features != expected_features:
            raise ValueError(
                f"Expected {expected_features} features per timestep, "
                f"got {actual_features} in {npz_path}"
            )
        expected_features = actual_features

        # --- Handle 2D (flat) vs 3D (already windowed) input ----------------
        if X.ndim == 2:
            # Flat feature vectors: (N, F) -> create sliding windows
            if len(X) < seq_len:
                raise ValueError(
                    f"Not enough samples ({len(X)}) to form even one window "
                    f"of length seq_len={seq_len} in {npz_path}"
                )
            print(
                f"  [EEGWindowDataset] 2D input {X.shape} detected in {npz_path}; "
                f"creating sliding windows with seq_len={seq_len} ..."
            )
            n_windows = len(X) - seq_len + 1
            # Build (n_windows, seq_len, F) via stride tricks for efficiency
            X = np.lib.stride_tricks.sliding_window_view(X, window_shape=seq_len, axis=0)
            # sliding_window_view gives (n_windows, F, seq_len), transpose to (n_windows, seq_len, F)
            X = np.moveaxis(X, -1, 1).copy()  # copy to make contiguous
            # Use the label of the LAST timestep in each window
            y = y[seq_len - 1:]
            print(
                f"  [EEGWindowDataset] Created {X.shape[0]} windows of shape "
                f"({seq_len}, {expected_features})"
            )
        elif X.ndim == 3:
            # Already windowed: (N, seq_len, F)
            pass
        else:
            raise ValueError(
                f"Expected X of shape (N, F) or (N, seq_len, F), "
                f"got shape {X.shape} in {npz_path}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch in {npz_path}: {len(X)} vs {len(y)}"
            )

        # (N, seq_len, C) -> (N, C, seq_len) for Conv1d
        self.X = torch.from_numpy(X).float().permute(0, 2, 1).contiguous()
        self.y = torch.from_numpy(y).long()

    @staticmethod
    def _find_array(archive, candidate_keys, path, label):
        for key in candidate_keys:
            if key in archive.files:
                return archive[key]
        raise KeyError(
            f"Could not find {label} array in {path}. "
            f"Tried keys {candidate_keys}, found {archive.files}."
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloaders(
    train_npz: str,
    val_npz: str,
    batch_size: int = 64,
    num_workers: int = 2,
    seq_len: int = 64,
    feature_scaling: str = "none",
    oversampling: str = "none",
    preictal_oversample: int = 2,
    ictal_oversample: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = EEGWindowDataset(train_npz, seq_len=seq_len)
    val_ds = EEGWindowDataset(val_npz, seq_len=seq_len)

    # --- Feature Scaling (RobustScaler) ---
    if feature_scaling == "robust":
        from sklearn.preprocessing import RobustScaler
        # train_ds.X shape: (N, C, T) — scale along the feature (C) dimension
        # Reshape to (N*T, C), fit, transform, reshape back
        N_tr, C, T = train_ds.X.shape
        X_tr_flat = train_ds.X.permute(0, 2, 1).reshape(-1, C).numpy()  # (N*T, C)
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_flat)
        train_ds.X = torch.from_numpy(
            X_tr_scaled.reshape(N_tr, T, C)
        ).float().permute(0, 2, 1).contiguous()

        N_v, C_v, T_v = val_ds.X.shape
        X_v_flat = val_ds.X.permute(0, 2, 1).reshape(-1, C_v).numpy()
        X_v_scaled = scaler.transform(X_v_flat)
        val_ds.X = torch.from_numpy(
            X_v_scaled.reshape(N_v, T_v, C_v)
        ).float().permute(0, 2, 1).contiguous()

        print(f"  [RobustScaler] Fitted on {N_tr*T} train samples, "
              f"applied to {N_tr} train + {N_v} val windows")

    # --- Seizure-focused oversampling (training only) ---
    if oversampling == "seizure":
        y_np = train_ds.y.numpy()
        preictal_mask = (y_np == 1)
        ictal_mask = (y_np == 2)
        n_pre = preictal_mask.sum()
        n_ict = ictal_mask.sum()

        extra_X, extra_y = [], []
        if n_pre > 0 and preictal_oversample > 1:
            for _ in range(preictal_oversample - 1):
                extra_X.append(train_ds.X[preictal_mask])
                extra_y.append(train_ds.y[preictal_mask])
        if n_ict > 0 and ictal_oversample > 1:
            for _ in range(ictal_oversample - 1):
                extra_X.append(train_ds.X[ictal_mask])
                extra_y.append(train_ds.y[ictal_mask])

        if extra_X:
            train_ds.X = torch.cat([train_ds.X] + extra_X, dim=0)
            train_ds.y = torch.cat([train_ds.y] + extra_y, dim=0)
            added_pre = n_pre * (preictal_oversample - 1)
            added_ict = n_ict * (ictal_oversample - 1)
            print(f"  [Oversampling] Added {added_pre} Preictal "
                  f"({preictal_oversample}x) + {added_ict} Ictal "
                  f"({ictal_oversample}x) duplicate samples")
            new_counts = np.bincount(train_ds.y.numpy(), minlength=4)
            print(f"  [Oversampling] New training distribution: "
                  f"{dict(enumerate(new_counts.tolist()))}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 2. Model: Depthwise-Separable Causal TCN
# --------------------------------------------------------------------------- #

class Chomp1d(nn.Module):
    """Removes the extra right-padding introduced by causal convolution
    so the output length matches the input length exactly."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution = depthwise (per-channel) conv
    followed by a pointwise (1x1) conv that mixes channels.
    Cuts parameter count roughly by a factor of `in_channels` compared
    to a standard Conv1d, which is the key lever for edge deployment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        # causal padding: pad only on the left in effect, achieved here
        # by padding both sides by (kernel_size-1)*dilation then chomping
        # the right side off (standard causal-TCN trick).
        padding = (kernel_size - 1) * dilation

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,   # depthwise: one filter per input channel
            bias=False,
        )
        self.chomp = Chomp1d(padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.chomp(x)
        x = self.pointwise(x)
        return x


class TemporalResidualBlock(nn.Module):
    """
    One residual block of the TCN:
        DSC -> ReLU -> Dropout -> DSC -> ReLU -> Dropout -> (+ residual)
    A 1x1 conv projects the residual path when channel counts differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class SelfAttention1d(nn.Module):
    """
    Lightweight multi-head self-attention over the temporal dimension.

    Takes (B, C, T) from the TCN output, applies multi-head attention
    across the T dimension, and returns (B, C, T) with global temporal
    context mixed in.  A residual connection + layer norm keeps gradients
    stable and lets the model fall back to pure-TCN behavior if attention
    doesn't help.

    With T=64 and num_heads=2, this adds ~2-5K parameters and negligible
    inference latency (O(T^2) where T=64 -> 4096 dot products per head).
    """

    def __init__(self, channels: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        # nn.MultiheadAttention expects (T, B, C) or batch_first=(B, T, C)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T)"""
        # Transpose to (B, T, C) for attention
        x_t = x.permute(0, 2, 1)  # (B, T, C)

        # Self-attention with residual connection
        attn_out, _ = self.attn(x_t, x_t, x_t)  # (B, T, C)
        x_t = self.norm(x_t + self.dropout(attn_out))  # residual + norm

        # Back to (B, C, T)
        return x_t.permute(0, 2, 1)


class EEG_TCN(nn.Module):
    """
    Depthwise-Separable causal TCN for 4-state EEG classification.

    Default: 4 blocks with dilation [1,2,4,8] -> receptive field = 61
    (closely matching the default 64-step input sequence).

    Input:  (batch, 25, seq_len)   -- 25 = 5 time-domain features x 5 channels
    Output: forward()        -> (batch, 4) raw logits (use with CrossEntropyLoss)
            predict_proba()  -> (batch, 4) softmax probabilities (use for inference)
    """

    NUM_CLASSES = 4
    DEFAULT_IN_FEATURES = 25

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_FEATURES,
        num_classes: int = NUM_CLASSES,
        channel_sizes: Tuple = (32, 32, 48, 48),
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilation_schedule: Optional[List[int]] = None,
        use_attention: bool = False,
        attention_heads: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        num_blocks = len(channel_sizes)

        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(num_blocks)]
        assert len(dilation_schedule) == num_blocks, (
            f"dilation_schedule length ({len(dilation_schedule)}) must match "
            f"channel_sizes length ({num_blocks})"
        )

        self.kernel_size = kernel_size
        self.dilation_schedule = dilation_schedule
        self.channel_sizes = channel_sizes

        layers = []
        prev_channels = in_channels
        for i, out_channels in enumerate(channel_sizes):
            dilation = dilation_schedule[i]
            layers.append(
                TemporalResidualBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            prev_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        # Optional self-attention layer between TCN and pooling
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention1d(
                channels=prev_channels,
                num_heads=attention_heads,
                dropout=dropout,
            )

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, T) -> (B, C, 1)
        self.classifier = nn.Linear(prev_channels, num_classes)

        # Print architecture summary
        self._print_architecture_summary()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, num_classes)."""
        y = self.tcn(x)                         # (B, C, T)
        if self.use_attention:
            y = self.attention(y)                # (B, C, T) with global context
        y = self.global_pool(y).squeeze(-1)      # (B, C)
        logits = self.classifier(y)              # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities via Softmax — use this at inference
        time / on-device, not during training."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """Calculate the theoretical receptive field in timesteps.
        Each TemporalResidualBlock has 2 causal convolutions with the same
        dilation.  Each conv contributes (kernel_size - 1) * dilation."""
        rf = 1
        for d in self.dilation_schedule:
            rf += 2 * (self.kernel_size - 1) * d  # 2 convs per block
        return rf

    def _print_architecture_summary(self):
        """Print a compact architecture and receptive field report."""
        rf = self.receptive_field()
        print(f"\n{'-'*50}")
        print(f"  TCN Architecture ({len(self.channel_sizes)} blocks"
              f"{' + Self-Attention' if self.use_attention else ''})")
        print(f"{'-'*50}")
        print(f"  Input features:     {self.in_channels}"
              f" ({'custom' if self.in_channels != self.DEFAULT_IN_FEATURES else 'default'})")
        print(f"  {'Block':>5s}  {'Channels':>10s}  {'Dilation':>8s}  "
              f"{'RF contribution':>15s}  {'Cumulative RF':>14s}")
        cum_rf = 1
        for i, (ch, d) in enumerate(zip(self.channel_sizes,
                                         self.dilation_schedule)):
            in_ch = self.in_channels if i == 0 else self.channel_sizes[i-1]
            contrib = 2 * (self.kernel_size - 1) * d
            cum_rf += contrib
            print(f"  {i:>5d}  {in_ch:>4d}->{ch:<4d}  {d:>8d}  "
                  f"{contrib:>15d}  {cum_rf:>14d}")
        if self.use_attention:
            print(f"  {'attn':>5s}  {'---':>10s}  {'---':>8s}  "
                  f"{'global':>15s}  {'inf':>14s}")
        print(f"{'-'*50}")
        print(f"  Kernel size:        {self.kernel_size}")
        print(f"  Receptive field:    {rf} timesteps (conv only)")
        print(f"  Self-attention:     {'Yes' if self.use_attention else 'No'}")
        print(f"  Trainable params:   {self.count_parameters():,}")
        print(f"{'-'*50}\n")


# --------------------------------------------------------------------------- #
# 3. Training / Validation loop
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 2.5 Focal Loss
# --------------------------------------------------------------------------- #

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017), generalized to accept an
    optional per-class alpha weight vector -- functions exactly like
    nn.CrossEntropyLoss's `weight=` argument, applied on top of the focal
    term rather than instead of it.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 down-weights the loss contribution from already-confidently-
    correct predictions, so gradient focuses more on hard/misclassified
    examples. alpha (if given) does per-class reweighting independently of
    gamma -- the two mechanisms stack.

    With alpha=None and gamma=0 this is exactly nn.CrossEntropyLoss().
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (num_classes,) tensor or None; moved to device by caller
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)                                 # (B, C)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        target_probs = target_log_probs.exp()                                     # p_t, (B,)

        focal_term = (1.0 - target_probs).clamp(min=0.0) ** self.gamma
        loss = -focal_term * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
            if self.reduction == "mean":
                # Match nn.CrossEntropyLoss(weight=...) convention: normalize
                # by the sum of applied weights, not the batch size, so that
                # at gamma=0 this is EXACTLY weighted CrossEntropyLoss.
                return loss.sum() / alpha_t.sum().clamp(min=1e-12)
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 10          # early stopping patience in epochs (only used when early_stopping=True)
    out_path: str = "eeg_tcn_best.pth"
    test_npz: str = ""   # optional held-out test set
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    class_weighting: str = "none"   # "inverse_freq", "custom", "dynamic", "focal", or "none"
    custom_weights: str = "0.7,1.5,2.0,1.5"   # used by "custom" AND as alpha for "focal"
    dynamic_monitor_frac: float = 0.10  # fraction of training data held out for dynamic weight monitoring
    focal_gamma: float = 2.0            # focusing parameter, only used when class_weighting == "focal"
    early_stopping: bool = True         # if False, always trains the full `epochs` (patience is ignored)
    lr_scheduler: str = "plateau"       # "plateau" (ReduceLROnPlateau) or "none" (constant LR)
    lr_patience: int = 5                # epochs without improvement before reducing LR
    lr_cooldown: int = 2                # epochs to wait after a reduction before resuming normal operation
    lr_min: float = 1e-5                # lower bound on the learning rate

    # --- New experiment framework fields ---
    seq_len: int = 64                    # sequence length (replaces global args_seq_len)
    selection_metric: str = "macro_f1"   # "macro_f1" or "preictal_f1"
    feature_scaling: str = "none"        # "none" or "robust"
    oversampling: str = "none"           # "none" or "seizure"
    preictal_oversample: int = 2         # multiplier for Preictal oversampling
    ictal_oversample: int = 2            # multiplier for Ictal oversampling
    target_f1: str = "0.65,0.80,0.75,0.60"  # target F1 per class for dynamic weighting
    results_dir: str = "results"         # base directory for experiment results
    seed: int = 42


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch. Returns (loss, accuracy, macro_f1, per_class_f1_dict)."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(X)                 # raw logits -> correct for CE loss
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # Per-class F1 for configurable selection metric
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1, 2, 3], zero_division=0,
    )
    per_class_f1 = {
        "interictal_f1": float(f1_per_class[0]),
        "preictal_f1": float(f1_per_class[1]),
        "ictal_f1": float(f1_per_class[2]),
        "postictal_f1": float(f1_per_class[3]),
    }

    return total_loss / total, correct / total, macro_f1, per_class_f1


def train_model(model: nn.Module, train_loader, val_loader, cfg: TrainConfig,
                exp_dir: str = None, exp_config: dict = None):
    device = torch.device(cfg.device)
    model.to(device)

    # --- Class weighting (toggleable) ---
    # Get all training labels so we can report the distribution either way,
    # and only build class weights if the config asks for them. Stacking
    # inverse-frequency weights on top of an already-undersampled training
    # set tends to massively over-correct the rarest class (Ictal), so this
    # is exposed as a CLI flag (--class_weighting none|inverse_freq) rather
    # than always-on.
    all_y = train_loader.dataset.y.numpy()
    class_counts = np.bincount(all_y, minlength=4)
    print(f"Training class counts: {dict(enumerate(class_counts.tolist()))}")

    # Dynamic weighting objects (only used when class_weighting == 'dynamic')
    dynamic_weights_manager = None
    monitor_loader = None
    actual_train_loader = train_loader  # may be replaced by a subset for dynamic

    if cfg.class_weighting == "inverse_freq":
        total_samples = len(all_y)
        weights = total_samples / (4.0 * class_counts)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Applied inverse-frequency class weights: {weights}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    elif cfg.class_weighting == "custom":
        custom_weights = [float(w) for w in cfg.custom_weights.split(',')]
        if len(custom_weights) != 4:
            raise ValueError(
                f"--custom_weights must have exactly 4 comma-separated values "
                f"(Interictal,Preictal,Ictal,Postictal), got {custom_weights}"
            )
        weight_tensor = torch.tensor(custom_weights, dtype=torch.float32).to(device)
        print(f"Applied custom class weights: {custom_weights}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    elif cfg.class_weighting == "dynamic":
        from dynamic_weights import DynamicClassWeights, DynamicWeightConfig

        # Parse target F1 from config
        target_f1_values = [float(x) for x in cfg.target_f1.split(',')]
        if len(target_f1_values) != 4:
            raise ValueError(
                f"--target_f1 must have exactly 4 comma-separated values, "
                f"got {target_f1_values}"
            )

        # Split training data into train-proper + monitor (for weight adaptation)
        full_dataset = train_loader.dataset
        n_total = len(full_dataset)
        n_monitor = max(1, int(n_total * cfg.dynamic_monitor_frac))
        n_train_proper = n_total - n_monitor

        generator = torch.Generator().manual_seed(cfg.seed)
        train_proper_ds, monitor_ds = torch.utils.data.random_split(
            full_dataset, [n_train_proper, n_monitor], generator=generator,
        )

        print(f"Dynamic weighting: split training data into "
              f"{n_train_proper} train-proper + {n_monitor} monitor samples")
        print(f"  Target F1: {target_f1_values}")

        # Rebuild train loader with the subset
        actual_train_loader = DataLoader(
            train_proper_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

        monitor_loader = DataLoader(
            monitor_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        dw_cfg = DynamicWeightConfig(target_f1=target_f1_values)
        dynamic_weights_manager = DynamicClassWeights(dw_cfg, device=device)
        criterion = dynamic_weights_manager.get_criterion()
    elif cfg.class_weighting == "focal":
        focal_weights = [float(w) for w in cfg.custom_weights.split(',')]
        if len(focal_weights) != 4:
            raise ValueError(
                f"--custom_weights must have exactly 4 comma-separated values "
                f"(Interictal,Preictal,Ictal,Postictal) to use as focal alpha, "
                f"got {focal_weights}"
            )
        alpha_tensor = torch.tensor(focal_weights, dtype=torch.float32).to(device)
        print(f"Applied Focal Loss: alpha={focal_weights}, gamma={cfg.focal_gamma}")
        criterion = FocalLoss(alpha=alpha_tensor, gamma=cfg.focal_gamma)
    elif cfg.class_weighting == "none":
        print("Class weighting disabled (cfg.class_weighting='none') -- "
              "relying on undersampling alone for balance.")
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(
            f"Unknown class_weighting='{cfg.class_weighting}'. "
            f"Expected 'inverse_freq', 'custom', 'dynamic', 'focal', or 'none'."
        )
    # --------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if cfg.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max", 
            factor=0.5, 
            patience=cfg.lr_patience,
            cooldown=cfg.lr_cooldown,
            min_lr=cfg.lr_min,
            threshold=1e-3,
            threshold_mode="abs"
        )
        print(f"LR Scheduler: ReduceLROnPlateau(patience={cfg.lr_patience}, cooldown={cfg.lr_cooldown}, min_lr={cfg.lr_min})")
    elif cfg.lr_scheduler == "none":
        scheduler = None
        print("LR Scheduler: None (Constant LR)")
    else:
        raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")

    # --- Selection metric setup ---
    print(f"Selection metric: {cfg.selection_metric}")
    best_metric_val = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    current_lr = cfg.lr

    print(f"Training on {device} | model params: {model.count_parameters():,}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1, train_pcf1 = run_epoch(
            model, actual_train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1, val_pcf1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        # Select the metric for checkpointing
        if cfg.selection_metric == "preictal_f1":
            current_metric = val_pcf1["preictal_f1"]
        else:
            current_metric = val_f1

        if scheduler is not None:
            scheduler.step(current_metric)
            
            # Check if LR was reduced
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                print(f"  >> LR reduced: {current_lr:.3e} -> {new_lr:.3e} (plateau patience={cfg.lr_patience} exhausted)")
                current_lr = new_lr

        # --- Dynamic weight update (if enabled) ---
        if dynamic_weights_manager is not None:
            criterion = dynamic_weights_manager.maybe_update(
                epoch, model, monitor_loader, device
            )

        dt = time.time() - t0

        metric_label = cfg.selection_metric
        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | {metric_label}={current_metric:.4f} | "
            f"preictal_f1={val_pcf1['preictal_f1']:.4f} ictal_f1={val_pcf1['ictal_f1']:.4f} | "
            f"macro_f1={val_f1:.4f} | LR={current_lr:.3e} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"{dt:.1f}s"
        )

        if current_metric > best_metric_val:
            best_metric_val = current_metric
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "val_preictal_f1": val_pcf1["preictal_f1"],
                    "selection_metric": cfg.selection_metric,
                    "selection_metric_value": current_metric,
                    "channel_sizes": [m.conv1.pointwise.out_channels for m in model.tcn],
                },
                cfg.out_path,
            )
            print(f"  -> new best model saved to {cfg.out_path} ({metric_label}={current_metric:.4f})")
        else:
            epochs_no_improve += 1
            if cfg.early_stopping and epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs).")
                break

    print(f"\nTraining complete. Best {cfg.selection_metric}={best_metric_val:.4f} "
          f"at epoch {best_epoch}. Weights saved to {cfg.out_path}")

    if dynamic_weights_manager is not None:
        plot_dir = exp_dir if exp_dir else "."
        plot_path = os.path.join(plot_dir, "weight_evolution.png")
        dynamic_weights_manager.plot_weight_and_performance(plot_path)

    # --- Final Evaluation ---
    checkpoint = torch.load(cfg.out_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_metrics = _evaluate_partition(model, val_loader, device, "VALIDATION")

    # --- Optional TEST evaluation ---
    test_metrics = None
    if cfg.test_npz and os.path.isfile(cfg.test_npz):
        print("\n" + "="*60)
        print("  FINAL TEST EVALUATION (held-out, untouched data)")
        print("="*60)
        test_ds = EEGWindowDataset(cfg.test_npz, seq_len=cfg.seq_len)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        test_metrics = _evaluate_partition(model, test_loader, device, "TEST")

    # --- Save experiment artifacts ---
    if exp_dir is not None:
        _save_experiment(exp_dir, cfg, model, val_metrics, test_metrics,
                         best_epoch, best_metric_val, exp_config)

    return {
        "best_val_metric": best_metric_val,
        "best_epoch": best_epoch,
        "selection_metric": cfg.selection_metric,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "out_path": cfg.out_path,
    }


# --------------------------------------------------------------------------- #
# 3b. Experiment artifact saving
# --------------------------------------------------------------------------- #

def _save_experiment(exp_dir, cfg, model, val_metrics, test_metrics,
                     best_epoch, best_metric_val, exp_config=None):
    """Save comprehensive experiment artifacts to exp_dir."""

    # config.json
    config_dict = {
        "train_config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                         for k, v in asdict(cfg).items()},
        "best_epoch": best_epoch,
        "best_metric": cfg.selection_metric,
        "best_metric_value": float(best_metric_val),
        "model_params": model.count_parameters(),
    }
    if exp_config:
        config_dict["experiment"] = exp_config

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # metrics.json
    metrics_dict = {"validation": val_metrics, "test": test_metrics}
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # confusion_matrix.json
    cm_dict = {"validation": val_metrics.get("confusion_matrix")}
    if test_metrics:
        cm_dict["test"] = test_metrics.get("confusion_matrix")
    with open(os.path.join(exp_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm_dict, f, indent=2)

    # README.txt
    readme_lines = [
        "=" * 60,
        f"  Experiment: {os.path.basename(exp_dir)}",
        "=" * 60, "",
        f"Selection metric: {cfg.selection_metric}",
        f"Best {cfg.selection_metric}: {best_metric_val:.4f} (epoch {best_epoch})",
        "",
        "--- Validation ---",
        f"  Macro F1:          {val_metrics['macro_f1']:.4f}",
        f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}",
    ]
    for cls in ["preictal", "ictal", "interictal", "postictal"]:
        pc = val_metrics["per_class"][cls]
        readme_lines.append(
            f"  {cls.capitalize():12s}  P={pc['precision']:.3f}  "
            f"R={pc['recall']:.3f}  F1={pc['f1']:.3f}  "
            f"(support={pc['support']}, predicted={pc['predicted']})")
    if test_metrics:
        readme_lines.extend(["", "--- Test ---",
            f"  Macro F1:          {test_metrics['macro_f1']:.4f}",
            f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}"])
        for cls in ["preictal", "ictal", "interictal", "postictal"]:
            pc = test_metrics["per_class"][cls]
            readme_lines.append(
                f"  {cls.capitalize():12s}  P={pc['precision']:.3f}  "
                f"R={pc['recall']:.3f}  F1={pc['f1']:.3f}  "
                f"(support={pc['support']}, predicted={pc['predicted']})")
    readme_lines.extend(["",
        f"Model parameters: {model.count_parameters():,}",
        f"Channel sizes: {[m.conv1.pointwise.out_channels for m in model.tcn]}",
        f"Epochs trained: {best_epoch} (best) / {cfg.epochs} (max)",
        f"Feature scaling: {cfg.feature_scaling}",
        f"Class weighting: {cfg.class_weighting}",
        f"Oversampling: {cfg.oversampling}",
        f"Seed: {cfg.seed}"])
    with open(os.path.join(exp_dir, "README.txt"), "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    print(f"\n  Experiment artifacts saved to {exp_dir}/")
    print(f"    config.json, metrics.json, confusion_matrix.json, README.txt")


# --------------------------------------------------------------------------- #
# 3c. Comprehensive evaluation helper
# --------------------------------------------------------------------------- #

def _evaluate_partition(model, loader, device, name=""):
    """Run comprehensive evaluation on a data loader and print all metrics."""
    target_names = ["Interictal (0)", "Preictal (1)",
                    "Ictal (2)", "Postictal (3)"]
    all_preds, all_targets = [], []

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    print(f"\n--- {name} Evaluation (Best Model) ---")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, target_names=target_names,
        labels=[0, 1, 2, 3], zero_division=0,
    ))

    # Per-class prediction counts
    pred_counts = np.bincount(y_pred, minlength=4)
    true_counts = np.bincount(y_true, minlength=4)
    print("Per-class counts:")
    print(f"  {'Class':>15s}  {'True':>8s}  {'Predicted':>10s}")
    for i in range(4):
        flag = " ⚠ ZERO PREDICTIONS" if pred_counts[i] == 0 else ""
        print(f"  {target_names[i]:>15s}  {true_counts[i]:>8d}  "
              f"{pred_counts[i]:>10d}{flag}")

    # Per-class precision/recall/f1
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0,
    )

    # Balanced accuracy and F1 scores
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # --- Prominent Preictal/Ictal display ---
    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  Preictal  P={precision[1]:.3f}  R={recall[1]:.3f}  F1={f1_per_class[1]:.3f}  \t║")
    print(f"  ║  Ictal     P={precision[2]:.3f}  R={recall[2]:.3f}  F1={f1_per_class[2]:.3f}  \t║")
    print(f"  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Macro F1={macro_f1:.4f}  Bal Acc={bal_acc:.4f}       \t║")
    print(f"  ╚══════════════════════════════════════════════╝")
    print(f"  Weighted F1:       {weighted_f1:.4f}")

    # Zero-prediction alert
    zero_classes = [target_names[i] for i in range(4) if pred_counts[i] == 0]
    if zero_classes:
        print(f"\n  ⚠ WARNING: Model predicted ZERO samples for: "
              f"{', '.join(zero_classes)}")

    print(f"{'-'*50}\n")

    class_keys = ["interictal", "preictal", "ictal", "postictal"]
    per_class = {
        class_keys[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
            "predicted": int(pred_counts[i]),
        }
        for i in range(4)
    }

    return {
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# 4. CLI entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Depthwise-Separable causal TCN for "
                    "4-state EEG classification."
    )
    # Data
    p.add_argument("--train_npz", type=str, required=True,
                   help="Path to training .npz")
    p.add_argument("--val_npz", type=str, required=True,
                   help="Path to validation .npz")
    p.add_argument("--test_npz", type=str, default="",
                   help="Path to held-out test .npz (optional, for final "
                        "evaluation only)")
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience in epochs (default 10). "
                        "Only used when --early_stopping is on.")
    p.add_argument("--early_stopping", type=str, default="on", choices=["on", "off"],
                   help="'on' (default): stop after --patience epochs with no "
                        "val_macro_f1 improvement. 'off': always train the full "
                        "--epochs (best checkpoint is still saved/tracked "
                        "throughout, this only disables the early break).")
    p.add_argument("--class_weighting", type=str, default="none",
                   choices=["inverse_freq", "custom", "dynamic", "focal", "none"],
                   help="'none' (default), 'inverse_freq', 'custom' (weights via "
                        "--custom_weights), 'dynamic' (adaptive weights based on "
                        "per-class F1, see dynamic_weights.py), or 'focal' (focal "
                        "loss with --custom_weights used as alpha and "
                        "--focal_gamma as the focusing parameter)")
    p.add_argument("--custom_weights", type=str, default="0.619,0.593,1.694,1.668",
                   help="Comma-separated 4 weights (Interictal,Preictal,Ictal,"
                        "Postictal). Used when --class_weighting is 'custom' "
                        "(as CE weights) or 'focal' (as alpha). "
                        "e.g. '0.7,1.5,3.0,1.5'")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss focusing parameter (default 2.0). Only used "
                        "when --class_weighting focal. Higher = more focus on "
                        "hard/misclassified examples.")
    p.add_argument("--dynamic_monitor_frac", type=float, default=0.10,
                   help="Fraction of training data held out for dynamic weight "
                        "monitoring (default 0.10). Only used when "
                        "--class_weighting dynamic.")
    # LR Scheduler
    p.add_argument("--lr_scheduler", type=str, default="none", choices=["plateau", "none"],
                   help="'plateau' (default): ReduceLROnPlateau based on val_macro_f1. "
                        "'none': constant LR.")
    p.add_argument("--lr_patience", type=int, default=5,
                   help="Epochs without improvement before reducing LR (default 5).")
    p.add_argument("--lr_cooldown", type=int, default=2,
                   help="Epochs to wait after LR reduction before resuming normal operation (default 2).")
    p.add_argument("--lr_min", type=float, default=1e-5,
                   help="Lower bound on the learning rate (default 1e-5).")
    # Architecture
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--channel_sizes", type=str, default="32,32,48,48,64,64",
                   help="Comma-separated channel counts per block ")
    p.add_argument("--dilation_schedule", type=str, default=None,
                   help="Comma-separated dilation per block. Default: "
                        "exponential (1,2,4,8,...)")
    # I/O
    p.add_argument("--out_path", type=str, default="eeg_tcn_best.pth")
    p.add_argument("--seq_len", type=int, default=64,
                   help="Sequence length (used when .npz is 2D). Default: 64")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # Attention
    p.add_argument("--use_attention", action="store_true", default=False,
                   help="Add a self-attention layer after the TCN blocks "
                        "(before global avg pool). Off by default.")
    p.add_argument("--attention_heads", type=int, default=2,
                   help="Number of attention heads (default 2). Only used "
                        "when --use_attention is set.")
    # Input features
    p.add_argument("--in_channels", type=int, default=None,
                   help="Number of input features (channels). Auto-detected "
                        "from the training .npz if not specified. Use 25 for "
                        "feature sets A/B, 30 for C/D.")
    # --- Experiment framework ---
    p.add_argument("--selection_metric", type=str, default="macro_f1",
                   choices=["macro_f1", "preictal_f1"],
                   help="Metric used for model checkpointing and early stopping. "
                        "'macro_f1' (default) or 'preictal_f1' (prioritise "
                        "early seizure warning).")
    p.add_argument("--feature_scaling", type=str, default="none",
                   choices=["none", "robust"],
                   help="Per-feature scaling: 'none' (default) or 'robust' "
                        "(RobustScaler on training set, applied to val/test).")
    p.add_argument("--oversampling", type=str, default="none",
                   choices=["none", "seizure"],
                   help="Oversampling strategy: 'none' (default) or 'seizure' "
                        "(duplicate Preictal/Ictal samples in training set).")
    p.add_argument("--preictal_oversample", type=int, default=2,
                   help="Preictal oversampling multiplier (default 2). Only "
                        "used when --oversampling seizure.")
    p.add_argument("--ictal_oversample", type=int, default=2,
                   help="Ictal oversampling multiplier (default 2). Only "
                        "used when --oversampling seizure.")
    p.add_argument("--target_f1", type=str, default="0.65,0.80,0.75,0.60",
                   help="Comma-separated target F1 per class for dynamic "
                        "weighting (Interictal,Preictal,Ictal,Postictal). "
                        "Default: 0.65,0.80,0.75,0.60")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Base directory for saving experiment results. "
                        "Each run creates results/exp_XXX/ with config, "
                        "metrics, and model checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Create experiment directory ---
    exp_id, exp_dir = create_experiment_dir(args.results_dir)
    print(f"\n{'='*60}")
    print(f"  Experiment {exp_id}")
    print(f"  Directory: {exp_dir}")
    print(f"{'='*60}")

    # Default out_path into the experiment directory
    if args.out_path == "eeg_tcn_best.pth":
        args.out_path = os.path.join(exp_dir, "best_model.pth")

    # Parse architecture strings
    channel_sizes = tuple(int(x) for x in args.channel_sizes.split(','))
    dilation_schedule = None
    if args.dilation_schedule:
        dilation_schedule = [int(x) for x in args.dilation_schedule.split(',')]

    train_loader, val_loader = build_dataloaders(
        args.train_npz, args.val_npz, batch_size=args.batch_size,
        num_workers=args.num_workers, seq_len=args.seq_len,
        feature_scaling=args.feature_scaling,
        oversampling=args.oversampling,
        preictal_oversample=args.preictal_oversample,
        ictal_oversample=args.ictal_oversample,
    )

    # Auto-detect in_channels from the training data if not specified
    in_channels = args.in_channels
    if in_channels is None:
        in_channels = train_loader.dataset.X.shape[1]  # (N, C, T) -> C
        print(f"Auto-detected in_channels={in_channels} from training data")

    try:
        npz_data = np.load(args.train_npz)
        if 'feature_names' in npz_data:
            feature_names = npz_data['feature_names']
            print(f"Loaded features from NPZ: {list(feature_names)}")
            
            # Print unique channel names by stripping the feature suffix
            ch_names = list(dict.fromkeys([re.sub(r'_[a-z_]+.*$', '', f) for f in feature_names]))
            print(f"Channels used in this dataset ({len(ch_names)}): {ch_names}")
    except Exception as e:
        print(f"Could not load feature names: {e}")

    model = EEG_TCN(
        in_channels=in_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        channel_sizes=channel_sizes,
        dilation_schedule=dilation_schedule,
        use_attention=args.use_attention,
        attention_heads=args.attention_heads,
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        early_stopping=(args.early_stopping == "on"),
        out_path=args.out_path,
        test_npz=args.test_npz,
        class_weighting=args.class_weighting,
        custom_weights=args.custom_weights,
        focal_gamma=args.focal_gamma,
        dynamic_monitor_frac=args.dynamic_monitor_frac,
        lr_scheduler=args.lr_scheduler,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        lr_min=args.lr_min,
        # New experiment framework fields
        seq_len=args.seq_len,
        selection_metric=args.selection_metric,
        feature_scaling=args.feature_scaling,
        oversampling=args.oversampling,
        preictal_oversample=args.preictal_oversample,
        ictal_oversample=args.ictal_oversample,
        target_f1=args.target_f1,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    # Experiment config dict for saving
    exp_config = {
        "exp_id": exp_id,
        "train_npz": args.train_npz,
        "val_npz": args.val_npz,
        "test_npz": args.test_npz,
        "channel_sizes": list(channel_sizes),
        "dilation_schedule": dilation_schedule or "exponential",
        "kernel_size": args.kernel_size,
        "in_channels": in_channels,
        "use_attention": args.use_attention,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
    }

    print(f"\nExperiment config:")
    print(f"  train:              {args.train_npz}")
    print(f"  val:                {args.val_npz}")
    print(f"  test:               {args.test_npz or '(none)'}")
    print(f"  selection_metric:   {args.selection_metric}")
    print(f"  early_stopping:     {args.early_stopping}"
          + (f" (patience={args.patience})" if args.early_stopping == "on" else ""))
    print(f"  lr_scheduler:       {args.lr_scheduler}"
          + (f" (patience={args.lr_patience}, cooldown={args.lr_cooldown}, min={args.lr_min})" if args.lr_scheduler == "plateau" else ""))
    print(f"  class_weighting:    {args.class_weighting}")
    if args.class_weighting == "custom":
        print(f"  custom_weights:     {args.custom_weights}")
    if args.class_weighting == "focal":
        print(f"  focal alpha:        {args.custom_weights}")
        print(f"  focal gamma:        {args.focal_gamma}")
    if args.class_weighting == "dynamic":
        print(f"  monitor_frac:       {args.dynamic_monitor_frac}")
        print(f"  target_f1:          {args.target_f1}")
    print(f"  feature_scaling:    {args.feature_scaling}")
    print(f"  oversampling:       {args.oversampling}")
    if args.oversampling == "seizure":
        print(f"  preictal_oversample: {args.preictal_oversample}x")
        print(f"  ictal_oversample:    {args.ictal_oversample}x")
    print(f"  channel_sizes:      {channel_sizes}")
    print(f"  dilation:           {dilation_schedule or 'exponential'}")
    print(f"  kernel_size:        {args.kernel_size}")
    print(f"  in_channels:        {in_channels}")
    print(f"  use_attention:      {args.use_attention}")
    if args.use_attention:
        print(f"  attention_heads:    {args.attention_heads}")
    print(f"  seq_len:            {args.seq_len}")
    print(f"  seed:               {args.seed}")
    print(f"  out_path:           {args.out_path}")
    print()

    train_model(model, train_loader, val_loader, cfg,
                exp_dir=exp_dir, exp_config=exp_config)


if __name__ == "__main__":
    main()