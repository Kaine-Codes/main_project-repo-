"""
eeg_tcn.py
==========
Temporal Convolutional Network (TCN) for 4-state EEG classification
(Interictal / Preictal / Ictal / Postictal), built for edge deployment
on a Raspberry Pi 5.

Architecture:
    - Configurable N Residual Blocks (default 4), each using Depthwise
      Separable Causal Dilated Convolutions (DSC) to keep the parameter
      count (and therefore inference latency) low.
    - Default dilation schedule: 1, 2, 4, 8 (receptive field = 61,
      closely matching the default 64-step input).
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
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                              balanced_accuracy_score, f1_score,
                              precision_recall_fscore_support)


# --------------------------------------------------------------------------- #
# 1. Dataset / DataLoader
# --------------------------------------------------------------------------- #

class EEGWindowDataset(Dataset):
    """
    Loads a preprocessed .npz file of EEG features.

    Handles TWO on-disk formats:
        - 3D: X = (N, seq_len, 25), y = (N,)   -- already windowed
        - 2D: X = (N, 25), y = (N,)             -- flat feature vectors from
              preprocess_chbmit.py.  In this case the dataset creates sliding
              windows of `seq_len` consecutive feature vectors automatically,
              producing (N - seq_len + 1, seq_len, 25).  The label for each
              window is taken from the LAST timestep in the window (i.e. we
              only predict after seeing the full context).

    Internally X is permuted to (N, 25, seq_len) because Conv1d expects
    (batch, channels, length).
    """

    _X_KEYS = ("X", "x", "features", "data", "windows")
    _Y_KEYS = ("y", "Y", "labels", "target")

    def __init__(self, npz_path: str, seq_len: int = 64, expected_features: int = 25):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        archive = np.load(npz_path)
        X = self._find_array(archive, self._X_KEYS, npz_path, "features")
        y = self._find_array(archive, self._Y_KEYS, npz_path, "labels")

        # --- Handle 2D (flat) vs 3D (already windowed) input ----------------
        if X.ndim == 2:
            # Flat feature vectors: (N, 25) -> create sliding windows
            if X.shape[-1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features per timestep, "
                    f"got {X.shape[-1]} in {npz_path}"
                )
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
            # Build (n_windows, seq_len, 25) via stride tricks for efficiency
            X = np.lib.stride_tricks.sliding_window_view(X, window_shape=seq_len, axis=0)
            # sliding_window_view gives (n_windows, 25, seq_len), transpose to (n_windows, seq_len, 25)
            X = np.moveaxis(X, -1, 1).copy()  # copy to make contiguous
            # Use the label of the LAST timestep in each window
            y = y[seq_len - 1:]
            print(
                f"  [EEGWindowDataset] Created {X.shape[0]} windows of shape "
                f"({seq_len}, {expected_features})"
            )
        elif X.ndim == 3:
            # Already windowed: (N, seq_len, 25)
            if X.shape[-1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features per timestep, "
                    f"got {X.shape[-1]} in {npz_path}"
                )
        else:
            raise ValueError(
                f"Expected X of shape (N, 25) or (N, seq_len, {expected_features}), "
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
) -> Tuple[DataLoader, DataLoader]:
    train_ds = EEGWindowDataset(train_npz, seq_len=seq_len)
    val_ds = EEGWindowDataset(val_npz, seq_len=seq_len)

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


class EEG_TCN(nn.Module):
    """
    Depthwise-Separable causal TCN for 4-state EEG classification.

    Default: 4 blocks with dilation [1,2,4,8] → receptive field = 61
    (closely matching the default 64-step input sequence).

    Input:  (batch, 25, seq_len)   -- 25 = 5 time-domain features x 5 channels
    Output: forward()        -> (batch, 4) raw logits (use with CrossEntropyLoss)
            predict_proba()  -> (batch, 4) softmax probabilities (use for inference)
    """

    NUM_CLASSES = 4
    IN_FEATURES = 25

    def __init__(
        self,
        in_channels: int = IN_FEATURES,
        num_classes: int = NUM_CLASSES,
        channel_sizes: Tuple = (32, 32, 48, 48),
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilation_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
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
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, T) -> (B, C, 1)
        self.classifier = nn.Linear(prev_channels, num_classes)

        # Print architecture summary
        self._print_architecture_summary()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, num_classes)."""
        y = self.tcn(x)                 # (B, C, T)
        y = self.global_pool(y).squeeze(-1)  # (B, C)
        logits = self.classifier(y)     # (B, num_classes)
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
        print(f"\n{'─'*50}")
        print(f"  TCN Architecture ({len(self.channel_sizes)} blocks)")
        print(f"{'─'*50}")
        print(f"  {'Block':>5s}  {'Channels':>10s}  {'Dilation':>8s}  "
              f"{'RF contribution':>15s}  {'Cumulative RF':>14s}")
        cum_rf = 1
        for i, (ch, d) in enumerate(zip(self.channel_sizes,
                                         self.dilation_schedule)):
            in_ch = self.IN_FEATURES if i == 0 else self.channel_sizes[i-1]
            contrib = 2 * (self.kernel_size - 1) * d
            cum_rf += contrib
            print(f"  {i:>5d}  {in_ch:>4d}→{ch:<4d}  {d:>8d}  "
                  f"{contrib:>15d}  {cum_rf:>14d}")
        print(f"{'─'*50}")
        print(f"  Kernel size:        {self.kernel_size}")
        print(f"  Receptive field:    {rf} timesteps")
        print(f"  Trainable params:   {self.count_parameters():,}")
        print(f"{'─'*50}\n")


# --------------------------------------------------------------------------- #
# 3. Training / Validation loop
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 10          # early stopping patience
    out_path: str = "eeg_tcn_best.pth"
    test_npz: str = ""   # optional held-out test set
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    class_weighting: str = "none"   # "inverse_freq", "custom", "dynamic", or "none"
    custom_weights: str = "0.7,1.5,2.0,1.5"   # only used when class_weighting == "custom"
    dynamic_monitor_frac: float = 0.10  # fraction of training data for dynamic weight monitoring


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
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

    return total_loss / total, correct / total, macro_f1


def train_model(model: nn.Module, train_loader, val_loader, cfg: TrainConfig):
    device = torch.device(cfg.device)
    model.to(device)

    # --- Class weighting (toggleable) ---
    # Get all training labels so we can report the distribution either way,
    # and only build class weights if the config asks for them. Stacking
    # inverse-frequency weights on top of an already-undersampled training
    # set tends to massively over-correct the rarest class (Ictal), so this
    # is exposed as a CLI flag (--class_weighting none|inverse_freq|custom|dynamic)
    # rather than always-on.
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

        # Split training data into train-proper + monitor (for weight adaptation)
        full_dataset = train_loader.dataset
        n_total = len(full_dataset)
        n_monitor = max(1, int(n_total * cfg.dynamic_monitor_frac))
        n_train_proper = n_total - n_monitor

        generator = torch.Generator().manual_seed(42)
        train_proper_ds, monitor_ds = torch.utils.data.random_split(
            full_dataset, [n_train_proper, n_monitor], generator=generator,
        )

        print(f"Dynamic weighting: split training data into "
              f"{n_train_proper} train-proper + {n_monitor} monitor samples")

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

        dw_cfg = DynamicWeightConfig()  # conservative defaults
        dynamic_weights_manager = DynamicClassWeights(dw_cfg, device=device)
        criterion = dynamic_weights_manager.get_criterion()

    elif cfg.class_weighting == "none":
        print("Class weighting disabled (cfg.class_weighting='none') -- "
              "relying on undersampling alone for balance.")
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(
            f"Unknown class_weighting='{cfg.class_weighting}'. "
            f"Expected 'inverse_freq', 'custom', 'dynamic', or 'none'."
        )
    # --------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Model selection now tracks val_macro_f1 (not val_loss) -- so the LR
    # scheduler is switched to mode="max" on the same metric, otherwise the
    # scheduler and the early-stopping/checkpoint criterion would disagree
    # about what "improvement" means.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -1.0
    epochs_no_improve = 0

    print(f"Training on {device} | model params: {model.count_parameters():,}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, actual_train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_f1)

        # --- Dynamic weight update (if enabled) ---
        if dynamic_weights_manager is not None:
            criterion = dynamic_weights_manager.maybe_update(
                epoch, model, monitor_loader, device
            )

        dt = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | val_macro_f1={val_f1:.4f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{dt:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "channel_sizes": [m.conv1.pointwise.out_channels for m in model.tcn],
                },
                cfg.out_path,
            )
            print(f"  -> new best model saved to {cfg.out_path} (val_macro_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs).")
                break

    print(f"Training complete. Best val_macro_f1={best_val_f1:.4f}. Weights saved to {cfg.out_path}")

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
        test_ds = EEGWindowDataset(cfg.test_npz, seq_len=args_seq_len)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        test_metrics = _evaluate_partition(model, test_loader, device, "TEST")

    return {
        "best_val_macro_f1": best_val_f1,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "out_path": cfg.out_path,
    }


# --------------------------------------------------------------------------- #
# 3b. Comprehensive evaluation helper
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

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")

    # Zero-prediction alert
    zero_classes = [target_names[i] for i in range(4) if pred_counts[i] == 0]
    if zero_classes:
        print(f"\n  ⚠ WARNING: Model predicted ZERO samples for: "
              f"{', '.join(zero_classes)}")

    print(f"{'─'*50}\n")

    # Per-class precision/recall/f1, returned so callers (e.g. a weight
    # sweep) can compare experiments without re-parsing printed text.
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0,
    )
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
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--class_weighting", type=str, default="none",
                   choices=["inverse_freq", "custom", "dynamic", "none"],
                   help="'none' (default), 'inverse_freq' (inverse-frequency), "
                        "'custom' (weights set via --custom_weights), or "
                        "'dynamic' (adaptive weights based on per-class F1)")
    p.add_argument("--custom_weights", type=str, default="0.7,1.5,2.0,1.5",
                   help="Comma-separated 4 weights (Interictal,Preictal,Ictal,"
                        "Postictal), only used when --class_weighting custom. "
                        "e.g. '0.7,1.5,3.0,1.5'")
    p.add_argument("--dynamic_monitor_frac", type=float, default=0.10,
                   help="Fraction of training data held out for dynamic weight "
                        "monitoring (default: 0.10). Only used when "
                        "--class_weighting dynamic.")
    # Architecture
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--channel_sizes", type=str, default="32,32,48,48",
                   help="Comma-separated channel counts per block "
                        "(default: 32,32,48,48 = 4 blocks)")
    p.add_argument("--dilation_schedule", type=str, default=None,
                   help="Comma-separated dilation per block. Default: "
                        "exponential (1,2,4,8,...)")
    # I/O
    p.add_argument("--out_path", type=str, default="eeg_tcn_best.pth")
    p.add_argument("--seq_len", type=int, default=64,
                   help="Sequence length (used when .npz is 2D). Default: 64")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Module-level variable so train_model's test evaluation can access seq_len
args_seq_len = 64


def main():
    global args_seq_len
    args = parse_args()
    args_seq_len = args.seq_len

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse architecture strings
    channel_sizes = tuple(int(x) for x in args.channel_sizes.split(','))
    dilation_schedule = None
    if args.dilation_schedule:
        dilation_schedule = [int(x) for x in args.dilation_schedule.split(',')]

    train_loader, val_loader = build_dataloaders(
        args.train_npz, args.val_npz, batch_size=args.batch_size,
        num_workers=args.num_workers, seq_len=args.seq_len,
    )

    model = EEG_TCN(
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        channel_sizes=channel_sizes,
        dilation_schedule=dilation_schedule,
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        out_path=args.out_path,
        test_npz=args.test_npz,
        class_weighting=args.class_weighting,
        custom_weights=args.custom_weights,
        dynamic_monitor_frac=args.dynamic_monitor_frac,
    )

    print(f"\nExperiment config:")
    print(f"  train:            {args.train_npz}")
    print(f"  val:              {args.val_npz}")
    print(f"  test:             {args.test_npz or '(none)'}")
    print(f"  class_weighting:  {args.class_weighting}")
    if args.class_weighting == "custom":
        print(f"  custom_weights:   {args.custom_weights}")
    if args.class_weighting == "dynamic":
        print(f"  monitor_frac:     {args.dynamic_monitor_frac}")
    print(f"  channel_sizes:    {channel_sizes}")
    print(f"  dilation:         {dilation_schedule or 'exponential'}")
    print(f"  kernel_size:      {args.kernel_size}")
    print(f"  seq_len:          {args.seq_len}")
    print(f"  seed:             {args.seed}")
    print()

    train_model(model, train_loader, val_loader, cfg)


if __name__ == "__main__":
    main()