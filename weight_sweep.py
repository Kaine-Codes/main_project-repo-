"""
weight_sweep.py
================
Runs eeg_tcn.py's training loop once per candidate `custom` class-weight
vector, then prints (and saves) a comparison table so you can pick the
best set of weights without manually re-running the CLI and copy-pasting
numbers out of the console.

Trains a FRESH model for every weight combo (same architecture, same seed
before each init, same data) -- only the loss weights differ between runs.
Each run's dataloaders are reused across experiments (data doesn't change),
but torch/np RNG is reset before each model is built so init + first-epoch
shuffling are matched across experiments as closely as practical.

Usage (mirrors eeg_tcn.py's CLI args):
    python weight_sweep.py \\
        --train_npz chb15_train_balanced.npz \\
        --val_npz   chb15_val.npz \\
        --test_npz  chb15_test.npz \\
        --channel_sizes 32,32,48,48 \\
        --epochs 50 --lr 1e-3 --batch_size 64 --patience 10 \\
        --seed 42

Edit EXPERIMENTS below to add/remove weight combos.
"""

import argparse
import copy
import csv
import os

import numpy as np
import torch

import eeg_tcn
from eeg_tcn import EEG_TCN, TrainConfig, build_dataloaders, train_model


# --------------------------------------------------------------------------- #
# Candidate weight vectors: [Interictal, Preictal, Ictal, Postictal]
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "C": [1.0, 1.5, 3.0, 0.7],
    "D": [0.9, 1.3, 3.0, 0.7],
    "E": [0.9, 1.7, 3.0, 0.7],
    "F": [0.9, 1.5, 2.5, 0.7],
    "G": [0.9, 1.5, 3.5, 0.7],
    "H": [0.9, 1.5, 3.0, 0.9],
}


def parse_args():
    p = argparse.ArgumentParser(description="Sweep custom class weights for eeg_tcn.py")
    p.add_argument("--train_npz", type=str, required=True)
    p.add_argument("--val_npz", type=str, required=True)
    p.add_argument("--test_npz", type=str, default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--channel_sizes", type=str, default="32,32,48,48")
    p.add_argument("--dilation_schedule", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=".",
                    help="Directory to save per-experiment checkpoints + the summary CSV")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eeg_tcn.args_seq_len = args.seq_len  # used internally by train_model's test-set loading

    channel_sizes = tuple(int(x) for x in args.channel_sizes.split(','))
    dilation_schedule = None
    if args.dilation_schedule:
        dilation_schedule = [int(x) for x in args.dilation_schedule.split(',')]

    # Data doesn't change across weight experiments -- build once.
    train_loader, val_loader = build_dataloaders(
        args.train_npz, args.val_npz, batch_size=args.batch_size,
        num_workers=args.num_workers, seq_len=args.seq_len,
    )

    results = []

    for name, weights in EXPERIMENTS.items():
        print(f"\n{'#'*70}")
        print(f"  EXPERIMENT {name}: weights = {weights}")
        print(f"{'#'*70}")

        # Reset RNG before each model so init (and first-epoch shuffling)
        # is matched across experiments -- isolates the effect of the
        # weights themselves as much as practical.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
            out_path=os.path.join(args.out_dir, f"eeg_tcn_{name}.pth"),
            test_npz=args.test_npz,
            class_weighting="custom",
            custom_weights=",".join(str(w) for w in weights),
        )

        result = train_model(model, train_loader, val_loader, cfg)
        result["experiment"] = name
        result["weights"] = weights
        results.append(result)

    # ------------------------------------------------------------------- #
    # Summary table
    # ------------------------------------------------------------------- #
    print(f"\n\n{'='*100}")
    print("  WEIGHT SWEEP SUMMARY")
    print(f"{'='*100}")

    header = (f"{'Exp':<5}{'Weights':<24}{'val_macro_f1':>13}{'val_bal_acc':>12}"
              f"{'val_preictal_R':>15}{'val_ictal_R':>12}{'test_macro_f1':>14}{'test_preictal_R':>17}")
    print(header)
    print("-" * len(header))

    csv_rows = []
    for r in results:
        vm = r["val_metrics"]
        tm = r["test_metrics"]
        val_preictal_r = vm["per_class"]["preictal"]["recall"]
        val_ictal_r = vm["per_class"]["ictal"]["recall"]
        test_macro_f1 = tm["macro_f1"] if tm else float("nan")
        test_preictal_r = tm["per_class"]["preictal"]["recall"] if tm else float("nan")

        print(f"{r['experiment']:<5}{str(r['weights']):<24}{vm['macro_f1']:>13.4f}"
              f"{vm['balanced_accuracy']:>12.4f}{val_preictal_r:>15.4f}"
              f"{val_ictal_r:>12.4f}{test_macro_f1:>14.4f}{test_preictal_r:>17.4f}")

        csv_rows.append({
            "experiment": r["experiment"],
            "weights": r["weights"],
            "val_macro_f1": vm["macro_f1"],
            "val_balanced_accuracy": vm["balanced_accuracy"],
            "val_preictal_recall": val_preictal_r,
            "val_preictal_precision": vm["per_class"]["preictal"]["precision"],
            "val_ictal_recall": val_ictal_r,
            "val_interictal_recall": vm["per_class"]["interictal"]["recall"],
            "val_postictal_recall": vm["per_class"]["postictal"]["recall"],
            "test_macro_f1": test_macro_f1,
            "test_preictal_recall": test_preictal_r,
            "checkpoint": r["out_path"],
        })

    best_by_val_f1 = max(results, key=lambda r: r["val_metrics"]["macro_f1"])
    best_by_preictal_recall = max(
        results, key=lambda r: r["val_metrics"]["per_class"]["preictal"]["recall"]
    )
    print(f"\nBest val_macro_f1:        {best_by_val_f1['experiment']} "
          f"(weights={best_by_val_f1['weights']}, "
          f"val_macro_f1={best_by_val_f1['val_metrics']['macro_f1']:.4f})")
    print(f"Best val Preictal recall: {best_by_preictal_recall['experiment']} "
          f"(weights={best_by_preictal_recall['weights']}, "
          f"recall={best_by_preictal_recall['val_metrics']['per_class']['preictal']['recall']:.4f})")
    if best_by_val_f1["experiment"] != best_by_preictal_recall["experiment"]:
        print("  NOTE: these two disagree -- macro F1 optimizes all 4 classes equally,\n"
              "  so check both before picking a final weight vector.")

    csv_path = os.path.join(args.out_dir, "weight_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSummary saved -> {csv_path}")


if __name__ == "__main__":
    main()
