"""
pipeline_patient.py
===================
Unified single-patient experiment pipeline.

Replaces the old workflow:  split_patient.py → undersample_interictal.py
with a scientifically rigorous pipeline that:

  1. Lists all EDFs for one patient.
  2. Splits them chronologically into TRAIN / VAL / TEST (default 70/15/15).
  3. Guarantees at least one seizure-containing EDF in VAL and TEST.
  4. Computes calibration μ/σ from TRAINING interictal segments ONLY.
  5. Processes each partition independently (filter → normalize → window →
     features → label → sequences) using the SAME training calibration.
  6. Undersamples Interictal in TRAINING only.
  7. Saves .npz files + config JSON for reproducibility.
  8. Runs automated sanity checks.

Usage:
    python pipeline_patient.py --patient_dir "CHB DATASET/chb15" \\
        --seq_len 64 --seq_stride 4 --seed 42

Then train:
    python eeg_tcn.py \\
        --train_npz chb15_train_balanced.npz \\
        --val_npz   chb15_val.npz \\
        --test_npz  chb15_test.npz \\
        --class_weighting none --channel_sizes 32,32,48,48
"""

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np

from preprocess_chbmit import (
    parse_seizure_summary, load_edf, bandpass_filter,
    zscore_apply, segment_windows, extract_features, label_windows,
    compute_patient_calibration,
    CHANNELS, TARGET_SFREQ, WINDOW_SEC, DEFAULT_OVERLAP,
    PREICTAL_SEC, POSTICTAL_SEC, LABEL_MAP, FEATURE_NAMES,
)
from sanity_checks import run_all_data_checks

LABEL_NAMES = {0: "Interictal", 1: "Preictal", 2: "Ictal", 3: "Postictal"}


# ────────────────────────────────────────────────────────────────────────── #
# Configuration
# ────────────────────────────────────────────────────────────────────────── #

@dataclass
class PipelineConfig:
    patient_dir: str = ""
    patient_id: str = ""
    seed: int = 42

    # Windowing
    seq_len: int = 64
    seq_stride: int = 4
    window_sec: float = WINDOW_SEC
    overlap: float = DEFAULT_OVERLAP

    # Labeling
    preictal_sec: float = PREICTAL_SEC
    postictal_sec: float = POSTICTAL_SEC

    # Splitting
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Undersampling
    undersample_target: Optional[int] = None  # None = match preictal count
    undersample_postictal_target: Optional[int] = None  # None = match preictal count

    # Class weighting (for reference in config, enforced in eeg_tcn.py)
    use_class_weights: bool = False

    # Output
    output_dir: str = "."


# ────────────────────────────────────────────────────────────────────────── #
# 1. EDF analysis
# ────────────────────────────────────────────────────────────────────────── #

def analyze_edfs(patient_dir: str) -> Tuple[List[dict], str]:
    """
    List all EDFs in a patient directory and analyze their seizure content.

    Returns
    -------
    edf_info : list[dict]   – one entry per EDF with seizure details
    summary_path : str      – path to the summary file
    """
    summary_candidates = glob.glob(os.path.join(patient_dir, '*-summary.txt'))
    if not summary_candidates:
        raise FileNotFoundError(
            f"No *-summary.txt found in {patient_dir}")
    summary_path = summary_candidates[0]

    edf_paths = sorted(glob.glob(os.path.join(patient_dir, '*.edf')))
    if not edf_paths:
        raise FileNotFoundError(f"No .edf files found in {patient_dir}")

    edf_info = []
    for edf_path in edf_paths:
        filename = os.path.basename(edf_path)
        try:
            seizures = parse_seizure_summary(summary_path, filename)
        except ValueError:
            seizures = []

        edf_info.append({
            'path': edf_path,
            'filename': filename,
            'seizures': seizures,
            'has_seizure': len(seizures) > 0,
            'n_seizures': len(seizures),
            'total_ictal_sec': sum(e - s for s, e in seizures),
        })

    return edf_info, summary_path


# ────────────────────────────────────────────────────────────────────────── #
# 2. Chronological 3-way split
# ────────────────────────────────────────────────────────────────────────── #

def split_edfs(edf_info: List[dict], cfg: PipelineConfig
               ) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Chronological split into TRAIN / VAL / TEST with seizure guarantees.

    Strategy:
      1. Take the first ~70% as train, next ~15% as val, last ~15% as test.
      2. If val has no seizure-containing EDF, pull the last seizure-containing
         EDF from train into val (minimal chronological disruption).
      3. Same for test: if no seizure, pull from train.
    """
    n = len(edf_info)
    n_test = max(1, round(n * cfg.test_ratio))
    n_val = max(1, round(n * cfg.val_ratio))
    n_train = n - n_val - n_test

    train = list(edf_info[:n_train])
    val = list(edf_info[n_train:n_train + n_val])
    test = list(edf_info[n_train + n_val:])

    # Guarantee seizure in VAL
    if not any(e['has_seizure'] for e in val):
        seizure_idxs = [i for i, e in enumerate(train) if e['has_seizure']]
        if seizure_idxs:
            idx = seizure_idxs[-1]  # last seizure file in train
            val.insert(0, train.pop(idx))
            print("  NOTE: Moved a seizure EDF from train→val to guarantee "
                  "seizure representation.")
        else:
            print("  WARNING: No seizure EDFs available in train to move to val!")

    # Guarantee seizure in TEST
    if not any(e['has_seizure'] for e in test):
        seizure_idxs = [i for i, e in enumerate(train) if e['has_seizure']]
        if seizure_idxs:
            idx = seizure_idxs[-1]
            test.insert(0, train.pop(idx))
            print("  NOTE: Moved a seizure EDF from train→test to guarantee "
                  "seizure representation.")
        else:
            print("  WARNING: No seizure EDFs available in train to move to test!")

    return train, val, test


def print_split_info(train: List[dict], val: List[dict], test: List[dict]):
    """Print detailed info about the split."""
    for name, partition in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        n_edfs = len(partition)
        n_seizure_edfs = sum(1 for e in partition if e['has_seizure'])
        n_seizures = sum(e['n_seizures'] for e in partition)
        total_ictal = sum(e['total_ictal_sec'] for e in partition)
        files = [e['filename'] for e in partition]
        seizure_files = [e['filename'] for e in partition if e['has_seizure']]

        print(f"\n  {name} ({n_edfs} EDFs):")
        print(f"    Files: {files}")
        print(f"    Seizure files ({n_seizure_edfs}): {seizure_files}")
        print(f"    Seizures: {n_seizures}")
        print(f"    Ictal duration: {total_ictal:.0f}s "
              f"({total_ictal/60:.1f} min)")


# ────────────────────────────────────────────────────────────────────────── #
# 3. Calibration from training interictal data only
# ────────────────────────────────────────────────────────────────────────── #

def compute_training_calibration(train_edfs: List[dict], summary_path: str,
                                  cfg: PipelineConfig
                                  ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute per-patient Z-score calibration from TRAINING interictal
    segments only.  Returns (mean, std, n_samples_used).
    """
    edf_summary_pairs = [
        (e['path'], summary_path, e['filename'])
        for e in train_edfs
    ]
    mean, std, n_samples = compute_patient_calibration(
        edf_summary_pairs,
        preictal_sec=cfg.preictal_sec,
        postictal_sec=cfg.postictal_sec,
    )
    return mean, std, n_samples


# ────────────────────────────────────────────────────────────────────────── #
# 4. Process one partition
# ────────────────────────────────────────────────────────────────────────── #

def process_partition(partition_edfs: List[dict], summary_path: str,
                      cfg: PipelineConfig,
                      calib_mean: np.ndarray, calib_std: np.ndarray,
                      partition_name: str = ""
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                 Optional[np.ndarray]]:
    """
    Process a list of EDFs into 3D sequences using fixed calibration.

    Returns (X, y, groups)  or  (None, None, None) if nothing could be
    processed.

    X      : (N, seq_len, 25)  — feature sequences
    y      : (N,)              — integer labels (last timestep)
    groups : (N,)              — file index per sequence
    """
    all_seqs, all_labels, all_groups = [], [], []

    for file_idx, edf in enumerate(partition_edfs):
        edf_path = edf['path']
        filename = edf['filename']

        try:
            raw = load_edf(edf_path, channels=CHANNELS)
            sfreq = raw.info['sfreq']
            if sfreq != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose='ERROR')
            bandpass_filter(raw)

            data = raw.get_data()  # (n_channels, n_samples)
            data = zscore_apply(data, calib_mean, calib_std)

            windows, window_times = segment_windows(
                data, sfreq=TARGET_SFREQ,
                window_sec=cfg.window_sec, overlap=cfg.overlap,
            )
            features = extract_features(windows, sfreq=TARGET_SFREQ)

            seizures = parse_seizure_summary(summary_path, filename)
            _, label_ids = label_windows(
                window_times, cfg.window_sec, seizures,
                preictal_sec=cfg.preictal_sec,
                postictal_sec=cfg.postictal_sec,
            )

            if len(features) < cfg.seq_len:
                print(f"    SKIPPED {filename}: only {len(features)} windows, "
                      f"need seq_len={cfg.seq_len}")
                continue

            # Sliding sequences within this file
            X_3d = np.lib.stride_tricks.sliding_window_view(
                features, window_shape=cfg.seq_len, axis=0
            )
            X_3d = np.moveaxis(X_3d, -1, 1).copy()  # (N, seq_len, 25)
            y_seq = label_ids[cfg.seq_len - 1:]       # label = last timestep

            # Stride thinning
            X_3d = X_3d[::cfg.seq_stride]
            y_seq = y_seq[::cfg.seq_stride]

            all_seqs.append(X_3d)
            all_labels.append(y_seq)
            all_groups.append(np.full(len(y_seq), file_idx, dtype=np.int32))

            unique, counts = np.unique(y_seq, return_counts=True)
            dist = {LABEL_NAMES.get(u, u): int(c)
                    for u, c in zip(unique.tolist(), counts.tolist())}
            print(f"    {filename}: {len(y_seq):,} sequences → {dist}")

        except Exception as e:
            print(f"    ERROR {filename}: {e}")
            continue

    if not all_seqs:
        return None, None, None

    X = np.concatenate(all_seqs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    return X, y, groups


# ────────────────────────────────────────────────────────────────────────── #
# 5. Undersampling
# ────────────────────────────────────────────────────────────────────────── #

def undersample_interictal(X: np.ndarray, y: np.ndarray,
                            target: Optional[int] = None,
                            seed: int = 42
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersample the Interictal class in training data.

    Parameters
    ----------
    target : int or None
        Target count for Interictal.  If None, defaults to the Preictal count.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=4)

    if target is None:
        target = int(counts[1])  # match Preictal count
        print(f"  Undersample Interictal target (auto): {target} "
              f"(matching Preictal count)")

    interictal_idx = np.where(y == 0)[0]
    other_idx = np.where(y != 0)[0]

    if target >= len(interictal_idx):
        print(f"  Target ({target}) >= Interictal count "
              f"({len(interictal_idx)}); no undersampling needed.")
        return X, y

    keep_interictal = rng.choice(interictal_idx, size=target, replace=False)
    keep_idx = np.concatenate([keep_interictal, other_idx])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


def undersample_postictal(X: np.ndarray, y: np.ndarray,
                          target: Optional[int] = None,
                          seed: int = 42
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersample the Postictal class (label 3) in training data.

    Parameters
    ----------
    target : int or None
        Target count for Postictal.  If None, defaults to the Preictal count.
    """
    rng = np.random.default_rng(seed + 1)  # different seed from interictal
    counts = np.bincount(y, minlength=4)

    if target is None:
        target = int(counts[1])  # match Preictal count
        print(f"  Undersample Postictal target (auto): {target} "
              f"(matching Preictal count)")

    postictal_idx = np.where(y == 3)[0]
    other_idx = np.where(y != 3)[0]

    if target >= len(postictal_idx):
        print(f"  Target ({target}) >= Postictal count "
              f"({len(postictal_idx)}); no undersampling needed.")
        return X, y

    keep_postictal = rng.choice(postictal_idx, size=target, replace=False)
    keep_idx = np.concatenate([keep_postictal, other_idx])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


# ────────────────────────────────────────────────────────────────────────── #
# 6. Distribution printing
# ────────────────────────────────────────────────────────────────────────── #

def print_distribution(y: np.ndarray, name: str = ""):
    """Print class distribution with counts and percentages."""
    counts = np.bincount(y, minlength=4)
    total = len(y)
    print(f"  {name} ({total:,} sequences):")
    for cls in range(4):
        pct = 100.0 * counts[cls] / total if total > 0 else 0
        print(f"    {LABEL_NAMES[cls]:>12s}: {counts[cls]:>8,} ({pct:5.1f}%)")


# ────────────────────────────────────────────────────────────────────────── #
# 7. Save
# ────────────────────────────────────────────────────────────────────────── #

def save_partition(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   out_path: str):
    """Save a partition to .npz with standard keys."""
    np.savez_compressed(
        out_path,
        windows=X,
        features=X,
        labels=y,
        groups=groups,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"  Saved {X.shape[0]:,} sequences of shape {X.shape[1:]} "
          f"→ {out_path}")


# ────────────────────────────────────────────────────────────────────────── #
# 8. Main pipeline
# ────────────────────────────────────────────────────────────────────────── #

def run_pipeline(cfg: PipelineConfig):
    """Execute the full single-patient pipeline."""

    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)
    patient_id = cfg.patient_id or os.path.basename(cfg.patient_dir.rstrip('/\\'))

    print(f"\n{'='*60}")
    print(f"  Pipeline: {patient_id}")
    print(f"  seq_len={cfg.seq_len}  seq_stride={cfg.seq_stride}  "
          f"seed={cfg.seed}")
    print(f"{'='*60}")

    # ── 1. Analyze EDFs ──────────────────────────────────────────────────
    print("\n[1/7] Analyzing EDFs ...")
    edf_info, summary_path = analyze_edfs(cfg.patient_dir)
    n_seizure_files = sum(1 for e in edf_info if e['has_seizure'])
    n_seizures = sum(e['n_seizures'] for e in edf_info)
    total_ictal = sum(e['total_ictal_sec'] for e in edf_info)
    print(f"  Found {len(edf_info)} EDFs "
          f"({n_seizure_files} with seizures, "
          f"{n_seizures} total seizures, "
          f"{total_ictal:.0f}s ictal)")

    # ── 2. Split ─────────────────────────────────────────────────────────
    print(f"\n[2/7] Splitting EDFs ({cfg.train_ratio:.0%} / "
          f"{cfg.val_ratio:.0%} / {cfg.test_ratio:.0%}) ...")
    train_edfs, val_edfs, test_edfs = split_edfs(edf_info, cfg)
    print_split_info(train_edfs, val_edfs, test_edfs)

    train_files = [e['filename'] for e in train_edfs]
    val_files = [e['filename'] for e in val_edfs]
    test_files = [e['filename'] for e in test_edfs]

    # ── 3. Calibration ───────────────────────────────────────────────────
    print(f"\n[3/7] Computing calibration mean/std from TRAINING interictal "
          f"segments only ...")
    t0 = time.time()
    calib_mean, calib_std, n_calib = compute_training_calibration(
        train_edfs, summary_path, cfg)
    print(f"  Calibrated from {n_calib:,} interictal samples "
          f"across {len(train_edfs)} training files ({time.time()-t0:.1f}s)")
    print(f"  mean = {np.array2string(calib_mean, precision=6, separator=', ')}")
    print(f"  std = {np.array2string(calib_std, precision=6, separator=', ')}")

    # ── 4. Process partitions ────────────────────────────────────────────
    results = {}
    for part_name, part_edfs in [("TRAIN", train_edfs),
                                   ("VAL", val_edfs),
                                   ("TEST", test_edfs)]:
        part_num = {"TRAIN": "4a", "VAL": "4b", "TEST": "4c"}[part_name]
        print(f"\n[{part_num}/7] Processing {part_name} "
              f"({len(part_edfs)} EDFs) ...")
        X, y, groups = process_partition(
            part_edfs, summary_path, cfg, calib_mean, calib_std,
            partition_name=part_name,
        )
        if X is None:
            raise RuntimeError(
                f"No usable data in {part_name} partition! "
                f"Check EDF files and channel availability.")
        results[part_name] = (X, y, groups)
        print_distribution(y, part_name)

    X_train, y_train, g_train = results["TRAIN"]
    X_val, y_val, g_val = results["VAL"]
    X_test, y_test, g_test = results["TEST"]

    # ── 5. Sanity checks ────────────────────────────────────────────────
    print(f"\n[5/7] Running sanity checks ...")
    run_all_data_checks(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        calib_files=train_files,  # calibration source = training files
        calib_mean=calib_mean,
        calib_std=calib_std,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        use_class_weights=cfg.use_class_weights,
    )

    # ── 6. Save unbalanced partitions ────────────────────────────────────
    print(f"\n[6/7] Saving partitions ...")
    prefix = os.path.join(cfg.output_dir, patient_id)

    save_partition(X_train, y_train, g_train, f"{prefix}_train.npz")
    save_partition(X_val, y_val, g_val, f"{prefix}_val.npz")
    save_partition(X_test, y_test, g_test, f"{prefix}_test.npz")

    # ── 7. Undersample training (interictal + postictal) ─────────────────
    print(f"\n[7/7] Undersampling Interictal + Postictal in TRAINING ...")
    print_distribution(y_train, "BEFORE undersampling")

    X_bal, y_bal = undersample_interictal(
        X_train, y_train,
        target=cfg.undersample_target,
        seed=cfg.seed,
    )
    print_distribution(y_bal, "AFTER Interictal undersampling")

    X_bal, y_bal = undersample_postictal(
        X_bal, y_bal,
        target=cfg.undersample_postictal_target,
        seed=cfg.seed,
    )
    print_distribution(y_bal, "AFTER Postictal undersampling")

    save_partition(X_bal, y_bal,
                   np.zeros(len(y_bal), dtype=np.int32),  # groups not meaningful after shuffle
                   f"{prefix}_train_balanced.npz")

    # ── Save config ──────────────────────────────────────────────────────
    config_dict = asdict(cfg)
    config_dict.update({
        'patient_id': patient_id,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'calibration_mean': calib_mean.tolist(),
        'calibration_std': calib_std.tolist(),
        'calibration_n_samples': n_calib,
        'train_sequences': int(X_train.shape[0]),
        'val_sequences': int(X_val.shape[0]),
        'test_sequences': int(X_test.shape[0]),
        'train_balanced_sequences': int(X_bal.shape[0]),
        'train_distribution': np.bincount(y_train, minlength=4).tolist(),
        'train_balanced_distribution': np.bincount(y_bal, minlength=4).tolist(),
        'val_distribution': np.bincount(y_val, minlength=4).tolist(),
        'test_distribution': np.bincount(y_test, minlength=4).tolist(),
        'total_seizures': n_seizures,
        'total_ictal_seconds': total_ictal,
    })

    config_path = f"{prefix}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"\n  Config saved → {config_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pipeline complete for {patient_id}")
    print(f"{'='*60}")
    print(f"\n  Output files:")
    print(f"    {prefix}_train.npz          ({X_train.shape[0]:,} sequences)")
    print(f"    {prefix}_train_balanced.npz ({X_bal.shape[0]:,} sequences)")
    print(f"    {prefix}_val.npz            ({X_val.shape[0]:,} sequences)")
    print(f"    {prefix}_test.npz           ({X_test.shape[0]:,} sequences)")
    print(f"    {prefix}_config.json")
    print(f"\n  Next step — train TCN:")
    print(f"    python eeg_tcn.py \\")
    print(f"      --train_npz {prefix}_train_balanced.npz \\")
    print(f"      --val_npz   {prefix}_val.npz \\")
    print(f"      --test_npz  {prefix}_test.npz \\")
    print(f"      --class_weighting none \\")
    print(f"      --channel_sizes 32,32,48,48 \\")
    print(f"      --epochs 50 --lr 1e-3 --batch_size 64")
    print()


# ────────────────────────────────────────────────────────────────────────── #
# CLI
# ────────────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-patient EEG pipeline: split → calibrate → "
                    "process → undersample → save."
    )
    p.add_argument('--patient_dir', required=True,
                   help="Patient folder, e.g. 'CHB DATASET/chb15'")
    p.add_argument('--output_dir', default='.',
                   help="Directory for output .npz and config files")
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--seq_stride', type=int, default=4)
    p.add_argument('--train_ratio', type=float, default=0.70)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--preictal_min', type=float, default=30.0,
                   help="Preictal duration in minutes (default 30)")
    p.add_argument('--postictal_min', type=float, default=30.0,
                   help="Postictal duration in minutes (default 30)")
    p.add_argument('--overlap', type=float, default=0.5,
                   help="Window overlap fraction (default 0.5)")
    p.add_argument('--undersample_target', type=int, default=None,
                   help="Target Interictal count after undersampling. "
                        "Default: match Preictal count.")
    p.add_argument('--undersample_postictal_target', type=int, default=None,
                   help="Target Postictal count after undersampling. "
                        "Default: match Preictal count.")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = PipelineConfig(
        patient_dir=args.patient_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        preictal_sec=args.preictal_min * 60,
        postictal_sec=args.postictal_min * 60,
        overlap=args.overlap,
        undersample_target=args.undersample_target,
        undersample_postictal_target=args.undersample_postictal_target,
        use_class_weights=False,  # Experiment 1: no class weights
        seed=args.seed,
    )

    run_pipeline(cfg)


if __name__ == '__main__':
    main()
