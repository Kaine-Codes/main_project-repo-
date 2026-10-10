"""
Build a combined multi-patient TRAIN set (every patient folder except 'test')
and a TEST set from the 'test' folder.

Folder layout expected:
    CHB DATASET/
      chb01/   chb01_02.edf, chb01_03.edf, chb01-summary.txt, ...
      chb05/   ...
      chb06/   ...
      chb08/   ...
      chb10/   ...
      chb12/   ...
      chb15/   ...
      chb18/   ...
      chb21/   ...
      test/    held-out files, test-summary.txt

Run:
    python build_train_test.py
"""

import glob
import os
import numpy as np
from preprocess_chbmit import preprocess_recording, compute_patient_calibration


def process_subject_dir(subject_dir, seq_len, seq_stride=1):
    """
    Process every .edf in ONE patient folder into (sequences, labels,
    feature_names, groups). Sequences are built within each file separately
    (never across file boundaries) before being concatenated, so a sequence
    never splices two unrelated recordings.

    Before extracting features, computes ONE fixed Z-score calibration
    (mean/std) for this patient from interictal-only segments pooled across
    ALL of this patient's recordings, then applies that SAME calibration to
    every file -- including files that contain a seizure. This matches the
    hardware calibration spec (fixed per-patient constants from seizure-free
    data) and avoids a seizure's amplitude/variance biasing the normalization
    of the very windows (Preictal) you're trying to detect.

    `groups` gives each output sequence an integer id identifying which
    source .edf file it came from (0, 1, 2, ... in the order files were
    processed). This lets a downstream train/val splitter cut ONLY at file
    boundaries, so overlapping sliding-window sequences from the same file
    never end up split across train and val.

     controls redundancy: stride=1 keeps every possible overlapping
    sequence (very memory-heavy for long recordings); a larger stride keeps far
    fewer, less-redundant sequences.
    Returns None if nothing could be processed.
    """
    summary_candidates = glob.glob(os.path.join(subject_dir, '*-summary.txt'))
    if not summary_candidates:
        print(f"  SKIPPING FOLDER {subject_dir}: no *-summary.txt found")
        return None
    summary_path = summary_candidates[0]

    edf_paths = sorted(glob.glob(os.path.join(subject_dir, '*.edf')))
    if not edf_paths:
        print(f"  SKIPPING FOLDER {subject_dir}: no .edf files found")
        return None

    # --- Fixed per-patient calibration, fit on interictal segments across
    #     ALL of this patient's files (not per-file). ---
    edf_summary_pairs = [(p, summary_path, os.path.basename(p)) for p in edf_paths]
    try:
        calib_mean, calib_std, n_calib_samples = compute_patient_calibration(
            edf_summary_pairs
        )
        print(f"  Calibrated {subject_dir} from {n_calib_samples} interictal "
              f"samples across {len(edf_paths)} files.")
    except ValueError as e:
        print(f"  SKIPPING FOLDER {subject_dir}: calibration failed ({e})")
        return None

    seqs_list, labels_list, groups_list = [], [], []
    feature_names = None

    for file_idx, edf_path in enumerate(edf_paths):
        edf_filename = os.path.basename(edf_path)
        try:
            result = preprocess_recording(
                edf_path, summary_path, edf_filename=edf_filename,
                calibration_mean=calib_mean, calibration_std=calib_std,
            )
        except ValueError as e:
            print(f"    SKIPPED {edf_filename}: {e}")
            continue

        X_2d = result['windows']   # (n_windows, 25)
        y_1d = result['label_ids']

        if len(X_2d) < seq_len:
            print(f"    SKIPPED {edf_filename}: only {len(X_2d)} windows, need seq_len={seq_len}")
            continue

        # Sliding sequences of consecutive windows, within this file only.
        X_3d = np.lib.stride_tricks.sliding_window_view(X_2d, window_shape=seq_len, axis=0)
        X_3d = np.moveaxis(X_3d, -1, 1).copy()   # (n_sequences, seq_len, 25)
        y_seq = y_1d[seq_len - 1:]                 # label = last (most recent) window's state

        # Thin out overlap: keep every seq_stride-th sequence instead of ALL of them.
        X_3d = X_3d[::seq_stride]
        y_seq = y_seq[::seq_stride]

        seqs_list.append(X_3d)
        labels_list.append(y_seq)
        groups_list.append(np.full(len(y_seq), file_idx, dtype=np.int32))
        feature_names = result['feature_names']

        unique, counts = np.unique(y_seq, return_counts=True)
        print(f"    {edf_filename}: {X_3d.shape[0]} sequences -> {dict(zip(unique.tolist(), counts.tolist()))}")

    if not seqs_list:
        return None

    X = np.concatenate(seqs_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    return X, y, feature_names, groups


def build_dataset_from_dirs(subject_dirs, out_path, seq_len=64, seq_stride=1):
    """Process multiple patient folders and save ONE combined .npz."""
    all_X, all_y, all_groups = [], [], []
    feature_names = None
    group_offset = 0  # keeps group ids unique across patients, not just within one

    for subject_dir in subject_dirs:
        print(f"  Processing {subject_dir} ...")
        result = process_subject_dir(subject_dir, seq_len, seq_stride=seq_stride)
        if result is None:
            print(f"  SKIPPED FOLDER {subject_dir}: nothing usable")
            continue
        X, y, fn, groups = result
        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups + group_offset)
        group_offset += groups.max() + 1
        feature_names = fn

    if not all_X:
        raise RuntimeError(f"No usable data found across {subject_dirs}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    np.savez_compressed(
        out_path, windows=X, features=X, labels=y, groups=groups,
        feature_names=np.array(feature_names),
    )
    print(f"Saved {X.shape[0]} total sequences of shape {X.shape[1:]} to {out_path}")
    print(f"  ({groups.max() + 1} distinct source files tracked in 'groups')")

    unique, counts = np.unique(y, return_counts=True)
    print(f"Overall label distribution: {dict(zip(unique.tolist(), counts.tolist()))}\n")
    return X, y


if __name__ == '__main__':
    ROOT = 'CHB DATASET'
    TEST_FOLDER_NAME = 'test'
    SEQ_LEN = 64
    SEQ_STRIDE = 1   # keep 1/8th of possible overlapping sequences -- big memory reduction

    all_patient_dirs = sorted(
        d for d in glob.glob(os.path.join(ROOT, 'chb*')) if os.path.isdir(d)
    )
    train_dirs = [d for d in all_patient_dirs if os.path.basename(d) != TEST_FOLDER_NAME]
    test_dir = os.path.join(ROOT, TEST_FOLDER_NAME)

    print(f"Train folders ({len(train_dirs)}): {[os.path.basename(d) for d in train_dirs]}\n")

    print("Building TRAIN set from all non-test patient folders...")
    build_dataset_from_dirs(train_dirs, 'chb_train.npz', seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE)

    print("Building TEST set from 'test' folder...")
    build_dataset_from_dirs([test_dir], 'chb_test.npz', seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE)