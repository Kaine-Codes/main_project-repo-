"""
Build TRAIN/VAL sets from a SINGLE patient's recordings, chronologically
split ~80/20, with a hard guarantee that at least one seizure (Ictal-labeled
sequence) lands in the VAL split.

Why this needs its own script instead of just build_train_test.py +
split_patient.py: with many patients pooled, a plain chronological/group-
aware split is very likely to have seizures on both sides just by volume.
With ONE patient, seizures are rare and cluster in a handful of files -- a
plain 80/20 cut can easily land in a stretch of files that has none at all,
giving you a "test set" that can't actually evaluate seizure detection.

Strategy:
    1. Process every .edf in the patient's folder (reusing build_train_test.py's
       process_subject_dir, which computes ONE fixed interictal-only Z-score
       calibration for this patient and applies it to every file).
    2. Take the chronological suffix (last N files) whose sequence count is
       closest to the requested val ratio -- same philosophy as
       split_patient.py: val = the "most recent" chunk of the recording.
    3. Check whether that suffix already contains a seizure (any Ictal-
       labeled sequence). If yes, done.
    4. If not, force the SMALLEST seizure-containing file into val (pulled
       out of whatever position it's in), which guarantees the requirement
       while disturbing the target ratio as little as possible.

Usage:
    python build_single_patient_split.py --patient_dir "CHB DATASET/chb06" \
        --train_out chb06_train.npz --val_out chb06_val.npz \
        --ratio 0.8 --seq_len 64 --seq_stride 8
"""

import argparse
import numpy as np

from build_train_test import process_subject_dir

INTERICTAL, PREICTAL, ICTAL, POSTICTAL = 0, 1, 2, 3


def choose_split_groups(groups, y, val_ratio):
    """
    Returns (train_group_ids, val_group_ids, achieved_val_ratio, forced_seizure).
    val_group_ids is the chronological suffix closest to val_ratio, expanded
    by exactly one file if needed to guarantee a seizure is present.
    """
    unique_groups_in_order = groups[np.sort(np.unique(groups, return_index=True)[1])]
    total = len(groups)

    counts_by_group = {g: int(np.sum(groups == g)) for g in unique_groups_in_order}
    has_seizure_by_group = {g: bool(np.any(y[groups == g] == ICTAL)) for g in unique_groups_in_order}

    # 1) Best chronological suffix cut for the target ratio.
    n_groups = len(unique_groups_in_order)
    best_start_idx = n_groups - 1
    best_diff = float('inf')
    cum = 0
    # walk from the end backwards, growing the suffix
    for i in range(n_groups - 1, -1, -1):
        cum += counts_by_group[unique_groups_in_order[i]]
        diff = abs(cum / total - val_ratio)
        if diff < best_diff:
            best_diff = diff
            best_start_idx = i

    val_group_ids = list(unique_groups_in_order[best_start_idx:])
    forced_seizure = False

    # 2) Guarantee a seizure is present in val.
    if not any(has_seizure_by_group[g] for g in val_group_ids):
        seizure_groups = [g for g in unique_groups_in_order if has_seizure_by_group[g]]
        if not seizure_groups:
            raise RuntimeError(
                "This patient's data contains NO seizure-labeled (Ictal) sequences "
                "at all -- cannot guarantee a seizure in val. Check seq_stride "
                "isn't so large it's skipping over the short Ictal windows, or "
                "check the summary.txt parsing."
            )
        # pick smallest seizure-containing group to minimize ratio distortion
        smallest_seizure_group = min(seizure_groups, key=lambda g: counts_by_group[g])
        val_group_ids.append(smallest_seizure_group)
        forced_seizure = True

    val_group_ids = sorted(set(val_group_ids))
    train_group_ids = [g for g in unique_groups_in_order if g not in val_group_ids]
    achieved_ratio = sum(counts_by_group[g] for g in val_group_ids) / total

    return train_group_ids, val_group_ids, achieved_ratio, forced_seizure


def save_split(X, y, feature_names, groups, group_ids, out_path):
    mask = np.isin(groups, list(group_ids))
    idx = np.where(mask)[0]
    save_kwargs = dict(
        windows=X[idx], features=X[idx], labels=y[idx], groups=groups[idx],
        feature_names=np.array(feature_names),
    )
    np.savez_compressed(out_path, **save_kwargs)
    unique, counts = np.unique(y[idx], return_counts=True)
    label_names = ["Interictal", "Preictal", "Ictal", "Postictal"]
    dist = {label_names[u]: int(c) for u, c in zip(unique, counts)}
    print(f"  Saved {len(idx)} sequences to {out_path}")
    print(f"    Label distribution: {dist}")
    return dist


def main():
    parser = argparse.ArgumentParser(
        description="Single-patient 80/20 split with a guaranteed seizure in val."
    )
    parser.add_argument('--patient_dir', required=True,
                         help="e.g. 'CHB DATASET/chb06'")
    parser.add_argument('--train_out', default='chb06_train.npz')
    parser.add_argument('--val_out', default='chb06_val.npz')
    parser.add_argument('--ratio', type=float, default=0.8,
                         help="Fraction of sequences targeted for TRAIN (default 0.8)")
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--seq_stride', type=int, default=8)
    args = parser.parse_args()

    print(f"Processing {args.patient_dir} (fixed per-patient calibration, "
          f"seq_len={args.seq_len}, seq_stride={args.seq_stride}) ...")
    result = process_subject_dir(args.patient_dir, args.seq_len, seq_stride=args.seq_stride)
    if result is None:
        raise RuntimeError(f"Nothing usable found in {args.patient_dir}")
    X, y, feature_names, groups = result

    n_files = len(set(groups.tolist()))
    n_seizure_files = len({g for g in set(groups.tolist()) if np.any(y[groups == g] == ICTAL)})
    print(f"\n{X.shape[0]} total sequences across {n_files} files "
          f"({n_seizure_files} contain at least one Ictal-labeled sequence).")

    val_ratio = 1.0 - args.ratio
    train_ids, val_ids, achieved_val_ratio, forced = choose_split_groups(groups, y, val_ratio)

    print(f"\nSplit: {len(train_ids)} files -> train, {len(val_ids)} files -> val")
    print(f"  Target val ratio: {val_ratio:.1%} | Achieved: {achieved_val_ratio:.1%}")
    if forced:
        print("  NOTE: the natural chronological cut had no seizure in val, so the "
              "smallest seizure-containing file was pulled into val to satisfy the "
              "guarantee. Val is therefore not a perfectly contiguous 'tail' anymore.")
    else:
        print("  Val already contained a seizure naturally -- no adjustment needed.")

    print("\nTrain split:")
    save_split(X, y, feature_names, groups, train_ids, args.train_out)
    print("\nVal split:")
    val_dist = save_split(X, y, feature_names, groups, val_ids, args.val_out)

    assert val_dist.get("Ictal", 0) > 0, "Sanity check failed: no Ictal sequences in val!"
    print(f"\nConfirmed: val split contains {val_dist['Ictal']} Ictal sequence(s).")


if __name__ == '__main__':
    main()