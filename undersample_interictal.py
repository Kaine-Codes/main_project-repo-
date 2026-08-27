"""
Undersample the Interictal class in a training .npz to reduce class imbalance.

Only run this on TRAINING data (e.g. chb10_train.npz).
Never run this on TEST data (e.g. chb08_test.npz) -- the test set should keep
its real-world distribution so your evaluation metrics stay meaningful.

Usage:
    python undersample_interictal.py --in chb10_train.npz --out chb10_train_balanced.npz --target 10800
"""

import argparse
import numpy as np

INTERICTAL_LABEL = 0  # per LABEL_MAP in preprocess_chbmit.py


def undersample_interictal(npz_path, out_path, target_count, seed=42):
    d = np.load(npz_path)
    X, y = d['windows'], d['labels']
    feature_names = d['feature_names'] if 'feature_names' in d else None

    rng = np.random.default_rng(seed)

    interictal_idx = np.where(y == INTERICTAL_LABEL)[0]
    other_idx = np.where(y != INTERICTAL_LABEL)[0]

    if target_count >= len(interictal_idx):
        print(f"target_count ({target_count}) >= current Interictal count "
              f"({len(interictal_idx)}); nothing to remove.")
        keep_interictal_idx = interictal_idx
    else:
        keep_interictal_idx = rng.choice(interictal_idx, size=target_count, replace=False)

    keep_idx = np.concatenate([keep_interictal_idx, other_idx])
    rng.shuffle(keep_idx)

    X_bal, y_bal = X[keep_idx], y[keep_idx]

    save_kwargs = dict(windows=X_bal, features=X_bal, labels=y_bal)
    if feature_names is not None:
        save_kwargs['feature_names'] = feature_names
    np.savez_compressed(out_path, **save_kwargs)

    unique, counts = np.unique(y_bal, return_counts=True)
    print(f"Saved {X_bal.shape[0]} windows to {out_path}")
    print("New label distribution:", dict(zip(unique.tolist(), counts.tolist())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Undersample Interictal class in a training .npz")
    parser.add_argument('--in', dest='in_path', required=True, help="Input .npz (training set only)")
    parser.add_argument('--out', dest='out_path', required=True, help="Output .npz path")
    parser.add_argument('--target', type=int, required=True,
                         help="Target Interictal count (e.g. match your Preictal count)")
    args = parser.parse_args()

    undersample_interictal(args.in_path, args.out_path, args.target)
