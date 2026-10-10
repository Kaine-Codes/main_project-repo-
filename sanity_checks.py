"""
sanity_checks.py
================
Automated validation checks for the EEG classification pipeline.

Every check either passes silently or raises AssertionError with a descriptive
message.  Call run_all_data_checks() after building train/val/test partitions
to catch bugs before training.
"""

import numpy as np
from typing import List, Optional, Dict

VALID_LABELS = {0, 1, 2, 3}
LABEL_NAMES = {0: "Interictal", 1: "Preictal", 2: "Ictal", 3: "Postictal"}


# ────────────────────────────────────────────────────────────────────────── #
# Individual checks
# ────────────────────────────────────────────────────────────────────────── #

def check_labels_valid(y: np.ndarray, partition: str = ""):
    """Assert all label values are in {0, 1, 2, 3}."""
    unique = set(np.unique(y).tolist())
    invalid = unique - VALID_LABELS
    assert not invalid, (
        f"[{partition}] Invalid label value(s): {invalid}. "
        f"Expected only {VALID_LABELS}."
    )


def check_no_edf_overlap(train_files: List[str], val_files: List[str],
                          test_files: List[str]):
    """Assert no EDF filename appears in more than one partition."""
    t, v, te = set(train_files), set(val_files), set(test_files)
    tv = t & v
    tt = t & te
    vt = v & te
    assert not tv, f"EDFs in both TRAIN and VAL: {tv}"
    assert not tt, f"EDFs in both TRAIN and TEST: {tt}"
    assert not vt, f"EDFs in both VAL and TEST: {vt}"


def check_calibration_source(calib_files: List[str], val_files: List[str],
                              test_files: List[str]):
    """Assert calibration μ/σ was NOT computed from val/test files."""
    calib = set(calib_files)
    val_leak = calib & set(val_files)
    test_leak = calib & set(test_files)
    assert not val_leak, (
        f"Calibration uses VAL files (leakage!): {val_leak}"
    )
    assert not test_leak, (
        f"Calibration uses TEST files (leakage!): {test_leak}"
    )


def check_no_nan_inf(data: np.ndarray, name: str = "data"):
    """Assert no NaN or Inf values."""
    assert not np.any(np.isnan(data)), f"NaN found in {name}"
    assert not np.any(np.isinf(data)), f"Inf found in {name}"


def check_std_not_zero(std: np.ndarray, name: str = "calibration_std"):
    """Assert no zero standard deviation (would cause division by zero)."""
    zero_channels = np.where(std == 0)[0]
    assert len(zero_channels) == 0, (
        f"Zero standard deviation in {name} at channel indices: "
        f"{zero_channels.tolist()}. This would produce Inf during normalization."
    )


def check_ictal_exists(y: np.ndarray, partition: str = ""):
    """Warn (not assert) if no Ictal sequences exist — expected for some partitions."""
    n_ictal = int(np.sum(y == 2))
    if n_ictal == 0:
        print(f"  ⚠ WARNING [{partition}]: No Ictal sequences found. "
              f"This may be expected for non-seizure partitions.")
    return n_ictal


def check_seizure_count(found_count: int, expected_count: int,
                        patient_id: str = ""):
    """Assert the number of seizures matches expectations."""
    assert found_count == expected_count, (
        f"[{patient_id}] Seizure count mismatch: found {found_count}, "
        f"expected {expected_count}. Check summary parsing."
    )


def check_distribution_unchanged(y_original: np.ndarray, y_current: np.ndarray,
                                  partition: str = ""):
    """Assert a partition's label distribution was NOT modified
    (i.e., val/test were not accidentally undersampled)."""
    orig_counts = np.bincount(y_original, minlength=4)
    curr_counts = np.bincount(y_current, minlength=4)
    assert np.array_equal(orig_counts, curr_counts), (
        f"[{partition}] Distribution was modified!\n"
        f"  Original: {dict(enumerate(orig_counts.tolist()))}\n"
        f"  Current:  {dict(enumerate(curr_counts.tolist()))}\n"
        f"  Val/test should NEVER be undersampled."
    )


def check_class_weights_disabled(use_class_weights: bool, experiment: str = "1"):
    """Assert class weights are disabled for the first experiment."""
    assert not use_class_weights, (
        f"Class weights are ENABLED for Experiment {experiment}. "
        f"The first experiment should use undersampling only (no class weights)."
    )


def check_model_output_classes(model, num_classes: int = 4):
    """Assert the model's final layer outputs the expected number of classes."""
    import torch
    with torch.no_grad():
        dummy = torch.randn(1, 25, 64)
        if next(model.parameters()).is_cuda:
            dummy = dummy.cuda()
        out = model(dummy)
    assert out.shape[-1] == num_classes, (
        f"Model output has {out.shape[-1]} classes, expected {num_classes}."
    )


# ────────────────────────────────────────────────────────────────────────── #
# Composite runner
# ────────────────────────────────────────────────────────────────────────── #

def run_all_data_checks(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    calib_files: List[str],
    calib_mean: np.ndarray,
    calib_std: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    use_class_weights: bool = False,
):
    """Run all data-pipeline sanity checks. Raises AssertionError on failure."""
    print("\n=== Sanity Checks ===")
    passed = 0
    total = 0

    checks = [
        ("Labels valid (train)",
         lambda: check_labels_valid(y_train, "TRAIN")),
        ("Labels valid (val)",
         lambda: check_labels_valid(y_val, "VAL")),
        ("Labels valid (test)",
         lambda: check_labels_valid(y_test, "TEST")),
        ("No EDF in multiple partitions",
         lambda: check_no_edf_overlap(train_files, val_files, test_files)),
        ("Calibration uses training data only",
         lambda: check_calibration_source(calib_files, val_files, test_files)),
        ("Calibration mean not NaN/Inf",
         lambda: check_no_nan_inf(calib_mean, "calibration_mean")),
        ("Calibration std not NaN/Inf",
         lambda: check_no_nan_inf(calib_std, "calibration_std")),
        ("Calibration std not zero",
         lambda: check_std_not_zero(calib_std)),
    ]

    if X_train is not None:
        checks.append(("Train data not NaN/Inf",
                        lambda: check_no_nan_inf(X_train, "X_train")))
    if X_val is not None:
        checks.append(("Val data not NaN/Inf",
                        lambda: check_no_nan_inf(X_val, "X_val")))
    if X_test is not None:
        checks.append(("Test data not NaN/Inf",
                        lambda: check_no_nan_inf(X_test, "X_test")))

    for name, check_fn in checks:
        total += 1
        try:
            check_fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")

    # Ictal existence (warning, not assertion)
    for partition, y in [("TRAIN", y_train), ("VAL", y_val), ("TEST", y_test)]:
        check_ictal_exists(y, partition)

    print(f"\n  {passed}/{total} checks passed.")
    if passed < total:
        raise AssertionError(
            f"SANITY CHECK FAILURE: {total - passed} check(s) failed. "
            f"Fix the issues above before proceeding."
        )
    print("  All sanity checks passed. ✓\n")
