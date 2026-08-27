"""
CHB-MIT EEG Preprocessing + Feature Extraction Pipeline for an edge-deployed TCN.

Pipeline stages:
    1. Load a single .edf recording with MNE, restricted to 5 bipolar channels.
    2. Band-pass filter 0.5-45 Hz.
    3. Channel-wise Z-score normalization (mean/std returned for reuse at inference).
    4. Segment into overlapping 1-second (256-sample) windows.
    5. Per window, per channel, compute 5 time/frequency-domain features:
       Hjorth Activity, Hjorth Mobility, Hjorth Complexity, Zero Crossing Rate,
       and Theta/Alpha band-power Ratio -- flattened across 5 channels into a
       single 25-element feature vector per window.
    6. Label each window as Interictal / Preictal / Ictal / Postictal using the
       seizure onset/offset timestamps parsed from the CHB-MIT summary file.

Output 'windows' array shape: (n_windows, 25) -- one feature vector per 1-second
window (NOT raw (5, 256) samples). See FEATURE_NAMES below for column order.

Usage:
    python preprocess_chbmit.py --edf chb01_03.edf --summary chb01-summary.txt --out chb01_03_features.npz

Assumptions (CHB-MIT / seizure-prediction literature conventions — adjust via CLI flags
if your project defines these windows differently):
    - Preictal  = the PREICTAL_SEC seconds immediately BEFORE seizure onset (default 30 min).
    - Ictal     = from seizure onset to seizure offset, inclusive, per the summary file.
    - Postictal = the POSTICTAL_SEC seconds immediately AFTER seizure offset (default 30 min).
    - Interictal = everything else.
    - If a recording has no seizures, every window is Interictal.
    - A window is labeled by its CENTER timestamp. If a window straddles a class
      boundary, the center-sample rule keeps every window's label unambiguous.
    - If preictal/postictal periods overlap an adjacent seizure's ictal period
      (back-to-back seizures), ictal takes priority.
"""

import argparse
import re
import numpy as np
import mne
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHANNELS = ['FP1-F7', 'F7-T7', 'FP2-F8', 'F8-T8', 'FZ-CZ']
TARGET_SFREQ = 256          # Hz, per CHB-MIT native sampling rate
BANDPASS_LOW = 0.5          # Hz
BANDPASS_HIGH = 45.0        # Hz
WINDOW_SEC = 1.0            # seconds per window
DEFAULT_OVERLAP = 0.5       # 50% overlap between consecutive windows
PREICTAL_SEC = 30 * 60      # seconds before onset labeled Preictal
POSTICTAL_SEC = 30 * 60     # seconds after offset labeled Postictal

THETA_BAND = (4.0, 8.0)     # Hz
ALPHA_BAND = (8.0, 13.0)    # Hz

# Per-channel feature order; flattened as [ch0_f0..ch0_f4, ch1_f0..ch1_f4, ...]
PER_CHANNEL_FEATURES = [
    'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
    'zero_crossing_rate', 'theta_alpha_ratio',
]
FEATURE_NAMES = [f'{ch}_{feat}' for ch in CHANNELS for feat in PER_CHANNEL_FEATURES]

LABEL_MAP = {'Interictal': 0, 'Preictal': 1, 'Ictal': 2, 'Postictal': 3}


# ---------------------------------------------------------------------------
# 1. Summary file parsing
# ---------------------------------------------------------------------------
def parse_seizure_summary(summary_path, edf_filename):
    """
    Parse a CHB-MIT '-summary.txt' file and return the seizure intervals
    (in seconds, relative to the start of the recording) for a single .edf file.

    Handles both the single-seizure format:
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    and the multi-seizure format:
        Seizure 1 Start Time: 2996 seconds
        Seizure 1 End Time: 3036 seconds
        Seizure 2 Start Time: 4000 seconds
        Seizure 2 End Time: 4050 seconds

    Returns
    -------
    list[tuple[float, float]]: (onset_sec, offset_sec) pairs, sorted by onset.
    """
    with open(summary_path, 'r') as f:
        text = f.read()

    # Split into per-file blocks anchored on "File Name:" lines.
    blocks = re.split(r'(?=File Name:\s*\S+)', text)
    target_block = None
    for block in blocks:
        m = re.search(r'File Name:\s*(\S+)', block)
        if m and m.group(1).strip() == edf_filename:
            target_block = block
            break

    if target_block is None:
        raise ValueError(
            f"'{edf_filename}' not found in summary file '{summary_path}'."
        )

    starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds', target_block)
    ends = re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds', target_block)

    if len(starts) != len(ends):
        raise ValueError(
            f"Mismatched seizure start/end count for '{edf_filename}' "
            f"({len(starts)} starts vs {len(ends)} ends)."
        )

    seizures = sorted(
        [(float(s), float(e)) for s, e in zip(starts, ends)],
        key=lambda x: x[0],
    )
    return seizures


# ---------------------------------------------------------------------------
# 2. Load + channel selection
# ---------------------------------------------------------------------------
def load_edf(edf_path, channels=CHANNELS):
    """Load an .edf file with MNE and restrict to the requested channel set."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')

    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in {edf_path}. "
            f"Available channels: {raw.ch_names}"
        )

    raw.pick(channels)
    raw.reorder_channels(channels)
    return raw


# ---------------------------------------------------------------------------
# 3. Band-pass filter
# ---------------------------------------------------------------------------
def bandpass_filter(raw, l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH):
    """Apply a zero-phase FIR band-pass filter in place and return `raw`."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', phase='zero',
               fir_design='firwin', verbose='ERROR')
    return raw


# ---------------------------------------------------------------------------
# 4. Channel-wise Z-score normalization
# ---------------------------------------------------------------------------
def zscore_normalize(data):
    """
    Channel-wise Z-score normalization, fit AND applied on the same data.

    NOTE: This computes mean/std from whatever `data` you pass in. If you
    call this per-file on a recording that happens to contain a seizure,
    the seizure's high amplitude/variance biases the mean/std used to
    normalize every window in that file -- including the Preictal windows
    right before it. For multi-file, per-patient processing, prefer
    `compute_patient_calibration()` + `zscore_apply()` instead, which fit
    fixed mean/std from interictal-only segments across all of a patient's
    recordings (matching the hardware calibration spec: fixed per-patient
    constants derived from seizure-free segments, applied uniformly).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)

    Returns
    -------
    normalized : np.ndarray, shape (n_channels, n_samples)
    mean       : np.ndarray, shape (n_channels,)
    std        : np.ndarray, shape (n_channels,)
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)  # guard against flat/dead channels
    normalized = (data - mean) / std_safe
    return normalized, mean.flatten(), std.flatten()


def zscore_apply(data, mean, std):
    """
    Channel-wise Z-score normalization using FIXED, externally-supplied
    mean/std (e.g. from compute_patient_calibration()), rather than
    computing them from `data` itself.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    mean : np.ndarray, shape (n_channels,)
    std  : np.ndarray, shape (n_channels,)

    Returns
    -------
    normalized : np.ndarray, shape (n_channels, n_samples)
    """
    mean = np.asarray(mean).reshape(-1, 1)
    std = np.asarray(std).reshape(-1, 1)
    std_safe = np.where(std == 0, 1.0, std)
    return (data - mean) / std_safe


# ---------------------------------------------------------------------------
# 4.5 Fixed per-patient calibration (from interictal segments only)
# ---------------------------------------------------------------------------
def _load_filtered_data(edf_path, channels=CHANNELS):
    """Load + resample + band-pass an .edf, returning raw (n_channels, n_samples)
    data in volts, PRE-normalization, plus the sampling rate."""
    raw = load_edf(edf_path, channels=channels)
    sfreq = raw.info['sfreq']
    if sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose='ERROR')
    bandpass_filter(raw)
    return raw.get_data(), TARGET_SFREQ


def collect_interictal_samples(edf_path, summary_path, edf_filename=None,
                                channels=CHANNELS, preictal_sec=PREICTAL_SEC,
                                postictal_sec=POSTICTAL_SEC):
    """
    Load ONE .edf file (filtered, NOT normalized) and return only the
    sample columns that fall in that file's Interictal periods -- i.e.
    excluding every seizure's Preictal/Ictal/Postictal window. If the file
    has no seizures, the entire recording is returned.

    Returns
    -------
    np.ndarray, shape (n_channels, n_interictal_samples) -- may have 0
    columns if the whole file falls inside pre/ictal/postictal windows.
    """
    if edf_filename is None:
        edf_filename = edf_path.split('/')[-1]

    data, sfreq = _load_filtered_data(edf_path, channels=channels)
    n_channels, n_samples = data.shape

    seizures = parse_seizure_summary(summary_path, edf_filename)
    if not seizures:
        return data

    sample_times = np.arange(n_samples) / sfreq
    is_interictal = np.ones(n_samples, dtype=bool)
    for onset, offset in seizures:
        preictal_start = onset - preictal_sec
        postictal_end = offset + postictal_sec
        in_event_window = (sample_times >= preictal_start) & (sample_times <= postictal_end)
        is_interictal &= ~in_event_window

    return data[:, is_interictal]


def compute_patient_calibration(edf_summary_pairs, channels=CHANNELS,
                                 preictal_sec=PREICTAL_SEC, postictal_sec=POSTICTAL_SEC,
                                 trim_pct=0.05):
    """
    Compute FIXED per-patient Z-score calibration constants from seizure-free
    (Interictal-only) segments pooled across ALL of a patient's recordings --
    per the hardware calibration spec (Block B, Phase 1): sort values, trim
    `trim_pct` from each tail to reject artifacts, then take the trimmed
    mean/std. These constants should then be reused (via zscore_apply) for
    EVERY recording of that patient, seizure or not.

    Parameters
    ----------
    edf_summary_pairs : list[(edf_path, summary_path, edf_filename_or_None)]
        Every .edf belonging to ONE patient, paired with its summary file.
    trim_pct : float
        Fraction trimmed from each tail before computing mean/std (default
        0.05 = trim 5% off each end, matching the hardware calibration doc).

    Returns
    -------
    mean : np.ndarray, shape (n_channels,)
    std  : np.ndarray, shape (n_channels,)
    n_interictal_samples : int -- total samples the calibration was fit on
        (useful to sanity-check you didn't calibrate off almost nothing)
    """
    chunks = []
    for edf_path, summary_path, edf_filename in edf_summary_pairs:
        chunk = collect_interictal_samples(
            edf_path, summary_path, edf_filename=edf_filename,
            channels=channels, preictal_sec=preictal_sec, postictal_sec=postictal_sec,
        )
        if chunk.shape[1] > 0:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "No interictal samples found across the given recordings -- "
            "cannot compute a patient calibration. Check preictal/postictal "
            "window sizes and seizure timestamps."
        )

    pooled = np.concatenate(chunks, axis=1)  # (n_channels, total_interictal_samples)
    n_channels = pooled.shape[0]

    mean = np.zeros(n_channels, dtype=np.float64)
    std = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        vals = np.sort(pooled[ch])
        n = len(vals)
        k = int(n * trim_pct)
        trimmed = vals[k: n - k] if n - 2 * k > 0 else vals
        mean[ch] = trimmed.mean()
        std[ch] = trimmed.std()

    return mean, std, pooled.shape[1]


# ---------------------------------------------------------------------------
# 5. Windowing
# ---------------------------------------------------------------------------
def segment_windows(data, sfreq=TARGET_SFREQ, window_sec=WINDOW_SEC, overlap=DEFAULT_OVERLAP):
    """
    Segment (n_channels, n_samples) data into overlapping fixed-length windows.

    Returns
    -------
    windows      : np.ndarray, shape (n_windows, n_channels, window_len)
    window_times : np.ndarray, shape (n_windows,) -- start time of each window, in seconds
    """
    window_len = int(round(window_sec * sfreq))
    step = int(round(window_len * (1 - overlap)))
    step = max(step, 1)

    n_channels, n_samples = data.shape
    starts = np.arange(0, n_samples - window_len + 1, step)

    windows = np.stack([data[:, s:s + window_len] for s in starts], axis=0)
    window_times = starts / sfreq
    return windows, window_times


# ---------------------------------------------------------------------------
# 5.5 Per-window feature extraction
# ---------------------------------------------------------------------------
def _hjorth_params(sig):
    """Hjorth Activity (variance), Mobility, and Complexity for a 1D signal."""
    activity = np.var(sig)
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)

    mobility = np.sqrt(var_d1 / activity) if activity > 0 else 0.0
    mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0.0
    complexity = (mobility_d1 / mobility) if mobility > 0 else 0.0
    return activity, mobility, complexity


def _zero_crossing_rate(sig):
    """Fraction of adjacent-sample sign changes in the signal."""
    signs = np.sign(sig)
    signs[signs == 0] = 1  # treat exact zeros as positive to avoid spurious crossings
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return crossings / len(sig)


def _theta_alpha_ratio(sig, sfreq, theta_band=THETA_BAND, alpha_band=ALPHA_BAND):
    """Ratio of Theta (4-8 Hz) to Alpha (8-13 Hz) band power, via Welch's PSD."""
    nperseg = min(len(sig), 256)
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)

    theta_mask = (freqs >= theta_band[0]) & (freqs < theta_band[1])
    alpha_mask = (freqs >= alpha_band[0]) & (freqs < alpha_band[1])

    theta_power = np.sum(psd[theta_mask])
    alpha_power = np.sum(psd[alpha_mask])
    return theta_power / alpha_power if alpha_power > 0 else 0.0


def extract_window_features(window, sfreq=TARGET_SFREQ):
    """
    Compute the 5 per-channel features for one window and flatten across channels.

    Parameters
    ----------
    window : np.ndarray, shape (n_channels, window_len)

    Returns
    -------
    np.ndarray, shape (n_channels * 5,)
    """
    feats = []
    for ch_idx in range(window.shape[0]):
        sig = window[ch_idx]
        activity, mobility, complexity = _hjorth_params(sig)
        zcr = _zero_crossing_rate(sig)
        ta_ratio = _theta_alpha_ratio(sig, sfreq)
        feats.extend([activity, mobility, complexity, zcr, ta_ratio])
    return np.array(feats, dtype=np.float32)


def extract_features(windows, sfreq=TARGET_SFREQ):
    """
    Vectorize extract_window_features() over a batch of windows.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_len)

    Returns
    -------
    np.ndarray, shape (n_windows, n_channels * 5)
    """
    return np.stack([extract_window_features(w, sfreq=sfreq) for w in windows], axis=0)


# ---------------------------------------------------------------------------
# 6. Labeling
# ---------------------------------------------------------------------------
def label_windows(window_times, window_sec, seizures,
                   preictal_sec=PREICTAL_SEC, postictal_sec=POSTICTAL_SEC):
    """
    Assign one of {Interictal, Preictal, Ictal, Postictal} to each window,
    based on the window's center timestamp.

    Parameters
    ----------
    window_times : np.ndarray  -- start time (sec) of each window
    window_sec   : float       -- window duration in seconds
    seizures     : list[(onset_sec, offset_sec)]

    Returns
    -------
    labels     : np.ndarray[str], shape (n_windows,)
    label_ids  : np.ndarray[int], shape (n_windows,)  -- via LABEL_MAP
    """
    centers = window_times + window_sec / 2.0
    labels = np.full(centers.shape, 'Interictal', dtype=object)

    for onset, offset in seizures:
        preictal_start = onset - preictal_sec
        postictal_end = offset + postictal_sec

        is_preictal = (centers >= preictal_start) & (centers < onset)
        is_ictal = (centers >= onset) & (centers <= offset)
        is_postictal = (centers > offset) & (centers <= postictal_end)

        # Ictal takes priority over any conflicting pre/postictal assignment
        # from a neighboring seizure (back-to-back seizure case).
        labels[is_preictal & (labels != 'Ictal')] = 'Preictal'
        labels[is_postictal & (labels != 'Ictal')] = 'Postictal'
        labels[is_ictal] = 'Ictal'

    label_ids = np.array([LABEL_MAP[l] for l in labels])
    return labels, label_ids


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------
def preprocess_recording(edf_path, summary_path, edf_filename=None,
                          channels=CHANNELS, overlap=DEFAULT_OVERLAP,
                          preictal_sec=PREICTAL_SEC, postictal_sec=POSTICTAL_SEC,
                          calibration_mean=None, calibration_std=None):
    """
    Run the full pipeline on a single .edf recording.

    Parameters
    ----------
    calibration_mean, calibration_std : np.ndarray, shape (n_channels,), optional
        Fixed per-patient calibration constants from
        compute_patient_calibration(), fit on interictal-only segments
        pooled across the patient's recordings. If provided, both must be
        given together and normalization uses zscore_apply() with these
        fixed values (recommended -- matches the hardware calibration spec).
        If omitted, falls back to the OLD per-file behavior: mean/std are
        computed from this file alone via zscore_normalize(), which biases
        the calibration if this particular file contains a seizure. This
        fallback exists for standalone single-file CLI use only; multi-file
        pipelines (build_train_test.py) should always pass calibration
        constants explicitly.

    Returns a dict with windows, integer + string labels, normalization
    stats, and metadata -- ready to hand to a TCN training loop.
    """
    if (calibration_mean is None) != (calibration_std is None):
        raise ValueError("Provide both calibration_mean and calibration_std, or neither.")

    if edf_filename is None:
        edf_filename = edf_path.split('/')[-1]

    raw = load_edf(edf_path, channels=channels)

    sfreq = raw.info['sfreq']
    if sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose='ERROR')

    bandpass_filter(raw)

    data = raw.get_data()  # (n_channels, n_samples), volts

    if calibration_mean is not None:
        ch_mean, ch_std = np.asarray(calibration_mean), np.asarray(calibration_std)
        data = zscore_apply(data, ch_mean, ch_std)
    else:
        print(f"  [preprocess_recording] WARNING: no calibration_mean/std passed for "
              f"{edf_filename} -- falling back to per-file z-score (biased if this "
              f"file contains a seizure). Pass fixed per-patient calibration instead.")
        data, ch_mean, ch_std = zscore_normalize(data)

    raw_windows, window_times = segment_windows(data, sfreq=TARGET_SFREQ, overlap=overlap)
    features = extract_features(raw_windows, sfreq=TARGET_SFREQ)  # (n_windows, 25)

    seizures = parse_seizure_summary(summary_path, edf_filename)
    labels, label_ids = label_windows(
        window_times, WINDOW_SEC, seizures,
        preictal_sec=preictal_sec, postictal_sec=postictal_sec,
    )

    return {
        'windows': features.astype(np.float32),        # (n_windows, 25) -- feature vectors, NOT raw samples
        'labels': labels,                                # string labels
        'label_ids': label_ids,                           # int labels per LABEL_MAP
        'window_times': window_times,                      # start time (s) of each window
        'channel_mean': ch_mean,                            # (n_channels,) pre-normalization mean
        'channel_std': ch_std,                               # (n_channels,) pre-normalization std
        'channels': channels,
        'feature_names': FEATURE_NAMES,                       # column order for the 25-element vector
        'seizures': seizures,
        'label_map': LABEL_MAP,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess a CHB-MIT .edf recording for TCN input.")
    parser.add_argument('--edf', required=True, help="Path to the .edf file")
    parser.add_argument('--summary', required=True, help="Path to the CHB-MIT *-summary.txt file")
    parser.add_argument('--out', required=True, help="Output .npz path")
    parser.add_argument('--overlap', type=float, default=DEFAULT_OVERLAP,
                         help="Fractional overlap between consecutive windows (default 0.5)")
    parser.add_argument('--preictal-min', type=float, default=PREICTAL_SEC / 60,
                         help="Minutes before onset labeled Preictal (default 30)")
    parser.add_argument('--postictal-min', type=float, default=POSTICTAL_SEC / 60,
                         help="Minutes after offset labeled Postictal (default 30)")
    args = parser.parse_args()

    result = preprocess_recording(
        args.edf, args.summary,
        overlap=args.overlap,
        preictal_sec=args.preictal_min * 60,
        postictal_sec=args.postictal_min * 60,
    )

    np.savez_compressed(
        args.out,
        windows=result['windows'],
        label_ids=result['label_ids'],
        labels=result['labels'],
        window_times=result['window_times'],
        channel_mean=result['channel_mean'],
        channel_std=result['channel_std'],
        channels=np.array(result['channels']),
        feature_names=np.array(result['feature_names']),
    )

    unique, counts = np.unique(result['labels'], return_counts=True)
    print(f"Saved {result['windows'].shape[0]} windows of shape {result['windows'].shape[1:]} to {args.out}")
    print("Label distribution:", dict(zip(unique, counts)))


if __name__ == '__main__':
    main()