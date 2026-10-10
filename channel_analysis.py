"""
=============================================================
CHB08_05 - 23-channel EEG feature-evolution analysis
=============================================================

Analyzes chb08_05.edf using all 23 EEG channels.

Segments:
    interictal : 60s - 660s
    preictal   : 2256s - 2856s
    ictal      : 2856s - 3046s
    postictal  : 3046s - min(4246s, file_end)

Features computed per 1s window (50% overlap = 0.5s step),
per channel:

    1. Hjorth Activity
    2. Hjorth Mobility
    3. Hjorth Complexity
    4. Zero Crossing Rate (ZCR)
    5. Alpha/Theta band-power ratio
    6. Higuchi Fractal Dimension
    7. Gamma/Beta band-power ratio
    8. Theta/Beta band-power ratio

Preprocessing:
    4th-order Butterworth bandpass 0.5-45 Hz

Outputs:
    chb08_05_features.xlsx
    plots/<feature>.png

Requirements:
    pip install mne numpy pandas scipy matplotlib openpyxl
"""

# =============================================================
# IMPORTS
# =============================================================

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch

warnings.filterwarnings("ignore")


# =============================================================
# CONFIG
# =============================================================

FS = 256  # Expected sampling frequency in Hz

SEIZURE_START = 2856.0
SEIZURE_END = 3046.0
FILE_DURATION = 3600.0

PREICTAL_SEC = 10 * 60
POSTICTAL_SEC = 20 * 60

INTERICTAL_START = 60.0
INTERICTAL_DUR = 10 * 60


# =============================================================
# 23 CHANNELS
# =============================================================
#
# IMPORTANT:
# The EDF contains TWO channels named "T8-P8":
#
# EDF Channel 15 -> T8-P8
# EDF Channel 23 -> T8-P8
#
# We therefore list T8-P8 twice here.
#
# The loading function below automatically maps:
#
# first T8-P8  -> EDF channel 15
# second T8-P8 -> EDF channel 23
#
# and gives them unique names:
#
# T8-P8_1
# T8-P8_2
#
# =============================================================

CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",

    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",

    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",

    "FP2-F8",
    "F8-T8",
    "T8-P8-0",
    "P8-O2",

    "FZ-CZ",
    "CZ-PZ",

    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8-1",
]


# =============================================================
# FEATURE / SIGNAL CONFIG
# =============================================================

WIN_SEC = 1.0
OVERLAP = 0.5

STEP_SEC = WIN_SEC * (1 - OVERLAP)

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
BANDPASS_ORDER = 4

FEATURE_NAMES = [
    "Hjorth_Activity",
    "Hjorth_Mobility",
    "Hjorth_Complexity",
    "ZCR",
    "Alpha_Theta",
    "Higuchi_FD",
    "Gamma_Beta",
    "Theta_Beta",
]

SEGMENT_COLORS = {
    "interictal": "#9e9e9e",
    "preictal": "#ff9800",
    "ictal": "#e53935",
    "postictal": "#1e88e5",
}

SEGMENT_ORDER = [
    "interictal",
    "preictal",
    "ictal",
    "postictal",
]


# =============================================================
# SIGNAL LOADING
# =============================================================

def load_edf_channels(edf_path, channel_names):
    """Load requested channels from an EDF file using MNE."""

    import mne

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    all_names = raw.ch_names

    print("\nChannels detected by MNE:")
    for i, name in enumerate(all_names):
        print(f"Channel {i + 1}: {name}")

    picked_idx = []
    picked_names = []

    for name in channel_names:

        matches = [
            i for i, n in enumerate(all_names)
            if n.strip().upper() == name.strip().upper()
        ]

        if not matches:
            raise ValueError(
                f"Channel '{name}' not found in EDF. "
                f"Available: {all_names}"
            )

        idx = matches[0]

        picked_idx.append(idx)
        picked_names.append(name)

        print(
            f"Selected: {name} "
            f"-> EDF Channel {idx + 1}"
        )

    data = raw.get_data(
        picks=picked_idx
    )

    # Convert volts -> microvolts
    data = data * 1e6

    fs = raw.info["sfreq"]

    if abs(fs - FS) > 1e-6:
        print(
            f"[WARNING] EDF sampling rate {fs} Hz "
            f"differs from expected {FS} Hz; "
            f"using {fs} Hz."
        )

    return data, picked_names, fs
# =============================================================
# PREPROCESSING
# =============================================================

def bandpass_filter(
    x,
    fs,
    low=BANDPASS_LOW,
    high=BANDPASS_HIGH,
    order=BANDPASS_ORDER
):
    """
    Apply Butterworth bandpass filter.
    """

    nyq = fs / 2.0

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype="band"
    )

    return filtfilt(b, a, x)


# =============================================================
# FEATURE FUNCTIONS
# =============================================================

def hjorth_params(x):
    """
    Calculate Hjorth Activity, Mobility and Complexity.
    """

    var0 = np.var(x)

    dx = np.diff(x)
    var1 = np.var(dx)

    ddx = np.diff(dx)
    var2 = np.var(ddx)

    activity = var0

    mobility = (
        np.sqrt(var1 / var0)
        if var0 > 0
        else 0.0
    )

    mobility_d1 = (
        np.sqrt(var2 / var1)
        if var1 > 0
        else 0.0
    )

    complexity = (
        mobility_d1 / mobility
        if mobility > 0
        else 0.0
    )

    return activity, mobility, complexity


# -------------------------------------------------------------

def zero_crossing_rate(x):
    """
    Calculate Zero Crossing Rate.
    """

    signs = x >= 0

    crossings = np.sum(
        signs[:-1] != signs[1:]
    )

    return crossings / (len(x) - 1)


# -------------------------------------------------------------

def higuchi_fd(x, kmax=6):
    """
    Higuchi Fractal Dimension.

    kmax = 6
    """

    x = np.asarray(x, dtype=float)

    N = len(x)

    Lk = np.zeros(kmax)

    for k in range(1, kmax + 1):

        Lmk_vals = []

        for m in range(1, k + 1):

            n_max = int(
                np.floor((N - m) / k)
            )

            if n_max < 1:
                continue

            idx = (
                m - 1
                + np.arange(1, n_max + 1) * k
            )

            idx_prev = idx - k

            Lmki = np.sum(
                np.abs(
                    x[idx] - x[idx_prev]
                )
            )

            norm = (
                (N - 1)
                / (n_max * k)
            )

            Lmk_vals.append(
                Lmki * norm / k
            )

        if Lmk_vals:
            Lk[k - 1] = np.mean(Lmk_vals)
        else:
            Lk[k - 1] = np.nan

    valid = (
        ~np.isnan(Lk)
        & (Lk > 0)
    )

    if valid.sum() < 2:
        return np.nan

    lnLk = np.log(Lk[valid])

    lnk = np.log(
        1.0 / np.arange(1, kmax + 1)
    )[valid]

    slope, _ = np.polyfit(
        lnk,
        lnLk,
        1
    )

    return slope


# -------------------------------------------------------------

def band_power(x, fs, band):
    """
    Calculate power in a frequency band
    using Welch's PSD estimate.
    """

    nperseg = min(
        len(x),
        int(fs)
    )

    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg
    )

    idx = (
        (freqs >= band[0])
        & (freqs <= band[1])
    )

    if not np.any(idx):
        return 0.0

    return np.trapezoid(
        psd[idx],
        freqs[idx]
    )


# =============================================================
# FEATURE EXTRACTION
# =============================================================

def extract_window_features(x, fs):
    """
    Calculate all 8 features for one EEG window.
    """

    # -------------------------
    # Hjorth parameters
    # -------------------------

    activity, mobility, complexity = hjorth_params(x)

    # -------------------------
    # Zero Crossing Rate
    # -------------------------

    zcr = zero_crossing_rate(x)

    # -------------------------
    # Higuchi Fractal Dimension
    # -------------------------

    hfd = higuchi_fd(
        x,
        kmax=6
    )

    # -------------------------
    # Frequency band powers
    # -------------------------

    p_theta = band_power(
        x,
        fs,
        BANDS["theta"]
    )

    p_alpha = band_power(
        x,
        fs,
        BANDS["alpha"]
    )

    p_beta = band_power(
        x,
        fs,
        BANDS["beta"]
    )

    p_gamma = band_power(
        x,
        fs,
        BANDS["gamma"]
    )

    # -------------------------
    # Ratios
    # -------------------------

    alpha_theta = (
        p_alpha / p_theta
        if p_theta > 0
        else np.nan
    )

    gamma_beta = (
        p_gamma / p_beta
        if p_beta > 0
        else np.nan
    )

    theta_beta = (
        p_theta / p_beta
        if p_beta > 0
        else np.nan
    )

    # -------------------------
    # Return all features
    # -------------------------

    return {
        "Hjorth_Activity": activity,
        "Hjorth_Mobility": mobility,
        "Hjorth_Complexity": complexity,
        "ZCR": zcr,
        "Alpha_Theta": alpha_theta,
        "Higuchi_FD": hfd,
        "Gamma_Beta": gamma_beta,
        "Theta_Beta": theta_beta,
    }


# =============================================================
# SEGMENTATION
# =============================================================

def get_segments():
    """
    Return dictionary:

        segment_name -> (start_time, end_time)

    Postictal segment is clipped to the actual file duration.
    """

    segs = {

        "interictal": (
            INTERICTAL_START,
            INTERICTAL_START + INTERICTAL_DUR
        ),

        "preictal": (
            SEIZURE_START - PREICTAL_SEC,
            SEIZURE_START
        ),

        "ictal": (
            SEIZURE_START,
            SEIZURE_END
        ),

        "postictal": (
            SEIZURE_END,
            min(
                SEIZURE_END + POSTICTAL_SEC,
                FILE_DURATION
            )
        ),
    }

    requested_postictal_end = (
        SEIZURE_END + POSTICTAL_SEC
    )

    if requested_postictal_end > FILE_DURATION:

        actual_duration = (
            segs["postictal"][1]
            - segs["postictal"][0]
        )

        print(
            f"\n[WARNING] Requested postictal window "
            f"would end at "
            f"{requested_postictal_end:.0f}s."
        )

        print(
            f"But the EDF ends at "
            f"{FILE_DURATION:.0f}s."
        )

        print(
            f"Postictal clipped to "
            f"{actual_duration:.0f}s "
            f"({actual_duration / 60:.1f} min).\n"
        )

    return segs


# =============================================================
# MAIN FEATURE EXTRACTION PIPELINE
# =============================================================

def process_file(edf_path):

    print("=" * 70)
    print("CHB08_05 EEG FEATURE ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load EDF
    # ---------------------------------------------------------

    print(f"\nLoading: {edf_path}")

    data, ch_names, fs = load_edf_channels(
        edf_path,
        CHANNELS
    )

    n_channels, n_samples = data.shape

    print(
        f"\nLoaded {n_channels} channels, "
        f"{n_samples} samples @ {fs} Hz"
    )

    print(
        f"Recording duration: "
        f"{n_samples / fs:.1f} seconds"
    )

    # ---------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------

    print(
        "\nApplying "
        f"{BANDPASS_LOW}-{BANDPASS_HIGH} Hz "
        f"Butterworth bandpass filter..."
    )

    filtered = np.array([
        bandpass_filter(
            data[i],
            fs
        )
        for i in range(n_channels)
    ])

    print("Filtering complete.")

    # ---------------------------------------------------------
    # Segments
    # ---------------------------------------------------------

    segments = get_segments()

    print("\nSegments:")
    print("-" * 70)

    for seg_name in SEGMENT_ORDER:

        t_start, t_end = segments[seg_name]

        print(
            f"{seg_name:<12}: "
            f"{t_start:7.1f}s - "
            f"{t_end:7.1f}s "
            f"({(t_end - t_start) / 60:.2f} min)"
        )

    # ---------------------------------------------------------
    # Window configuration
    # ---------------------------------------------------------

    win_samples = int(
        round(WIN_SEC * fs)
    )

    step_samples = int(
        round(STEP_SEC * fs)
    )

    print("\nWindow configuration:")
    print(f"  Window length : {WIN_SEC} sec")
    print(f"  Overlap       : {OVERLAP * 100:.0f}%")
    print(f"  Step          : {STEP_SEC} sec")
    print(f"  Window samples: {win_samples}")
    print(f"  Step samples  : {step_samples}")

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    rows = []

    print("\nStarting feature extraction...")
    print("=" * 70)

    for seg_name in SEGMENT_ORDER:

        t_start, t_end = segments[seg_name]

        if t_end <= t_start:

            print(
                f"[WARNING] Segment '{seg_name}' "
                f"has zero/negative duration. Skipping."
            )

            continue

        s_start = int(
            round(t_start * fs)
        )

        s_end = int(
            round(t_end * fs)
        )

        s_end = min(
            s_end,
            n_samples
        )

        n_windows = (
            (s_end - s_start - win_samples)
            // step_samples
        ) + 1

        print(
            f"\nSegment: {seg_name}"
        )

        print(
            f"Time: {t_start:.1f}s - "
            f"{t_end:.1f}s"
        )

        print(
            f"Expected windows: {n_windows}"
        )

        window_counter = 0

        for w_start in range(
            s_start,
            s_end - win_samples + 1,
            step_samples
        ):

            w_end = (
                w_start
                + win_samples
            )

            win_time_s = (
                w_start / fs
            )

            # -------------------------------------------------
            # Process every channel
            # -------------------------------------------------

            for ci, ch_name in enumerate(ch_names):

                x = filtered[
                    ci,
                    w_start:w_end
                ]

                feats = extract_window_features(
                    x,
                    fs
                )

                row = {
                    "segment": seg_name,

                    "time_s": win_time_s,

                    "time_rel_onset_s":
                        win_time_s
                        - SEIZURE_START,

                    "channel": ch_name,
                }

                row.update(feats)

                rows.append(row)

            window_counter += 1

            # Progress display
            if (
                window_counter % 500 == 0
                or window_counter == n_windows
            ):

                print(
                    f"  Progress: "
                    f"{window_counter}/"
                    f"{n_windows} windows",
                    end="\r"
                )

        print()

    # ---------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)

    print(
        f"Feature extraction complete."
    )

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Channels: {df['channel'].nunique()}"
    )

    print(
        f"Features: {len(FEATURE_NAMES)}"
    )

    print("=" * 70)

    return df, segments


# =============================================================
# PLOTTING
# =============================================================

def plot_feature_evolution(
    df,
    feature,
    outpath
):
    """
    Generate one figure per feature.

    All channels are plotted on the same sequential
    x-axis, with segments concatenated.

    This avoids the large blank gap between the
    interictal and preictal segments.
    """

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    df = df.copy()

    # ---------------------------------------------------------
    # Order segments
    # ---------------------------------------------------------

    df["segment"] = pd.Categorical(
        df["segment"],
        categories=SEGMENT_ORDER,
        ordered=True
    )

    df = df.sort_values(
        ["segment", "time_s"]
    )

    # ---------------------------------------------------------
    # Sequential index
    # ---------------------------------------------------------

    unique_times = (
        df[
            ["segment", "time_s"]
        ]
        .drop_duplicates()
        .sort_values(
            ["segment", "time_s"]
        )
    )

    unique_times = (
        unique_times
        .reset_index(drop=True)
    )

    unique_times["seq_idx"] = np.arange(
        len(unique_times)
    )

    df = df.merge(
        unique_times,
        on=["segment", "time_s"],
        how="left"
    )

    # ---------------------------------------------------------
    # Segment backgrounds
    # ---------------------------------------------------------

    boundaries = (
        unique_times
        .groupby(
            "segment",
            observed=True
        )["seq_idx"]
        .agg(["min", "max"])
    )

    for seg_name in SEGMENT_ORDER:

        if seg_name not in boundaries.index:
            continue

        lo = boundaries.loc[
            seg_name,
            "min"
        ]

        hi = boundaries.loc[
            seg_name,
            "max"
        ]

        ax.axvspan(
            lo - 0.5,
            hi + 0.5,
            color=SEGMENT_COLORS[seg_name],
            alpha=0.12,
            lw=0
        )

        ax.text(
            (lo + hi) / 2,
            1.02,
            seg_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color=SEGMENT_COLORS[seg_name],
            fontweight="bold"
        )

    # ---------------------------------------------------------
    # Plot all channels
    # ---------------------------------------------------------

    for ch in ch_names_for_plot(df):

        sub = (
            df[df["channel"] == ch]
            .sort_values("seq_idx")
        )

        ax.plot(
            sub["seq_idx"],
            sub[feature],
            label=ch,
            linewidth=0.8,
            alpha=0.75
        )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    ax.set_title(
        f"{feature} - evolution across "
        f"interictal / preictal / ictal / postictal\n"
        f"CHB08 chb08_05.edf"
    )

    ax.set_xlabel(
        "Window index "
        "(sequential, segments concatenated)"
    )

    ax.set_ylabel(feature)

    ax.legend(
        loc="upper right",
        fontsize=6,
        ncol=3
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=130
    )

    plt.close(fig)


# =============================================================
# HELPER FOR PLOT CHANNEL ORDER
# =============================================================

def ch_names_for_plot(df):
    """
    Return unique channel names in their original order.
    """

    return list(
        dict.fromkeys(
            df["channel"].tolist()
        )
    )


# =============================================================
# EXCEL OUTPUT
# =============================================================

def write_excel(
    df,
    plot_paths,
    outpath
):
    """
    Write feature data, summary statistics,
    individual feature sheets and plots
    into an Excel workbook.
    """

    print("\nCreating Excel workbook...")

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    summary = (
        df
        .groupby(
            ["segment", "channel"],
            observed=True
        )[FEATURE_NAMES]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{feat}_{stat}"
        for feat, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary["segment"] = pd.Categorical(
        summary["segment"],
        categories=SEGMENT_ORDER,
        ordered=True
    )

    summary = summary.sort_values(
        ["segment", "channel"]
    )

    # ---------------------------------------------------------
    # Write Excel
    # ---------------------------------------------------------

    with pd.ExcelWriter(
        outpath,
        engine="openpyxl"
    ) as writer:

        # Full long-format table
        df.to_excel(
            writer,
            sheet_name="Features_Long",
            index=False
        )

        # Summary statistics
        summary.to_excel(
            writer,
            sheet_name="Summary_Stats",
            index=False
        )

        # -----------------------------------------------------
        # Individual feature sheets
        # -----------------------------------------------------

        for feature in FEATURE_NAMES:

            pivot = (
                df
                .pivot_table(
                    index=[
                        "segment",
                        "time_s"
                    ],
                    columns="channel",
                    values=feature,
                    observed=True
                )
            )

            pivot = pivot.reset_index()

            pivot["segment"] = pd.Categorical(
                pivot["segment"],
                categories=SEGMENT_ORDER,
                ordered=True
            )

            pivot = pivot.sort_values(
                ["segment", "time_s"]
            )

            # Excel sheet names max = 31 characters
            sheet_name = feature[:31]

            pivot.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    # ---------------------------------------------------------
    # Add plots to Excel sheets
    # ---------------------------------------------------------

    from openpyxl import load_workbook
    from openpyxl.drawing.image import Image as XLImage

    wb = load_workbook(outpath)

    for feature in FEATURE_NAMES:

        sheet_name = feature[:31]

        if (
            sheet_name not in wb.sheetnames
            or feature not in plot_paths
        ):
            continue

        ws = wb[sheet_name]

        img = XLImage(
            plot_paths[feature]
        )

        img.anchor = "L2"

        ws.add_image(img)

    wb.save(outpath)

    print(
        f"Excel workbook saved: {outpath}"
    )


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CHB08 chb08_05.edf "
            "23-channel EEG feature analysis"
        )
    )

    parser.add_argument(
        "--edf",
        required=True,
        help="Path to chb08_05.edf"
    )

    parser.add_argument(
        "--outdir",
        default="./output",
        help="Output directory"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    plot_dir = os.path.join(
        args.outdir,
        "plots"
    )

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Process EDF
    # ---------------------------------------------------------

    df, segments = process_file(
        args.edf
    )

    # ---------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING FEATURE PLOTS")
    print("=" * 70)

    plot_paths = {}

    for feature in FEATURE_NAMES:

        outpath = os.path.join(
            plot_dir,
            f"{feature}.png"
        )

        print(
            f"Generating {feature}..."
        )

        plot_feature_evolution(
            df,
            feature,
            outpath
        )

        plot_paths[feature] = outpath

        print(
            f"  Saved: {outpath}"
        )

    # ---------------------------------------------------------
    # Excel
    # ---------------------------------------------------------

    xlsx_path = os.path.join(
        args.outdir,
        "chb08_05_features.xlsx"
    )

    print("\n")
    print("=" * 70)
    print("CREATING EXCEL OUTPUT")
    print("=" * 70)

    write_excel(
        df,
        plot_paths,
        xlsx_path
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nExcel workbook:"
        f"\n  {xlsx_path}"
    )

    print(
        f"\nPlots:"
        f"\n  {plot_dir}\\*.png"
    )

    print(
        "\nSegment windows used "
        "(seconds from file start):"
    )

    for seg in SEGMENT_ORDER:

        t0, t1 = segments[seg]

        print(
            f"  {seg:11s}: "
            f"{t0:7.0f}s - "
            f"{t1:7.0f}s "
            f"({(t1 - t0) / 60:.1f} min)"
        )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()
