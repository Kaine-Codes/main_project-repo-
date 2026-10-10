"""
Compute trimmed Z-score calibration constants from testbench EEG data.
Uses the 256 raw values from tb_mean_power_pipe.sv as a stand-in dataset.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import math

# --- 256 raw EEG values from testbench ---
raw_values = [
    -10.74481074, -1.75824176, -6.05616606, -0.19536020, 0.97680098, -1.36752137, -4.49328449, -2.93040293,
    -7.61904762, -12.30769231, -6.05616606, -3.71184371, -2.53968254, 13.47985348, -1.36752137, -20.90354090,
    -27.54578755, -33.40659341, -31.45299145, -25.20146520, -21.29426129, -11.91697192, -14.26129426, -8.40048840,
    -16.21489621, -26.37362637, -25.59218559, -15.04273504, -10.74481074, 11.91697192, 0.19536020, -21.29426129,
    -23.63858364, -27.93650794, -29.89010989, -31.06227106, -28.32722833, -36.14163614, -38.09523810, -22.85714286,
    -36.92307692, -39.65811966, -30.67155067, -30.28083028, -33.79731380, -12.30769231, -31.84371184, -41.22100122,
    -31.84371184, -36.53235653, -34.18803419, -29.49938950, -27.54578755, -33.79731380, -27.15506716, -20.90354090,
    -16.99633700, -9.96336996, -4.88400488, 2.14896215, 2.14896215, 10.74481074, 7.22832723, -5.27472527,
    2.14896215, 10.35409035, 5.66544567, 9.96336996, 3.71184371, -2.53968254, -0.19536020, 6.05616606,
    6.83760684, 6.83760684, 12.69841270, 17.77777778, 24.02930403, 42.39316239, 27.93650794, 25.20146520,
    26.76434676, 20.51282051, 22.46642247, 16.21489621, 15.04273504, 13.08913309, 20.90354090, 20.12210012,
    8.79120879, 4.88400488, 4.10256410, 1.36752137, 8.79120879, 25.98290598, 3.71184371, -7.61904762,
    -2.53968254, -6.83760684, -10.74481074, 2.93040293, -2.93040293, -13.47985348, -2.93040293, 1.75824176,
    3.32112332, 4.88400488, 11.91697192, 16.21489621, 22.46642247, 42.78388278, 43.56532357, 21.68498168,
    12.69841270, 13.08913309, 7.61904762, 24.81074481, 27.15506716, 23.63858364, 11.52625153, 14.65201465,
    9.18192918, 8.00976801, 20.51282051, 22.46642247, 24.42002442, 36.92307692, 22.46642247, 10.35409035,
    16.99633700, 18.55921856, 24.42002442, 23.63858364, 23.63858364, 20.51282051, 14.65201465, 23.63858364,
    27.93650794, 23.24786325, 27.54578755, 28.32722833, 25.20146520, 47.08180708, 41.61172161, 27.93650794,
    31.84371184, 22.46642247, 22.85714286, 33.01587302, 35.75091575, 30.67155067, 29.89010989, 25.59218559,
    16.60561661, 8.79120879, 9.96336996, 14.26129426, 15.82417582, 38.87667888, 27.93650794, 8.79120879,
    11.52625153, -0.97680098, -1.36752137, 0.97680098, -0.97680098, -1.36752137, -4.10256410, 6.05616606,
    -1.75824176, -0.97680098, 9.57264957, 18.55921856, 24.02930403, 38.48595849, 23.24786325, 10.35409035,
    10.74481074, 9.57264957, 10.74481074, 15.04273504, 13.47985348, 8.40048840, 10.35409035, 14.26129426,
    -0.19536020, -7.61904762, -13.47985348, -18.16849817, -8.40048840, 1.36752137, -6.05616606, -16.60561661,
    -19.34065934, -21.29426129, -28.71794872, -25.20146520, -29.49938950, -40.43956044, -29.89010989, -33.01587302,
    -25.59218559, -22.85714286, -25.20146520, -15.04273504, -13.08913309, 2.93040293, -2.93040293, -14.26129426,
    -9.57264957, -11.52625153, -7.61904762, -1.36752137, -1.36752137, -1.36752137, 0.97680098, 8.40048840,
    -4.10256410, -9.18192918, 0.19536020, 7.61904762, 12.69841270, 45.12820513, 27.54578755, 5.27472527,
    -1.36752137, -2.93040293, -2.53968254, 0.97680098, 0.58608059, -3.71184371, -0.58608059, 8.40048840,
    6.44688645, -2.14896215, -2.93040293, 0.58608059, 7.61904762, 30.67155067, 29.89010989, 9.18192918,
    2.14896215, -0.58608059, -5.66544567, 4.49328449, 12.69841270, 10.74481074, 15.43345543, 27.54578755,
    27.93650794, 16.99633700, 24.02930403, 17.38705739, 16.60561661, 44.73748474, 32.62515263, 13.87057387,
]

N = len(raw_values)
print(f"Total samples: {N}")
print("=" * 70)

# =============================================
# 1. Raw (untrimmed) statistics
# =============================================
raw_mean = sum(raw_values) / N
raw_var = sum((x - raw_mean)**2 for x in raw_values) / N
raw_std = math.sqrt(raw_var)

print(f"\n--- RAW (Untrimmed) Statistics ---")
print(f"  Mean (μ):       {raw_mean:.6f}")
print(f"  Std Dev (σ):    {raw_std:.6f}")
print(f"  1/σ:            {1/raw_std:.6f}")
print(f"  Min:            {min(raw_values):.6f}")
print(f"  Max:            {max(raw_values):.6f}")

# =============================================
# 2. Trimmed statistics (5% trim on each side)
# =============================================
TRIM_PERCENT = 5  # trim 5% from each tail
trim_count = int(N * TRIM_PERCENT / 100)

sorted_vals = sorted(raw_values)
trimmed_vals = sorted_vals[trim_count : N - trim_count]
N_trimmed = len(trimmed_vals)

trimmed_mean = sum(trimmed_vals) / N_trimmed
trimmed_var = sum((x - trimmed_mean)**2 for x in trimmed_vals) / N_trimmed
trimmed_std = math.sqrt(trimmed_var)

print(f"\n--- TRIMMED Statistics (5% each tail, {trim_count} samples removed per side) ---")
print(f"  Samples used:   {N_trimmed} out of {N}")
print(f"  Trimmed Mean (μ_τ):     {trimmed_mean:.6f}")
print(f"  Trimmed Std Dev (σ_τ):  {trimmed_std:.6f}")
print(f"  1/σ_τ:                  {1/trimmed_std:.6f}")
print(f"  Trimmed range:          [{sorted_vals[trim_count]:.4f}, {sorted_vals[N - trim_count - 1]:.4f}]")
print(f"  Excluded low:           {[f'{v:.2f}' for v in sorted_vals[:trim_count]]}")
print(f"  Excluded high:          {[f'{v:.2f}' for v in sorted_vals[N - trim_count:]]}")

# =============================================
# 3. Clipping boundaries (±6σ_τ from μ_τ)
# =============================================
clip_low = trimmed_mean - 6 * trimmed_std
clip_high = trimmed_mean + 6 * trimmed_std

print(f"\n--- Clipping Thresholds (±6σ_τ) ---")
print(f"  Clip Low:   {clip_low:.6f}")
print(f"  Clip High:  {clip_high:.6f}")

# Count how many raw samples would be clipped
clipped_count = sum(1 for x in raw_values if x < clip_low or x > clip_high)
print(f"  Samples that would be clipped: {clipped_count} / {N} ({100*clipped_count/N:.2f}%)")

# =============================================
# 4. Apply Z-score normalization to all 256 values
# =============================================
print(f"\n--- Normalized Data (z = (x - μ_τ) × (1/σ_τ), clamped to ±6) ---")
normalized = []
for x in raw_values:
    z = (x - trimmed_mean) * (1.0 / trimmed_std)
    z_clamped = max(-6.0, min(6.0, z))
    normalized.append(z_clamped)

norm_min = min(normalized)
norm_max = max(normalized)
norm_mean = sum(normalized) / len(normalized)
norm_var = sum((z - norm_mean)**2 for z in normalized) / len(normalized)
norm_std = math.sqrt(norm_var)

print(f"  Normalized Min:   {norm_min:.6f}")
print(f"  Normalized Max:   {norm_max:.6f}")
print(f"  Normalized Mean:  {norm_mean:.6f}")
print(f"  Normalized Std:   {norm_std:.6f}")

# =============================================
# 5. Q4.12 Fixed-Point Encoding Check
# =============================================
print(f"\n--- Q4.12 Fixed-Point Encoding Verification ---")
Q_FRAC = 12
SCALE = 2**Q_FRAC  # 4096

def to_q4_12(val):
    """Convert float to Q4.12 signed 16-bit integer."""
    scaled = round(val * SCALE)
    # Clamp to 16-bit signed range [-32768, 32767]
    scaled = max(-32768, min(32767, scaled))
    return scaled

def from_q4_12(ival):
    """Convert Q4.12 signed 16-bit integer back to float."""
    return ival / SCALE

# Encode and decode to check round-trip error
max_error = 0.0
for z in normalized:
    encoded = to_q4_12(z)
    decoded = from_q4_12(encoded)
    error = abs(z - decoded)
    if error > max_error:
        max_error = error

print(f"  Max quantization error: {max_error:.8f}")
print(f"  Q4.12 LSB precision:   {1/SCALE:.8f}")
print(f"  Relative error:        {max_error / (norm_max - norm_min) * 100:.4f}%")

# =============================================
# 6. Hardware Constants Summary (ready to flash)
# =============================================
print(f"\n{'=' * 70}")
print(f"  HARDWARE CONSTANTS (for testbench single-channel data)")
print(f"{'=' * 70}")
print(f"  μ_τ          = {trimmed_mean:.6f}")
print(f"  σ_τ          = {trimmed_std:.6f}")
print(f"  1/σ_τ        = {1/trimmed_std:.6f}")
print(f"  Clip Low     = {clip_low:.6f}  (raw domain)")
print(f"  Clip High    = {clip_high:.6f}  (raw domain)")
print(f"")
print(f"  Q4.12 encoded μ_τ:    {to_q4_12(trimmed_mean)} (0x{to_q4_12(trimmed_mean) & 0xFFFF:04X})")
print(f"  Q4.12 encoded 1/σ_τ:  {to_q4_12(1/trimmed_std)} (0x{to_q4_12(1/trimmed_std) & 0xFFFF:04X})")
print(f"{'=' * 70}")

# =============================================
# 7. Print first 20 normalized values for spot-check
# =============================================
print(f"\n--- First 20 Normalized Values (for spot-checking) ---")
print(f"  {'Index':>5}  {'Raw':>12}  {'Normalized':>12}  {'Clamped':>12}  {'Q4.12 Hex':>10}")
for i in range(20):
    x = raw_values[i]
    z = (x - trimmed_mean) / trimmed_std
    z_c = max(-6.0, min(6.0, z))
    q = to_q4_12(z_c)
    print(f"  {i:5d}  {x:12.4f}  {z:12.6f}  {z_c:12.6f}  0x{q & 0xFFFF:04X}")

# =============================================
# 8. Mean power comparison: FP32 raw vs fixed-point normalized
# =============================================
print(f"\n--- Mean Power Comparison ---")
raw_mean_power = sum(x**2 for x in raw_values) / N
norm_mean_power = sum(z**2 for z in normalized) / len(normalized)

print(f"  Raw mean power (FP32):         {raw_mean_power:.6f}")
print(f"  Normalized mean power (Q4.12): {norm_mean_power:.6f}")
print(f"  (These are in different scales — the normalized value is what the")
print(f"   fixed-point pipeline will compute and pass to the TCN)")
