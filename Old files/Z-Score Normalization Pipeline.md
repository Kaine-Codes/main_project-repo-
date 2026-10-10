# Fused Z-Score Normalization & Mean Power Pipeline — Blocks B+C Specification

## 1. Overview

This document specifies the fused Robust Z-Score Normalization and Mean Power computation pipeline that sits at **Blocks B and C (Preprocessing & Feature Extraction)** in the system architecture. The pipeline normalizes raw EEG data into a bounded, fixed-point-friendly range using patient-specific calibration constants, and computes the 256-sample mean power per channel. This design enables all downstream blocks (C through G) to operate entirely in **16-bit fixed-point arithmetic** instead of IEEE 754 floating-point.

The design splits the workload into two phases:
- **Phase 1 (Offline):** Heavy statistical computation on a PC — done once per patient before IC deployment.
- **Phase 2 (On-Chip):** Real-time processing through a 6-stage time-division multiplexed (TDM) pipeline.

---

## 2. Phase 1: Software Calibration (Offline, on PC)

This phase is performed once per patient using the 48-hour stimulated observation dataset, before flashing the IC.

### Step 1: Isolate the Interictal Baseline
From the full CSV dataset, extract a **clean, seizure-free continuous window** representing the Interictal state (State 1 from Block G). Discard all Preictal, Ictal, and Postictal segments for this calibration step. This ensures the baseline statistics represent the patient's normal resting brain activity, not seizure-contaminated data.

### Step 2: Calculate Trimmed Parameters (Per Channel)
For each of the **23 EEG electrodes independently**, compute:

- **Trimmed Mean (μ_τ):** Discard the top and bottom 5% of sample values, then compute the arithmetic mean of the remaining 90%. This rejects electrode pops, cable artifacts, and baseline drift extremes.
  
- **Trimmed Standard Deviation (σ_τ):** Using the same trimmed subset, compute the standard deviation. This provides a robust measure of normal physiological signal spread without inflation from extreme outliers.

> [!IMPORTANT]
> These must be calculated **per-channel (per-electrode)**, not globally. Each electrode has unique impedance, placement depth, and baseline amplitude. A frontal electrode (Fp1/Fp2) will have fundamentally different μ and σ than an occipital one (O1/O2).

### Step 3: Extract the Hardware Constants
For each of the 23 channels, compute:

| Constant | Formula | Purpose |
|---|---|---|
| **μ_τ[ch]** | Trimmed mean of channel `ch` | Baseline offset removal |
| **1/σ_τ[ch]** | Reciprocal of trimmed std dev | Scaling (avoids hardware division) |

**Total constants to flash: 23 × 2 = 46 fixed-point values**, stored in on-chip registers.

### Step 4: Flash to IC
Encode all 46 constants in the target fixed-point format and program them into the IC's calibration register file. These values remain static during deployment and define the patient-specific normalization behavior.

### Reference Calibration: Testbench Data (Single-Channel, 256 Samples)

The following constants were computed from the 256 raw EEG values in `tb_mean_power_pipe.sv` using 5% trimming (12 samples discarded per tail, 232 retained). These serve as verification constants for testbench development.

| Constant | Float Value | Q4.12 Integer | Q4.12 Hex |
|---|---|---|---|
| **μ_τ** (trimmed mean) | 2.920298 | 11962 | `0x2EBA` |
| **σ_τ** (trimmed std dev) | 16.545390 | — | — |
| **1/σ_τ** (reciprocal) | 0.060440 | 248 | `0x00F8` |

**Post-normalization statistics:**

| Metric | Value |
|---|---|
| Normalized range | [-2.668, +2.669] |
| Normalized mean | -0.004 (≈ 0) |
| Normalized std | 1.193 (≈ 1) |
| Normalized mean power | 1.422 |
| Samples clipped at ±6σ | 0 / 256 (0%) |
| Max Q4.12 quantization error | 0.000122 (0.0023% relative) |

> [!NOTE]
> These are **single-channel reference values** computed from testbench data. In production, each of the 23 electrodes will have its own unique μ_τ and 1/σ_τ, computed from the patient's interictal baseline during the offline calibration phase. The calibration script is at `scratch/zscore_calibration.py`.

---

## 3. Phase 2: Hardware Execution (Real-Time on IC)

### 3.1 Block B+C Processing Pipeline

Each raw EEG sample flows through a fused 6-stage pipeline:

```
Raw 24-bit ADC sample (from Block A)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Mean Subtraction  │  diff = x_raw - μ_τ[ch]
│  (integer subtract)         │  Constants indexed by channel counter
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 2: Scaling           │  prod = diff × (1/σ_τ[ch])
│  (fixed-point multiply)     │  Single-cycle on DSP48 slice
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 3: Saturation Clamp  │  z = clamp(prod >> shift, -6.0, +6.0)
│  (comparator + mux)         │  Hardcoded ±6, universal for all channels
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 4: Square            │  z_sq = z × z
│  (fixed-point multiply)     │  Single-cycle on DSP48 slice
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 5: Accumulate        │  acc[ch] += z_sq
│  (integer add + memory)     │  samp_cnt[ch] += 1
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Stage 6: Store (if WIN)    │  mean_power[ch] = acc[ch] >> shift
│  (shift + round)            │  acc[ch] = 0, samp_cnt[ch] = 0
└─────────────────────────────┘
        │
        ▼
  Mean Power Q4.12 output per channel + window_valid signal
```

**Latency: 6 clock cycles.** Fully pipelined with 1 channel/cycle throughput. No pipeline drain overhead. No floating-point overhead.

### 3.2 Saturation Clamp Rationale

The clamp at **±6 (post-normalization)** is hardcoded — the same constant for all 23 channels. This works because after Z-score normalization, the value represents "number of standard deviations from mean." Clamping at ±6σ:

| Threshold | Coverage | What gets clipped |
|---|---|---|
| ±3σ | 99.73% | ⚠️ Too aggressive — clips genuine preictal spikes |
| ±5σ | 99.9999% | Safe but tight |
| **±6σ** | **99.9999998%** | **Recommended — only electrode pops & cable disconnects** |
| ±10σ | ~100% | Too loose — extreme artifacts pass through |

> [!WARNING]
> **Clip, never delete.** The TCN's dilated convolutions (T1–T6, dilation 1→32) require evenly-spaced time steps. Deleting outlier samples would create temporal gaps that corrupt every dilated receptive field. Clamping preserves the timestamp while bounding the amplitude.

### 3.3 Why ±6 Is Hardcoded (Not Per-Channel)

After normalization, every channel's data is in the same statistical space — standard deviations from that channel's own mean. The clipping threshold of ±6 is a universal physiological boundary, not a channel-specific one. This eliminates 46 additional constants (23 × upper/lower) from the register file.

---

## 4. Time-Division Multiplexed (TDM) Architecture

### The Problem
23 electrodes each need independent normalization (different μ_τ, 1/σ_τ). Building 23 parallel processing pipelines would consume 23× the silicon area, power, and routing resources — this is counterintuitive and wasteful.

### The Solution: One Shared Pipeline
A single, highly optimized hardware pipeline processes all 23 channels **sequentially** using Time-Division Multiplexing:

```
                    ┌─────────────────────────────────┐
                    │    TDM Feed FSM                 │
                    │    Cycles: 0 → 1 → 2 → ... → 22│
                    └────────────┬────────────────────-┘
                                 │ feed_idx
                    ┌────────────▼────────────────────-┐
                    │   Calibration Register File      │
                    │   23 entries × 2 constants each  │
                    │   [μ_τ[ch], (1/σ_τ)[ch]]        │
                    │   Addressed by ch_idx            │
                    └────────────┬────────────────────-┘
                          μ_τ[ch] │ (1/σ_τ)[ch]
                    ┌────────────▼────────────────────-┐
   x_batch ──────► │   6-Stage Processing Pipeline    │
                   │   Sub → Mul → Clamp → Sq → Acc   │ ────► mean_power[ch]
                   └──────────────────────────────────-┘
```

### Timing Feasibility

| Parameter | Value |
|---|---|
| Sampling rate | 256 Hz per electrode |
| Sample period | 1/256 s = **3,906.25 µs** |
| Clock cycles per sample period (@ 100 MHz) | **390,625 cycles** |
| Cycles to feed all 23 channels | **23 cycles** |
| **Pipeline utilization** | **23 / 390,625 = 0.0058%** |

The pipeline is idle for **>99.99% of the time**. One shared unit can handle all 23 channels with massive headroom.

### Register File Design

The 46 constants are stored in a small **dual-port register file** (or shallow BRAM):

```
Address [4:0]    │  μ_τ (Q-format)    │  1/σ_τ (Q-format)
─────────────────┼─────────────────────┼──────────────────
  0 (Ch 0)       │  μ_τ[0]            │  (1/σ_τ)[0]
  1 (Ch 1)       │  μ_τ[1]            │  (1/σ_τ)[1]
  ...            │  ...               │  ...
  22 (Ch 22)     │  μ_τ[22]           │  (1/σ_τ)[22]
```

This consumes **< 1 BRAM tile** on any Xilinx FPGA, or can be implemented entirely in distributed LUT-RAM (46 × 16-bit = 736 bits total).

---

## 5. Fixed-Point Format Specification

### Post-Normalization Format: Q4.12 (16-bit signed)

| Property | Value |
|---|---|
| Total width | 16 bits |
| Sign | 1 bit |
| Integer bits | 3 bits |
| Fractional bits | 12 bits |
| Range | [-8.0, +7.999755859375) |
| Precision | 1/4096 ≈ 0.000244 |
| Normalized data range | [-6.0, +6.0] (after clamp) |

This format comfortably contains the ±6 clamp range with 2 bits of headroom for intermediate computations.

### Intermediate Accumulator Width

For downstream feature extraction (e.g., mean_power = (1/256) × Σ(z_i²)):
- Max z² = 6² = 36
- Max Σ(z_i²) for 256 samples = 36 × 256 = 9,216
- **Accumulator format:** Q14.18 (32-bit) — range up to ±8,192 with 18 bits of fractional precision
- **After ÷256 (right-shift by 8):** result fits back in Q4.12

---

## 6. Accuracy Analysis: FP32 vs. Fixed-Point

### Empirical Results (256-sample testbench data)

| Method | Mean Power Result | Error vs Gold |
|---|---|---|
| Python float64 (gold reference) | 397.447 | — |
| FP32 hardware (Siemens Questa sim) | 369.995 | **~6.9%** |
| FP32 simulated in Python | 397.447 | ~0.00005% |
| Q4.12 fixed-point (theoretical) | ~396.85 | **~0.15%** |

> [!IMPORTANT]
> The current FP32 hardware pipeline exhibits **~7% error** vs the gold reference. This is caused by rounding errors compounding through the 4-stage multiplier and combinational adder — every FP32 addition requires mantissa alignment, which discards LSBs. Over 256 sequential additions, these small losses accumulate significantly.

### Why Fixed-Point Accumulation Is Actually More Accurate

Fixed-point (integer) addition is **exact** — there is no mantissa alignment, no rounding, no precision loss at each step. When you add two 32-bit integers, the result is mathematically exact (as long as it doesn't overflow). This means:

- **FP32 accumulation:** Each of the 256 additions introduces a small rounding error. These errors compound because the accumulator grows large relative to the individual squared values, causing mantissa alignment to discard increasingly many bits from the smaller operand.
- **Fixed-point accumulation:** All 256 additions are exact. The only precision loss is the initial quantization when converting from analog to Q4.12 (~0.000122 per sample), which does not compound during accumulation.

For this reason, a well-designed fixed-point pipeline can achieve **better numerical accuracy** than the current FP32 pipeline, while using half the data width and significantly less power.

### Practical Relevance

For neural network inference (the TCN in Blocks F), even 1% error in feature values is negligible. The network is trained with dropout (0.3), noisy biological signals, and quantized weights. The TCN will not distinguish between a mean power of 1.422 vs 1.420. Both FP32 and fixed-point are more than adequate — but fixed-point is cheaper, cooler, and in practice more numerically stable for accumulation-heavy operations.

---

## 7. Impact Summary: FP32 vs. Fused Fixed-Point Pipeline

| Metric | Current (FP32) | Fused Z-Score + Mean Power (Fixed-Point) |
|---|---|---|
| Multiplier | 4-stage pipeline, ~1,500 LUTs | Single-cycle, DSP48 slice |
| Adder | Combinational FP32, ~400 LUTs | Integer add, minimal LUTs |
| Division by 256 | Exponent subtract (FP32 logic) | Right-shift by 20 (zero cost) |
| Pipeline states | IDLE→LOAD→DRAIN→FINAL→DONE | IDLE→FEED |
| Dynamic power estimate | Baseline | **~4–8× lower** |
| Data width through pipeline | 32-bit | **16-bit / 40-bit internal** |
| Accumulation accuracy | ~7% error (rounding compounds) | **~0.15% error (exact adds)** |
| Parallel channel support | 1 pipeline per channel (expensive) | **1 TDM pipeline for all 23 channels** |

---

## 8. Control Unit Architecture

The TDM pipeline requires a control unit to sequence operations across 23 channels. Critically, **this controller is simpler than the current FP32 mean_power_pipe FSM**, because fixed-point operations are single-cycle (no pipeline drain tracking).

### 8.1 Fused Pipeline Controller (Blocks B+C)

A lightweight 2-state TDM Feed FSM:

```
┌──────────┐  batch_valid          ┌──────────────┐
│  S_IDLE  │ ────────────────────► │  S_FEED      │
│          │                       │              │
│ Wait for │                       │ Feed ch 0    │
│ new batch│                       │ through 22   │
│          │ ◄──────────────────── │ (1 ch/cycle) │
└──────────┘  (feed_idx == 22)     └──────────────┘
```

**Signals:**

| Signal | Width | Direction | Description |
|---|---|---|---|
| `x_batch` | 24×N | Input | Array of raw ADC samples for all channels |
| `batch_valid` | 1 | Input | Asserted when new ADC samples are ready |
| `feed_idx` | 5 | Internal | Channel counter, cycles 0→22 during feeding |
| `ch_pipe` | 5×6 | Internal | Delay line tracking channel ID through the 6 stages |
| `mean_power` | 16×N | Output | Array of normalized Q4.12 mean power values |
| `window_valid` | 1 | Output | High for 1 cycle when all mean_power values are updated for the window |

The entire controller is simple — a counter, a register file read, and a pipelined datapath. This eliminates the need for complex pipeline drain logic found in the previous FP32 design.

### 8.2 Hierarchical Control for Full Pipeline

The broader system uses a **hierarchical producer-consumer architecture** with simple handshaking between levels. No single monolithic controller is needed:

```
 Level 1: Per-Sample (runs every 1/256 s = 3.9 ms)
 ┌──────────────────────────────────────────────────────┐
 │  Blocks B+C Controller (Z-Score + Mean Power TDM)    │
 │  - 2-state Feed FSM                                 │
 │  - Feeds 23 channels in 23 cycles                   │
 │  - Outputs window_valid every 256 batches           │
 └──────────────────┬───────────────────────────────────┘
                    │ window_valid trigger
                    ▼
 Level 2: Per-Window (runs every 1 second)
 ┌──────────────────────────────────────────────────────┐
 │  Block C/D Controller (Feature Extraction TDM)       │
 │  - Triggers on window_valid                          │
 │  - Computes remaining features per channel           │
 │  - Outputs feature vector                            │
 └──────────────────┬───────────────────────────────────┘
                    │ FIFO
                    ▼
 Level 3: Per-Sequence (runs every SEQ_LEN windows)
 ┌──────────────────────────────────────────────────────┐
 │  Block E/F Controller (TCN Inference)                │
 │  - Triggers when SEQ_LEN vectors accumulated         │
 │  - Feeds SEQ_LEN×19 matrix through TCN               │
 │  - Outputs 4-class probability vector                │
 └─────────────────────────────────────────────────────-┘
```

Each level operates independently with its own simple FSM. They communicate through **FIFO buffers** (standard producer-consumer pattern). No complex centralized controller is required — the complexity is distributed across simple, testable modules.

> [!TIP]
> The key insight is that each level has **enormous timing slack**. Level 1 uses 69 out of 390,625 available cycles. Level 2 uses ~113K out of 400M available cycles. This means even with TDM and a single shared pipeline, timing is never a concern.

---

## 9. Complete Data Flow Summary

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PHASE 1: OFFLINE (PC) — Done once per patient                     │
  │                                                                    │
  │  48-hour CSV ──► Extract Interictal ──► Trimmed Stats ──► Flash IC │
  │                  (seizure-free)          μ_τ[0..22]                 │
  │                                         (1/σ_τ)[0..22]            │
  │                                         (46 constants)             │
  └──────────────────────────────────────────────────────────────────────┘
                              │
                    Flash to IC registers
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PHASE 2: ON-CHIP (Real-Time) — Continuous monitoring              │
  │                                                                    │
  │  Block A           Blocks B+C (This Pipeline)                      │
  │  ┌─────────┐    ┌───────┬────────┬───────┬───────┬───────┐         │
  │  │ 23 EEG  │───►│ -μ_τ  │ ×1/σ_τ │ Clamp │ Square│ Accum │───► ... │
  │  │ @256 Hz │    │       │        │  ±6   │       │       │         │
  │  │ 24-bit  │    └───────┴────────┴───────┴───────┴───────┘         │
  │  └─────────┘     6 cycles, TDM across 23 channels                  │
  │                  Outputs Mean Power per channel                    │
  └──────────────────────────────────────────────────────────────────────┘
```
