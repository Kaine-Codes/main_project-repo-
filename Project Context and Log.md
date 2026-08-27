# Master Context: Early Epilepsy Detection Custom IC

## 1. Project Objective
The objective is to synthesize an Application-Specific Integrated Circuit (ASIC) for real-time, low-power early epilepsy detection. The IC will monitor neurological EEG data to identify the "preictal" state (which typically begins 20 minutes before a seizure) and issue a hardware interrupt at least 2-4 minutes prior to the physical seizure onset. This physiological margin imposes a strict maximum hardware latency of 18-16 minutes for the entire data pipeline.

## 2. Clinical Data Acquisition & Training Workflow

The patient undergoes a **stimulated observation protocol**, where controlled epilepsy triggers are administered under clinical supervision to deliberately induce seizure activity. This provocation allows the 23 EEG electrodes to capture comprehensive diagnostic data across all four brain states (Interictal, Preictal, Ictal, Postictal) within a condensed observation window (typically 48 hours).

### Raw Dataset Characteristics
- **Format:** CSV file outputs from the clinical EEG recording system.
- **Scale:** Each CSV file represents approximately 1 hour of continuous recording and contains hundreds of thousands of columns (one per time-sample per channel). File sizes range from **100 MB to 1.5 GB** per hour.
- **Preprocessing:** The raw continuous recordings will be sliced into clean **4-second windows** by the data preparation team before hardware feature extraction, aligning with Block C's sliding window specification.

### Training Data Considerations
- The raw clinical data contains **significant outliers and large amplitude values** due to electrode artifacts, movement noise, and stimulated seizure events. The preprocessing and normalization pipeline must robustly handle these extreme values to prevent corruption of downstream feature extraction and model inference.
- After training on the full stimulated dataset, the IC is deployed as a **patient-specific customized version**, with model weights and calibration constants tailored to that individual's neurological profile.

## 3. System Pipeline Architecture (Blocks A to G)
The architecture follows a strict feed-forward pipeline from raw analog signal acquisition to a 4-state probability output.

### Signal Conditioning & Feature Extraction (Blocks A–E)

- **Block A: Raw EEG Signals.** 23 EEG electrodes sampled at 256 Hz with 24-bit resolution via **ADS1299 ADCs** (8 channels per chip, 3 chips for 23 electrodes). All channels fire simultaneously — every 1/256 s (3.906 ms), the ADC delivers one fresh 24-bit sample per electrode. Raw samples are encoded in **Q8.16 fixed-point** (24-bit signed, range [-128, +128)) for direct fixed-point processing. **No IEEE 754 conversion.** Along with EEG, sweat sensor data may be added later (different sampling rate/resolution — deferred).

- **Block B: Preprocessing (Z-Score Normalization).** Channel-wise Z-score normalization using patient-specific calibration constants computed offline (Phase 1).

  - **Calibration (Phase 1, offline on PC):** From the 48-hour clinical dataset, extract seizure-free interictal segments. Per channel: sort values, trim 5% from each tail (reject artifacts), compute trimmed mean (μ_τ) and trimmed std dev (σ_τ). Encode μ_τ in Q8.16, 1/σ_τ in Q4.12. Flash constants to IC registers. Done once per patient.
  - **Output format:** Q4.12 (16-bit signed, range [-8, +8), precision 1/4096). Normalized data bounded to ±6σ.

  #### Concept 1: TDM Single-Pipeline (Verified on FPGA)
  ✅ **Verified and implemented in simulation.** A **6-stage fused design** processing all 23 electrodes through a single shared pipeline via Time-Division Multiplexing:

  | Stage | Operation | Hardware | Cycles |
  |---|---|---|---|
  | 1 | Subtract μ_τ[ch] | Integer subtractor | 1 |
  | 2 | Multiply × 1/σ_τ[ch] | DSP48 slice | 1 |
  | 3 | Clamp to ±6σ | Comparator + mux | 1 |
  | 4 | Square z² | DSP48 (reused or second) | 1 |
  | 5 | Accumulate acc[ch] += z² | Integer adder | 1 |
  | 6 | Store (after 256 samples: acc >> 8) | Register write | 1 |

  - **TDM Timing:** 23 electrodes processed in ~27 cycles (250 ns) per sample period, out of 390,625 available cycles. Pipeline utilization: 0.007%. The chip clock-gates during idle time for power savings.
  - **Accuracy:** ~0.35% error vs float64 gold reference (verified in simulation). Compared to the old FP32 pipeline which had ~6.9% error due to mantissa rounding accumulation.
  - **Trade-off:** Minimal hardware (1 subtractor, 1 multiplier) but fuses normalization + feature accumulation into one pass. Mean power (z²) is computed inline.

  #### Concept 2: Triple-Buffered Parallel Pipeline (Under Development)
  🔶 **Proposed alternative architecture.** Decouples normalization from feature extraction using three rotating memory blocks and 8-wide parallel processing lanes.

  **Core Idea:** Three identical memory blocks (A, B, C) rotate through three states — Storing, Processing, Data Ready — at 1-second granularity. After a 2-second startup latency, every second produces one fully normalized block ready for feature extraction.

  **Memory Block Structure:** Each block is an **8-column × 32-row = 256-cell** register file, holding exactly 1 second of samples (256 samples at 256 Hz) for a single channel. During the "Storing" phase, incoming samples fill the block sequentially (1 sample every 3.9ms). An 8-bit write counter maps to row (bits [7:3]) and column (bits [2:0]).

  **Rotation Schedule:**

  | Time | Block A | Block B | Block C |
  |---|---|---|---|
  | 0–1s | STORING | — | — |
  | 1–2s | PROCESSING | STORING | — |
  | 2–3s | DATA READY | PROCESSING | STORING |
  | 3–4s | STORING ♻️ | DATA READY | PROCESSING |
  | 4–5s | PROCESSING | STORING ♻️ | DATA READY |
  | 5–6s | DATA READY | PROCESSING | STORING ♻️ |

  **Processing Phase — 8-Wide Parallel Normalization:**
  During processing, 8 parallel lanes (each with its own subtractor + multiplier + clamper) consume one row (8 values) per cycle:
  - Cycle 1: Row 0 (indices 0–7) → subtract μ → multiply 1/σ → clamp ±6σ → write back
  - Cycle 2: Row 1 (indices 8–15) → same
  - ...
  - Cycle 32: Row 31 (indices 248–255) → done
  - **Total: ~64–100 cycles (~640–1000 ns at 10ns clock).** The remaining ~99,999,000 ns of that second is free for feature extraction and TCN.

  **In-Place Overwrite:** Normalized values replace the raw values in the same memory block. Since the pipeline reads row N and writes back normalized row N-1 in the next cycle, there is no read-write collision with a standard 2-cycle pipeline. The 32-row structure is sufficient (no extra scratch row needed).

  **Data Ready Phase:** The block holds 256 normalized values. Feature extraction hardware reads freely from this block during the entire second.

  **Bus Routing — Minimal Muxing:**
  The data does not physically move between blocks. Bus connections are reconfigured each second via muxes controlled by a 2-bit rotating FSM counter:
  - 1× input demux per channel (1:3, 24-bit wide) — routes ADC data to the "storing" block
  - 1× normalization read mux per channel (3:1, 8×16-bit wide) — reads from the "processing" block
  - 1× feature extraction read mux per channel (3:1, 8×16-bit wide) — reads from the "data ready" block
  - FSM: ~15 flip-flops total (2-bit rotating counter + write counter + row counter)

  **Strengths:**
  - Clean phase separation — each block has exactly one job at any time
  - Burst-then-sleep — normalization finishes in ~100 cycles, hardware can be clock-gated for 99.9999% of the second
  - No sample loss — one block is always storing
  - Decoupled stages — feature extraction reads stable, complete data with no race conditions
  - Massive cycle budget freed for feature extraction and TCN processing

  **Concerns & Open Questions:**
  - ⚠️ **Dual-port memory needed:** Simultaneous read (row N) and write-back (row N-1) requires dual-port SRAM or register file. Single-port SRAM would double cycle count.
  - ⚠️ **8× hardware vs Concept 1:** 8 subtractors + 8 multipliers vs 1 of each. More transistors = more static leakage power. For a hearing-aid-sized device, leakage may dominate.
  - ⚠️ **Multi-channel scaling:** Each channel needs its own set of 3 memory blocks. For 5 channels = 15 blocks. Coordination and routing complexity scales. *(Not finalized — pending mentor discussion.)*
  - ⚠️ **Normalization constants switching:** When processing different channels sequentially through the shared 8-wide lanes, μ and 1/σ constants must be swapped via mux. For 5 channels × 32 rows = 160 row-loads ≈ 160–320 cycles total. Still trivial.
  - ⚠️ **Raw data lost after overwrite:** Original samples are destroyed. Re-normalization (recalibration) requires re-acquisition.
  - ⚠️ **Control signal complexity and heat dissipation:** The volume of muxes, parallel hardware, and bus routing raises concerns about control signal generation complexity and thermal output for a hearing-aid form factor.

  **Feature Extraction Integration (Tentative):**
  Mean power and other features are computed in a **separate feature extraction block** downstream (not fused into normalization as in Concept 1). The GPU-like MAC array concept is under consideration — a bank of parallel multiply-accumulate units that processes the normalized data during the massive remaining cycle budget. *(Architecture not finalized.)*

- **Block C: Feature Extraction.** Features are computed **per-second** using on-the-fly accumulation (no 256-sample buffer needed for most features). The IC accumulates z² (and other feature accumulators) alongside Block B normalization during each 1-second window (256 samples). After 256 samples, the accumulator is divided by 256 (right-shift by 8) to produce one feature value per electrode per second.

  **4-Second Windowed Averaging:** The final feature value is the average of 4 consecutive per-second results:
  - Second 0: mean_power₀ = (1/256) × Σ(z²) for samples 0–255
  - Second 1: mean_power₁ = (1/256) × Σ(z²) for samples 256–511
  - Second 2: mean_power₂
  - Second 3: mean_power₃
  - **Windowed result:** (mean_power₀ + ₁ + ₂ + ₃) >> 2

  This produces **19 features per electrode**, yielding a **23×19 feature matrix** per 4-second window. Storage: 4 scalar values per feature per electrode (23 × 19 × 4 = 1,748 values), far cheaper than buffering raw samples.
  **Note:** The number of electrodes (23) and features (19) are good starting values but not completely fixed.

- **Block D: Feature Vector (19 Features per Window).** Column-wise averaging condenses the 23×19 matrix into a **1×19 tensor**.
  - **Hardware Pruning (Deploy Phase):** While the software model is trained on all 23 channels, the final deployed IC will undergo a channel selection algorithm. The input layer is pruned to process only a **1×5 tensor** from the most informative electrodes, significantly reducing power consumption and the required silicon area.

- **Block E: Temporal Sequence Formation.** The 1×19 vectors are accumulated over time to form an input sequence matrix of dimensions **SEQ_LEN × 19**. This matrix holds the historical context required by the neural network.

### TCN Model Architecture (Block F)

The primary AI engine is a Temporal Convolutional Network (TCN). TCNs avoid the sequential processing bottlenecks of Recurrent Neural Networks (RNNs) by allowing massive parallelization on hardware Multiply-Accumulate (MAC) units.

- **T0: Input Sequence.** Ingests the SEQ_LEN x 19 matrix. 
    
- **T1–T6: Residual Blocks.** The data passes through six sequential residual blocks with exponentially increasing dilation factors: 1, 2, 4, 8, 16, and 32.

  - **Causal Convolutions:** Ensures no "future leakage" occurs by strictly restricting the sliding filters to exclusively process past and present data. [1]

  - **Dilated Convolutions:** The exponential scaling allows the model to expand its receptive field without increasing hardware parameters. [2] The formula for the Receptive Field is calculated as ![formula1](formula1.png). 
  This architecture inherently suppresses the "gridding effect" (blind spots) because the lower, densely sampled layers compress adjacent information before passing it to the highly dilated upper layers.

  - **Block Internals:** Each residual block utilizes Weight Normalization, a Rectified Linear Unit (ReLU) activation, and Spatial Dropout 1D. [3] A skip connection merges the input directly to the output, allowing the convolutional layers to learn the mathematical "deviation" from the identity map, which stabilizes deep network gradients. [4]
    
- **T7: Select Last Time Step.** Discards the historical matrices, passing forward a single vector representing the entire 20-minute contextual memory.

- **T8: Dense Layer.** Consists of 32 Units + ReLU activation.

- **T9: Dropout.** Applies a standard system dropout rate of 0.3.

- **T10: Terminal Dense Layer.** Consists of 4 Units + Softmax activation.
  - **Softmax Hardware Implementation:** True floating-point Softmax is unfeasible on a low-power IC due to the complexity of exponential and division operations. Instead, the hardware will utilize approximate computing techniques, such as fixed-point quadratic interpolation via Look-Up Tables (LUTs) or low-order Taylor series expansions, to calculate probabilities without consuming vast DSP resources. Research indicates quadratic LUT interpolation yields the lowest numerical error for fixed-point hardware.

### Diagnostic Output (Block G)
**Block G: Output Probability Vector.** Classifies the brainwave into four distinct states:
1. **Interictal:** Baseline monitoring between seizures.
2. **Preictal:** The critical warning phase (Triggers the 18-minute primary hardware interrupt).
3. **Ictal:** Active clinical seizure.
4. **Postictal:** Recovery phase following a seizure.

## 4. Global Hardware Constraints

- **System Clock:** 10 ns period (100 MHz). At 256 Hz sampling, this yields **390,625 clock cycles per sample period** — massive headroom for pipelined processing.

- **Fixed-Point Quantization:** The entire pipeline uses fixed-point arithmetic. Block B input: Q8.16 (24-bit). Normalized data and downstream processing: Q4.12 (16-bit). Accumulators: Q14.18 (32-bit). TCN weights and activations: 16-bit fixed-point. This completely eliminates IEEE 754 floating-point arithmetic, saving power and enabling zero-cost bit-shifting for division by powers of 2.

- **Pipeline Architecture (Two Concepts Under Evaluation):**
  - *Concept 1 — TDM:* All blocks share a single hardware pipeline across channels via TDM. No parallel channel duplication. Pipeline utilization ~0.007% for Block B, leaving >99.99% of cycles for downstream blocks.
  - *Concept 2 — Triple-Buffer:* Three rotating memory blocks with 8-wide parallel normalization lanes. Normalization completes in ~100 cycles per second, freeing >99.9999% of cycles for feature extraction and TCN. Higher hardware cost (8× normalization units) but cleaner phase separation.

- **Memory Management (BRAM):** The long receptive field necessitated by the dilation factor of 32 requires complex, hierarchical First-In-First-Out (FIFO) ring buffers constructed from the FPGA's Block RAM to cache intermediate feature maps during continuous streaming inference.

- **Target Form Factor:** Hearing-aid sized. Thermal dissipation and static leakage power are primary constraints. All idle hardware must be clock-gated.

## 5. Recommended Repository Directory Structure
To prevent context-drift and token exhaustion among multi-agent systems (like Claude Code and Google Antigravity), the GitHub repository should strictly mirror the architectural pipeline blocks:
- `/docs/` (Contains this Master Context Document)
- `/src/01_Signal_Conditioning/` (Blocks A-B: Jetson Nano scripts, Z-score normalization)
- `/src/02_Feature_Extraction/` (Blocks C-E: Verilog mean power calculators and temporal sequence matrices)
- `/src/03_TCN_Core/` (Block F, T0-T7: Residual blocks, pipelined MAC arrays, BRAM FIFOs)
- `/src/04_Classification/` (Block F, T8-T10 & Block G: Dense layers, LUT-based Softmax logic)
- `/tb/` (Vivado SystemVerilog testbenches for individual components)

## 6. ADC Front-End: ADS1299EEGFE-PDK Evaluation Board

### Board Architecture
The development platform uses the **Texas Instruments ADS1299EEGFE-PDK** (Performance Demonstration Kit), which consists of two boards:

| Board | Chip | Role |
|---|---|---|
| **ADS1299EEG-FE** (daughter board) | ADS1299 (8-ch, 24-bit delta-sigma ADC) | Analog front-end: electrode connectors, ESD protection, signal conditioning, A/D conversion |
| **MMB0** (motherboard) | TMS320VC5509A DSP | USB bridge: translates SPI ↔ USB for PC-based evaluation software. Proprietary firmware, not user-programmable. |

### ADS1299 Key Specifications

| Property | Value | Relevance |
|---|---|---|
| Channels | 8 per chip | 1 chip covers the 5-channel deployment plan |
| Resolution | 24-bit signed | Maps directly to Q8.16 format |
| Sample Rates | 250, 500, 1k, 2k, 4k, 8k, 16k SPS | Use 250 SPS (~256 Hz). Register-configurable. |
| Interface | **SPI only** | ASIC needs a hardware SPI master to read data |
| DRDY pin | Active-low when data ready | Hardware interrupt — triggers SPI read |
| Data frame | 24-bit status + 24-bit × 8 channels = 216 bits per DRDY | All 8 channels always transmitted, even if unused |
| SPI clock | Up to ~20 MHz | At 20 MHz: ~10.8 μs per frame. DRDY interval: 3,906 μs. Utilization: 0.28% |
| PGA Gain | 1, 2, 4, 6, 8, 12, 24× per channel | Determines full-scale input range |
| Input range | ±VREF/Gain (VREF ≈ 4.5V) | ADC saturates at rails — output always bounded to 24 bits |
| Lead-off detection | Built-in, per channel | Status bits indicate electrode disconnect — usable for anomaly flagging |
| Power | ~6 mW/channel | ~30 mW for 5 channels |

### Programmability
- **ADS1299 chip:** ❌ Not programmable. Fixed-function ADC. Configurable via SPI registers (gain, sample rate, channel enable) but cannot execute custom logic.
- **MMB0 motherboard:** ❌ Practically not programmable. TI loads proprietary firmware at boot. No SDK provided for custom development.
- **Custom processing:** Must be handled externally — either by a custom ASIC, FPGA, or microcontroller connected to the daughter board's SPI headers (bypassing the MMB0).

### SPI Interface Requirements for Custom ASIC
The ASIC (or FPGA prototype) must implement a hardware SPI master to read from the ADS1299. Key considerations:
- **Clock domain crossing:** SPI clock (~20 MHz) is asynchronous to the ASIC core clock (100 MHz). Requires a synchronizer or async FIFO at the boundary.
- **Data extraction:** From the 216-bit frame, extract only the active channel data (5 × 24 bits = 120 bits). Discard unused channels and status word (except lead-off bits).
- **DRDY-driven reads:** The SPI read cycle is triggered by the DRDY falling edge. If a DRDY is missed, that sample is lost.

## 7. Data Validation & Anomaly Handling

### Overflow Concern — Resolved
The ADS1299 ADC output is inherently bounded to 24 bits regardless of input voltage. Extreme electrode voltages (e.g., -2000mV spikes) saturate at the ADC's full-scale rails. No value from the ADC can overflow the Q8.16 register. **No pre-pipeline overflow protection is needed.**

### Anomaly Detection & Replacement Strategy
**Location:** Built into the ASIC's input stage, immediately after SPI data reception. No intermediate controller needed.

**Logic per channel (~50 gates):**
1. Incoming sample is checked against a validity range (comparator) and/or the ADS1299's lead-off status bits.
2. If valid: pass through to memory block, update the "last known good value" register.
3. If invalid: substitute with the **last known good value** (or the next valid value if available).

**Burst Anomaly Concern:** If consecutive anomalies exceed a threshold (e.g., 5+ samples), the entire 1-second window should be flagged as **corrupt** and excluded from feature extraction, rather than padding with stale repeated values that could create artificial patterns.

## 8. Development Phases

### Phase 1: Data Collection & Calibration (Current)
- Use ADS1299EEGFE-PDK board + TI evaluation software to record clinical EEG data.
- Process in MATLAB/Python: compute per-channel μ_τ, σ_τ calibration constants, train TCN model, generate gold reference values.

### Phase 2: FPGA Prototyping
- Bypass the MMB0 motherboard. Wire the ADS1299EEG-FE daughter board's SPI headers directly to an FPGA dev board (e.g., Xilinx Artix-7).
- FPGA implements: SPI master + data validation + triple-buffer pipeline + normalization + feature extraction.
- Verify against gold references from Phase 1.

### Phase 3: ASIC Synthesis
- Translate verified FPGA design to ASIC (standard cell synthesis). Target: hearing-aid form factor.
- Flash patient-specific calibration constants and TCN weights.
- Final deployed device: ADS1299 chip (or smaller variant for fewer channels) + custom ASIC.

## 9. Dual-Track Development Strategy

The project is split into two independent tracks with different disclosure levels:

### Track A — College Final Year Project (Public)
**Topic:** *"Edge-Deployable AI Platform for AI-Based NeuroFusion for EEG Abnormality Measurement"*

Split into two teams:
- **Team 1 (Hardware/Signal):** ADS1299 board interfacing, SPI data acquisition, analog filtering, signal conditioning. Goal: produce a clean, practical **5-electrode × 256-sample data matrix** identical to the software-simulated version.
- **Team 2 (AI/Deployment):** Takes the software-simulated 5×256 matrix as input (assumes Team 1's work is complete). Trains and validates a TCN (or hybrid TCN-LSTM) model for EEG state classification, then deploys onto an edge platform (Jetson Nano or Raspberry Pi 5) for real-time inference. Software simulation results are already complete.

**Presentable in seminars and vivas.** No proprietary IC architecture is disclosed.

### Track B — Custom IC Design (Confidential, Funded IP)
- Full ASIC pipeline: normalization, feature extraction, hardware TCN, softmax
- Concept 1 (TDM) / Concept 2 (Triple-Buffer) architecture decisions
- Fixed-point datapath, clock-gating, BRAM management
- **Not disclosed** in any academic presentations — kept as proprietary IP

### Link Between Tracks
Track A validates the **data entry point** — the same ADS1299 → SPI → conditioned signal path that the ASIC (Track B) will eventually consume. The college project doubles as front-end validation for the IC.

## 10. Reference Papers (Team 2 — Edge AI Deployment)

### Paper 1: NeuroFed-LightTCN
**"NeuroFed-LightTCN: Federated Lightweight Temporal Convolutional Networks for Privacy-Preserving Seizure Detection in EEG Data"**
- **Source:** MDPI Applied Sciences, 2025 — https://www.mdpi.com/2076-3417/15/17/9660
- **Architecture:** Lightweight TCN using depthwise separable convolutions + grouped convolutions + structured pruning. Federated learning for privacy.
- **Results:** 97.11% accuracy, 56 ms inference latency, 44.9% parameter reduction (65.4M → 34.9M). At 70% pruning: 97.23% recall, 96.99% specificity, 97.17% F1.
- **Relevance:** Demonstrates depthwise separable convolutions as a compression technique for edge TCNs — directly applicable to our model optimization before Jetson/RPi deployment. Pruning methodology is a reference for our quantization pipeline.

### Paper 2: Multi-Scale TCN on Edge Devices
**"Epileptic Seizure Detection on Resource Constrained Edge Devices Using Multi-Scale Temporal Convolutional Networks"**
- **Source:** Semantic Scholar / Conference Proceedings, April 2026 — https://www.semanticscholar.org/paper/fe42504b3bed85f6045c0dc58656187f16951b22
- **Authors:** Rahul Chiranjeevi V., P. Ezhumalai, Sharmila V.
- **Architecture:** Multi-Scale TCN (MS-TCN) capturing temporal patterns at multiple scales simultaneously. Optimized for edge deployment with reduced complexity.
- **Relevance:** Most directly aligned with our project — same goal (seizure detection on edge hardware using TCN). The multi-scale approach is an architectural option for capturing both short-term transients and long-term EEG trends.

### Paper 3: Energy-Efficient Tiny AI for Seizure Detection
**"Sustainable E-Health: Energy-Efficient Tiny AI for Epileptic Seizure Detection via EEG"**
- **Source:** PMC / Biomedical Engineering and Computational Biology, 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12336407/
- **DOI:** 10.1177/11795972241283101
- **Authors:** Hizem M., Aoueileyine M.O.E., Belhaouari S.B., EL Omri A., Bouallegue R.
- **Architecture:** TinyML workflow using ExtraTrees Classifier (not deep learning), converted to TFLite for microcontroller deployment. Dataset: Bonn EEG (4097 recordings/patient, 500 patients).
- **Results:** 99.6% AUC. After TFLite conversion with pruning + INT8 quantization: model size reduced ~10× to 256 KB, inference time 0.002185s, accuracy practically unchanged. Deployed on Raspberry Pi 4.
- **Relevance:** Provides a complete TinyML deployment workflow reference (train → prune → quantize → TFLite → RPi). The 256 KB model size benchmark is useful for comparing against our TCN model footprint. However, uses classical ML (ExtraTrees) not deep learning — our TCN will be larger and may need more aggressive compression.

### Paper 4: TCN-LSTM Hybrid for Seizure Detection
**"An Epileptic Seizure Detection Method Based on TCN-LSTM"**
- **Source:** ACM Digital Library, 2024 — https://dl.acm.org/doi/fullHtml/10.1145/3644116.3644180
- **Architecture:** End-to-end TCN-BiLSTM hybrid. TCN for parallel local feature extraction, BiLSTM for long-range temporal dependencies. Preprocessing: 0.5–45 Hz band-pass filter. Post-processing: moving average + thresholding + collar technique.
- **Results (CHB-MIT):** 97.09% accuracy, 94.31% sensitivity, 97.13% specificity, FDR 0.38/h. Detection time: 5.65s per 1 hour of EEG.
- **Results (SH-SDU):** 93.27% accuracy, 94.99% sensitivity, 99.35% event sensitivity.
- **Relevance:** Validates the TCN-LSTM hybrid architecture that closely mirrors our Block F design. The 0.5–45 Hz band-pass filter specification is useful for Team 1's signal conditioning stage. The collar technique for post-processing is a practical refinement we could adopt.

