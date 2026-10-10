# PROJECT ABSTRACT

## Edge-Deployable AI Platform for AI-Based NeuroFusion for EEG Abnormality Measurement

**Course:** B.Tech Final Year Project (Academic Year 2026–2027)  
**Department:** Electronics & Communication Engineering  
**Project Track:** Track A (Public/Academic Development)

---
  
## 1. Project Overview & Abstract

This project aims to develop a real-time, edge-deployable artificial intelligence platform to automate the detection and classification of electroencephalography (EEG) abnormalities, with a focus on epileptic seizure prediction. Continuous scalp EEG monitoring remains the gold standard for clinical diagnosis of epilepsy; however, traditional diagnostics are severely constrained by the need for manual, expert interpretation and the computational latency of cloud-dependent classification systems. 

To resolve these limitations, this project presents a dual-stage, non-cloud dependent edge AI system. 
* **Stage 1 (Hardware Signal Interface):** Focuses on multi-channel data acquisition via a Texas Instruments ADS1299 Analog Front-End (AFE) board. It samples 5 scalp EEG electrodes simultaneously at 256 Hz with 24-bit resolution, transferring data via serial peripheral interface (SPI) to an edge controller. The signals undergo digital band-pass filtering (0.5–45 Hz) and Z-score normalization to eliminate muscular artifacts, DC drift, and line noise, producing a standardized $5 \times 256$ sample matrix every second.
* **Stage 2 (Edge Deep Learning Inference):** Takes the standardized data matrix and performs local, low-latency state classification. A Temporal Convolutional Network (TCN) optimized with Depthwise Separable Convolutions (DSC) is trained to identify four clinical brain states: Interictal (baseline), Preictal (pre-seizure warning), Ictal (active seizure), and Postictal (recovery). The trained network is compressed using structured filter pruning and INT8 quantization, then compiled via TensorFlow Lite / NVIDIA TensorRT for execution on edge platforms (Raspberry Pi 5 / NVIDIA Jetson Nano).

The proposed system targets an edge inference latency of less than 100 milliseconds and a classification accuracy exceeding 95%, establishing a robust, privacy-preserving, and portable diagnostic tool suitable for clinical and ambulatory point-of-care environments.

---

## 2. Background & Introduction

Epilepsy is a chronic neurological disorder characterized by recurrent, unprovoked seizures, affecting over 50 million individuals globally. A key clinical challenge in epilepsy management is the identification of the **preictal state**—a transition period that typically precedes physical seizure onset by up to 20 minutes. Reliable, automated identification of this state would provide a critical window (2–4 minutes margin) for therapeutic intervention, such as automated drug delivery or warning alerts to caregivers.

Traditional deep learning models for EEG classification, such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, struggle with sequential processing bottlenecks and vanishing gradients when modeling long EEG time-series. While Temporal Convolutional Networks (TCNs) overcome these limitations through parallelized dilated causal convolutions, they are typically computationally heavy. Running these networks in the cloud introduces data privacy risks, network latency (200–500ms), and a critical dependency on continuous internet connectivity.

This project addresses this deployment gap by implementing hardware-aware optimization techniques to deploy deep learning models directly on localized edge single-board computers (SBCs), achieving sustainable, real-time, on-device biomedical classification.

---

## 3. Problem Statement

Current clinical implementations of automated EEG monitoring are bottlenecked by:
1. **Cloud Latency & Reliability:** Reliance on cloud servers creates a single point of failure in critical alert systems. Connectivity drops can lead to missed detection events, and data transfer latency slows real-time clinical intervention.
2. **Patient Data Privacy:** Transmitting raw, high-resolution EEG signals containing sensitive brain wave signatures over public networks violates strict medical privacy regulations (such as HIPAA).
3. **Compute and Power Constraints on the Edge:** High-performance neural network architectures (TCNs, hybrids) require millions of parameters, which quickly exhaust the memory and thermal budgets of portable, battery-powered edge hardware.
4. **Signal Artifact Contamination:** Raw scalp-recorded EEG signals are highly susceptible to movement artifacts, power line hum (50/60 Hz), and baseline wander, necessitating real-time signal conditioning before feature extraction.

---

## 4. Objectives

The primary objectives of the project are:
* **Real-Time Data Acquisition:** Design an SPI-driven interface between the 24-bit ADS1299 EEG front-end and the edge controller, ensuring zero-loss streaming at 256 Hz across 5 channels.
* **Effective Signal Conditioning:** Implement digital band-pass filtering (0.5–45 Hz) and channel-wise Z-score normalization to mitigate noise and generate clean, standardized input matrices.
* **Deep Learning Model Optimization:** Design a lightweight TCN architecture utilizing depthwise separable convolutions to extract temporal features at multiple scales.
* **On-Device Deployment:** Quantize the trained model from FP32 to INT8 precision and prune redundant filters, achieving a model footprint of $<5\text{ MB}$.
* **Low-Latency Edge Inference:** Deploy the optimized model using TensorFlow Lite (TFLite) or TensorRT on a Raspberry Pi 5 or NVIDIA Jetson Nano, demonstrating real-time inference with an execution latency of $<100\text{ ms}$ and a target accuracy of $>95\%$.

---

## 5. System Architecture & Data Flow

The system flows through a feed-forward pipeline from raw analog signal capture to diagnostic output classification:

```
[EEG Electrodes (5 Ch)]
          │ (Analog signal)
          ▼
[TI ADS1299 AFE Board]
          │ (Raw 24-bit samples @ 256 Hz via SPI)
          ▼
[Signal Conditioning Block] ─── (0.5–45 Hz Band-pass Filter & Z-Score Normalization)
          │
          ▼
[5 × 256 Normalized Matrix]
          │ (Standardized 1-Second Windows)
          ▼
[Feature Extraction Block] ─── (Computes 19 statistical/spectral features per channel)
          │
          ▼
[Temporal Sequence Formation] ── (Slices data into overlapping multi-second sequences)
          │
          ▼
[TCN Deep Learning Model] ──── (6 Residual Blocks with dilated causal convolutions)
          │
          ▼
[Post-Processing Block] ────── (Moving Average Smoothing, Thresholding & Collar Margin)
          │
          ▼
[4-State Diagnostic Output] ── (Interictal | Preictal | Ictal | Postictal)
```

---

## 6. Technical Specifications (Hardware & Software)

### Hardware Components
* **Analog Front-End:** Texas Instruments ADS1299EEG-FE Performance Demonstration Kit.
  * 8 low-noise, PGA-integrated channels (5 active).
  * 24-bit Delta-Sigma Analog-to-Digital Converter.
  * Active-low Data Ready ($\overline{\text{DRDY}}$) hardware interrupt pin.
* **Edge Processing Unit:** 
  * *Option A:* Raspberry Pi 5 (Quad-core ARM Cortex-A76 @ 2.4 GHz, 8GB RAM).
  * *Option B:* NVIDIA Jetson Nano (128-core Maxwell GPU, Quad-core ARM A57 @ 1.43 GHz, 4GB LPDDR4).
* **Communication Interface:** Dedicated SPI Bus (Master on Edge Unit, Slave on ADS1299) operating at up to 20 MHz.

### Software Stack
* **Operating System:** Linux (Ubuntu Core / Raspberry Pi OS Lite).
* **Development Libraries:** PyTorch (Model Training), SciPy/NumPy (Signal Filtering).
* **Embedded Inference Engines:** TensorFlow Lite (TFLite Runtime) / NVIDIA TensorRT.
* **Dataset Benchmarks:** CHB-MIT Scalp EEG Database (MIT) and the Bonn University EEG Database.

---

## 7. Deep Learning & Edge Optimization Methodology

To adapt the Temporal Convolutional Network (TCN) for efficient edge execution, the project employs several optimization strategies:

1. **Depthwise Separable Convolutions (DSC):**
   Standard TCN layers are replaced with DSC, splitting convolutions into a spatial *depthwise* stage (filtering each EEG channel independently) and a *pointwise* stage (a $1 \times 1$ convolution combining channel outputs). This reduces parameter counts by up to 45% with negligible impact on accuracy.
2. **Multi-Scale Convolutional Kernels:**
   The network uses parallel convolutional branches with varying kernel sizes (e.g., $k=3, 7, 15$) to simultaneously capture high-frequency transients (spikes) and slow-wave rhythmic variations.
3. **Structured Network Pruning:**
   Post-training, filters with low weight magnitude are systematically pruned to reduce floating-point operations (FLOPs) and inference latency.
4. **Integer Quantization (INT8):**
   Using Post-Training Quantization (PTQ), model weights and activation thresholds are mapped from float32 to signed 8-bit integers. This shrinks the model file size by approximately 75% and accelerates execution on the edge CPU/GPU.
5. **Post-Processing collar Technique:**
   The output probability vector undergoes moving average smoothing followed by a collar window technique, extending the boundaries of positive detections to stabilize seizure warnings and suppress transient false positives.

---

## 8. Work Division Matrix

The project development is divided between two independent, collaborative tracks:

### Team 1: Hardware Interfacing & Signal Conditioning
* **SPI Driver Development:** Writing high-speed SPI drivers to interface with the ADS1299, handling the 24-bit status and channel words under strict timing constraints driven by the $\overline{\text{DRDY}}$ interrupt.
* **Signal Conditioning:** Designing and implementing digital IIR band-pass filters (0.5–45 Hz) to isolate clinical EEG frequencies and reject power line interference.
* **Normalization Logic:** Implementing real-time channel-wise Z-score normalization using calibration constants (mean $\mu$ and standard deviation $\sigma$) calculated from seizure-free segments.
* **Integration Output:** Exposing a standardized, clean $5 \times 256$ normalized sample matrix output for the downstream AI engine.

### Team 2: AI Architecture & Edge Deployment
* **Model Design & Training:** Designing the TCN-BiLSTM hybrid network architecture using PyTorch and training it on the clinical CHB-MIT dataset.
* **Quantization & Pruning:** Implementing the PyTorch/TensorFlow optimization pipeline (Structured Pruning, INT8 Quantization).
* **Edge Compilation:** Converting the trained and pruned models into TFLite/TensorRT formats.
* **Hardware Benchmarking:** Deploying the compiled models on the Raspberry Pi 5 / Jetson Nano and profiling latency (ms), model size (KB), power consumption (W), and classification accuracy.

---

## 9. Expected Deliverables & Evaluation Metrics

The project performance will be evaluated against the following criteria:

* **Classification Performance:**
  * Sensitivity (True Positive Rate) $\ge 95\%$ on the CHB-MIT test database.
  * False Detection Rate (FDR) $\le 0.5$ false alarms per hour.
* **Latency Profile:**
  * End-to-end inference latency $\le 100\text{ ms}$ on the target edge device (Raspberry Pi 5 / Jetson Nano).
* **Computational Footprint:**
  * Deployed model storage footprint $\le 5\text{ MB}$.
  * System memory (RAM) usage $\le 500\text{ MB}$ during active inference.
* **Hardware Interfacing:**
  * Zero frame drop rate on the SPI bus over 1 hour of continuous streaming.

---

## 10. Key References

1. **Bai, S., Kolter, J. Z., & Koltun, V. (2018).** *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv preprint arXiv:1803.01271. (Introduced the dilated causal TCN architecture).
2. **Hizem, M., Aoueileyine, M. O. E., Belhaouari, S. B., EL Omri, A., & Bouallegue, R. (2025).** *Sustainable E-Health: Energy-Efficient Tiny AI for Epileptic Seizure Detection via EEG.* Biomedical Engineering and Computational Biology. DOI: 10.1177/11795972241283101. (Demonstrated a complete TinyML quantization and deployment workflow on Raspberry Pi).
3. **Ezhumalai, P., Chiranjeevi V., R., & Sharmila, V. (2026).** *Epileptic Seizure Detection on Resource Constrained Edge Devices Using Multi-Scale Temporal Convolutional Networks.* Conference Proceedings, Semantic Scholar. (Validated multi-scale TCN models for localized edge deployment).
4. **ACM Digital Library. (2024).** *An Epileptic Seizure Detection Method Based on TCN-LSTM.* DOI: 10.1145/3644116.3644180. (Evaluated TCN-BiLSTM hybrid networks with band-pass filtering and moving average post-processing).
