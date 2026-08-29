# EEG-TCN Seizure Prediction Project Context

## 1. Project Overview
This project implements an end-to-end machine learning pipeline for real-time EEG seizure prediction and detection. The goal is to classify continuous EEG data into 4 distinct clinical states (**Interictal, Preictal, Ictal, and Postictal**) using a Temporal Convolutional Network (TCN). 

The architecture is specifically designed to be lightweight (Depthwise-Separable Convolutions) so that it can be deployed on edge devices, such as a Raspberry Pi 5, for real-time monitoring.

## 2. Dataset Strategy (CHB-MIT)
The pipeline is built around the **CHB-MIT Scalp EEG Database**. 
- **The Data**: The dataset consists of `.edf` (European Data Format) continuous recordings across multiple pediatric patients, along with `*-summary.txt` files containing annotations for seizure start and end times.
- **The 4 States**:
  - `0 - Interictal`: Normal baseline brainwave activity.
  - `1 - Preictal`: The period immediately preceding a seizure (configured to 30 minutes prior to onset).
  - `2 - Ictal`: The actual seizure event.
  - `3 - Postictal`: The recovery period immediately following a seizure (configured to 30 minutes after offset).
  
### Cross-Patient Generalization Split
Because EEG patterns vary significantly between different human brains, the dataset is split to test for **true generalization**:
- **Test Set (`test/` folder)**: A manually curated folder containing specific held-out recordings (e.g., 1 seizure file and 1 non-seizure file per patient). This data is completely hidden during training to act as the ultimate test of real-world performance.
- **Train Set**: All other patient folders (`chb01`, `chb05`, etc.). These are compiled into one massive multi-patient dataset so the model can learn universal seizure patterns.

---

## 3. The Pipeline Scripts

### Step 1: Feature Extraction (`preprocess_chbmit.py`)
This script operates at the single-EDF level. It converts raw EEG voltage traces into dense, machine-learning-friendly feature vectors.
- **Methodology**:
  1. Loads an EDF using `mne` and selects 5 specific bipolar channels.
  2. Applies a 0.5–45 Hz bandpass FIR filter.
  3. Z-score normalizes the data per channel.
  4. Segments the continuous recording into overlapping **1-second windows**.
  5. Computes 5 mathematical features per channel per window: *Hjorth Activity, Hjorth Mobility, Hjorth Complexity, Zero Crossing Rate, and Theta/Alpha Band-Power Ratio*.
  6. Labels the window based on the timestamp parsed from the patient's `summary.txt`.
- **Key Parameters & Variables**:
  - `CHANNELS = ['FP1-F7', 'F7-T7', 'FP2-F8', 'F8-T8', 'FZ-CZ']`
  - `TARGET_SFREQ = 256` (Sampling frequency)
  - `BANDPASS_LOW = 0.5`, `BANDPASS_HIGH = 45.0` (Hz)
  - `WINDOW_SEC = 1.0` (Seconds per window)
  - `DEFAULT_OVERLAP = 0.5` (50% overlap between consecutive windows)
  - `PREICTAL_SEC = 30 * 60` (Seconds before onset labeled Preictal)
  - `POSTICTAL_SEC = 30 * 60` (Seconds after offset labeled Postictal)
  - `THETA_BAND = (4.0, 8.0)`, `ALPHA_BAND = (8.0, 13.0)` (Frequency bands for power ratio)
- **Output**: Returns a flat 2D matrix of shape `(n_windows, 25)` where 25 is (5 channels × 5 features).

### Step 2: Sequence Compilation (`build_train_test.py`)
The TCN requires temporal context (sequences of windows), not just isolated 1-second snapshots. This script builds the final datasets while protecting the timeline.
- **Methodology**:
  1. Iterates through all patient directories (excluding the `test` directory for training, and specifically targeting it for testing).
  2. Calls `preprocess_chbmit.py` on every valid EDF file.
  3. Converts the 2D feature matrix into 3D sliding sequences strictly within the boundaries of a single recording.
  4. Thin out overlaps using a stride parameter to massively reduce RAM usage without losing timeline continuity.
- **Key Parameters & Variables**:
  - `SEQ_LEN = 64` (Number of 1-second windows in a single temporal sequence passed to the TCN).
  - `SEQ_STRIDE = 8` (Keeps every 8th overlapping sequence to reduce memory/data redundancy).
  - `ROOT = 'CHB DATASET'`
  - `TEST_FOLDER_NAME = 'test'`
- **Output**: Saves `chb_train.npz` and `chb_test.npz`. Inside are 3D arrays of shape `(Total_Sequences, 64, 25)`.

### Step 3: Class Balancing (`undersample_interictal.py`)
EEG data is massively imbalanced. We must reduce the overwhelming majority class (`Interictal`) so the network doesn't collapse to predicting only Class 0.
- **Methodology**: 
  - Randomly drops sequences from the majority `Interictal` class until its count matches a target number (usually matching the `Preictal` class count).
  - *Note*: We ONLY run this on the training data. The test set is left unbalanced.
- **Key Parameters & Variables**:
  - `INTERICTAL_LABEL = 0`
  - `--target` CLI argument: The desired exact count for Class 0 (determined by looking at the Class 1 count after running the build script).
- **Output**: Generates `chb_train_balanced.npz` with heavily reduced Class 0 sequences.

### Step 4: Model Training (`eeg_tcn.py`)
This script defines the neural network architecture and runs the PyTorch training loop.
- **Methodology**:
  - **Architecture**: A 6-block Temporal Convolutional Network (TCN). It uses **Depthwise Separable Convolutions** and exponential dilation to achieve a massive receptive field while keeping the parameter count incredibly low.
  - **Loss Function & Weights**: Uses `CrossEntropyLoss`. Because the `Ictal` class is still tiny even after undersampling `Interictal`, the script dynamically calculates **Inverse Frequency Class Weights** (`weights = total_samples / (4.0 * class_counts)`).
  - **Evaluation**: At the end of training (or upon Early Stopping), it re-loads the best model weights and runs a final pass over the Test set, computing a Confusion Matrix and Classification Report.
- **Key Parameters & Variables (Default CLI Config)**:
  - `channel_sizes = (32, 32, 48, 48, 64, 64)` (Filters per TCN block)
  - `kernel_size = 3`
  - `--epochs 50`
  - `--lr 1e-3` (Adam Optimizer Learning Rate)
  - `--weight_decay 1e-4`
  - `--dropout 0.2`
  - `--batch_size 64`
  - `--patience 10` (Early stopping patience)
- **Output**: 
  1. Saves the trained neural network weights to `eeg_tcn_best.pth`.
  2. Prints a **Confusion Matrix** and **Classification Report**.
