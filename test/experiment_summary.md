# EEG Feature Relevance Experiment Summary

## Objective
The primary goal of this experiment was to analyze which EEG features and channels are most relevant for detecting **preictal** (pre-seizure) and **ictal** (during-seizure) moments. By comparing the EEG signal characteristics across different seizure stages against a baseline, the experiment aims to identify reliable biomarkers that could be used for early seizure prediction and detection.

## Methodology

### 1. Feature Extraction (`channel_analysis.py`)
The pipeline processes the `chb08_05.edf` EEG recording, extracting data from **23 channels**. The recording is segmented into four distinct stages:
- **Interictal**: Baseline/normal state (60s - 660s)
- **Preictal**: Period leading up to the seizure (2256s - 2856s)
- **Ictal**: The actual seizure event (2856s - 3046s)
- **Postictal**: The recovery period following the seizure (3046s onwards)

For each 1-second window (with 50% overlap), 8 specific features are calculated:
1. **Hjorth Activity** (Signal variance/power)
2. **Hjorth Mobility** (Mean frequency)
3. **Hjorth Complexity** (Change in frequency)
4. **Zero Crossing Rate (ZCR)** (Signal frequency changes)
5. **Alpha/Theta ratio** (Band-power ratio)
6. **Higuchi Fractal Dimension (FD)** (Signal complexity)
7. **Gamma/Beta ratio** (Band-power ratio)
8. **Theta/Beta ratio** (Band-power ratio)

### 2. Analysis and Heatmap Generation (`heatmap_gen.py`)
The extracted features are then aggregated to compute the **median value** for each feature, per channel, across all four stages. 

To determine the relevance of each feature for detecting seizures, the pipeline calculates the **percentage change** of the preictal, ictal, and postictal stages relative to the **interictal (baseline) stage**. 

The results are outputted in two main formats:
- **Excel Summaries (`EEG_Channel_Stage_Analysis.xlsx`)**: Contains detailed median values and percentage changes. Notably, it includes a `Preictal_Summary` that highlights channels where features exhibited a **>20% absolute change**, making it easy to identify the most responsive channels leading up to a seizure.
- **Heatmaps (`outputs/EEG_Heatmaps/`)**: 
  - **Stage Heatmaps**: Visualizes the raw median values of features across stages and channels.
  - **Change Heatmaps**: Diverging color maps showing the percentage change from the interictal baseline. These are crucial for visually spotting which features spike or drop significantly during the preictal and ictal phases.

## Conclusion
By observing the heatmaps and the summary tables (specifically the `Preictal_Change` and `Ictal_Change` sheets), we can pinpoint which of the 8 features are highly sensitive to the onset of a seizure. Features that consistently show significant deviation (e.g., >20% change) during the preictal stage are the most valuable for predictive models.
