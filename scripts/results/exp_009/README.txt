============================================================
  Experiment: exp_009
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6535 (epoch 22)

--- Validation ---
  Macro F1:          0.6535
  Balanced Accuracy: 0.6709
  Preictal      P=0.410  R=0.794  F1=0.540  (support=900, predicted=1746)
  Ictal         P=0.985  R=0.812  F1=0.890  (support=80, predicted=66)
  Interictal    P=0.736  R=0.439  F1=0.550  (support=2077, predicted=1238)
  Postictal     P=0.629  R=0.638  F1=0.634  (support=511, predicted=518)

--- Test ---
  Macro F1:          0.7262
  Balanced Accuracy: 0.7217
  Preictal      P=0.885  R=0.767  F1=0.821  (support=900, predicted=780)
  Ictal         P=0.485  R=0.614  F1=0.542  (support=132, predicted=167)
  Interictal    P=0.798  R=0.901  F1=0.846  (support=1910, predicted=2157)
  Postictal     P=0.817  R=0.605  F1=0.695  (support=626, predicted=464)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 22 (best) / 60 (max)
Feature scaling: none
Class weighting: focal
Oversampling: seizure
Seed: 42
