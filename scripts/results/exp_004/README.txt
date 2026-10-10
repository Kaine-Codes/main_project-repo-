============================================================
  Experiment: exp_004
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6185 (epoch 12)

--- Validation ---
  Macro F1:          0.6185
  Balanced Accuracy: 0.6431
  Preictal      P=0.428  R=0.611  F1=0.503  (support=900, predicted=1285)
  Ictal         P=0.690  R=0.863  F1=0.767  (support=80, predicted=100)
  Interictal    P=0.686  R=0.588  F1=0.633  (support=2077, predicted=1779)
  Postictal     P=0.646  R=0.511  F1=0.570  (support=511, predicted=404)

--- Test ---
  Macro F1:          0.2585
  Balanced Accuracy: 0.4113
  Preictal      P=0.526  R=0.011  F1=0.022  (support=900, predicted=19)
  Ictal         P=0.047  R=0.780  F1=0.089  (support=132, predicted=2180)
  Interictal    P=0.582  R=0.208  F1=0.307  (support=1910, predicted=684)
  Postictal     P=0.590  R=0.645  F1=0.616  (support=626, predicted=685)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 12 (best) / 60 (max)
Feature scaling: robust
Class weighting: focal
Oversampling: seizure
Seed: 42
