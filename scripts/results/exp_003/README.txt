============================================================
  Experiment: exp_003
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6643 (epoch 10)

--- Validation ---
  Macro F1:          0.6643
  Balanced Accuracy: 0.6527
  Preictal      P=0.482  R=0.510  F1=0.495  (support=900, predicted=953)
  Ictal         P=0.971  R=0.838  F1=0.899  (support=80, predicted=69)
  Interictal    P=0.682  R=0.662  F1=0.672  (support=2077, predicted=2017)
  Postictal     P=0.580  R=0.601  F1=0.590  (support=511, predicted=529)

--- Test ---
  Macro F1:          0.2844
  Balanced Accuracy: 0.4253
  Preictal      P=0.688  R=0.120  F1=0.204  (support=900, predicted=157)
  Ictal         P=0.049  R=0.803  F1=0.093  (support=132, predicted=2144)
  Interictal    P=0.317  R=0.109  F1=0.162  (support=1910, predicted=657)
  Postictal     P=0.687  R=0.669  F1=0.678  (support=626, predicted=610)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 10 (best) / 60 (max)
Feature scaling: robust
Class weighting: none
Oversampling: seizure
Seed: 42
