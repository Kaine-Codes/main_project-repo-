============================================================
  Experiment: exp_011
============================================================

Selection metric: preictal_f1
Best preictal_f1: 0.5710 (epoch 37)

--- Validation ---
  Macro F1:          0.5791
  Balanced Accuracy: 0.6823
  Preictal      P=0.413  R=0.927  F1=0.571  (support=900, predicted=2021)
  Ictal         P=0.563  R=0.838  F1=0.673  (support=80, predicted=119)
  Interictal    P=0.778  R=0.327  F1=0.460  (support=2077, predicted=873)
  Postictal     P=0.587  R=0.638  F1=0.612  (support=511, predicted=555)

--- Test ---
  Macro F1:          0.7329
  Balanced Accuracy: 0.7649
  Preictal      P=0.897  R=0.829  F1=0.861  (support=900, predicted=832)
  Ictal         P=0.405  R=0.674  F1=0.506  (support=132, predicted=220)
  Interictal    P=0.844  R=0.839  F1=0.841  (support=1910, predicted=1900)
  Postictal     P=0.729  R=0.717  F1=0.723  (support=626, predicted=616)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 37 (best) / 60 (max)
Feature scaling: none
Class weighting: focal
Oversampling: seizure
Seed: 42
