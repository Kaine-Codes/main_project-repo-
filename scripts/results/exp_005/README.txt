============================================================
  Experiment: exp_005
============================================================

Selection metric: preictal_f1
Best preictal_f1: 0.5144 (epoch 13)

--- Validation ---
  Macro F1:          0.6775
  Balanced Accuracy: 0.6602
  Preictal      P=0.453  R=0.594  F1=0.514  (support=900, predicted=1180)
  Ictal         P=0.969  R=0.787  F1=0.869  (support=80, predicted=65)
  Interictal    P=0.716  R=0.642  F1=0.677  (support=2077, predicted=1864)
  Postictal     P=0.686  R=0.616  F1=0.649  (support=511, predicted=459)

--- Test ---
  Macro F1:          0.6931
  Balanced Accuracy: 0.6601
  Preictal      P=0.877  R=0.683  F1=0.768  (support=900, predicted=701)
  Ictal         P=0.619  R=0.530  F1=0.571  (support=132, predicted=113)
  Interictal    P=0.755  R=0.920  F1=0.829  (support=1910, predicted=2329)
  Postictal     P=0.746  R=0.506  F1=0.603  (support=626, predicted=425)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 13 (best) / 60 (max)
Feature scaling: none
Class weighting: none
Oversampling: none
Seed: 42
