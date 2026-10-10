============================================================
  Experiment: exp_006
============================================================

Selection metric: preictal_f1
Best preictal_f1: 0.5670 (epoch 25)

--- Validation ---
  Macro F1:          0.6242
  Balanced Accuracy: 0.6694
  Preictal      P=0.478  R=0.698  F1=0.567  (support=900, predicted=1315)
  Ictal         P=0.642  R=0.850  F1=0.731  (support=80, predicted=106)
  Interictal    P=0.720  R=0.568  F1=0.635  (support=2077, predicted=1639)
  Postictal     P=0.565  R=0.562  F1=0.563  (support=511, predicted=508)

--- Test ---
  Macro F1:          0.2490
  Balanced Accuracy: 0.3954
  Preictal      P=0.689  R=0.069  F1=0.125  (support=900, predicted=90)
  Ictal         P=0.042  R=0.773  F1=0.079  (support=132, predicted=2452)
  Interictal    P=0.210  R=0.048  F1=0.078  (support=1910, predicted=438)
  Postictal     P=0.736  R=0.692  F1=0.713  (support=626, predicted=588)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 25 (best) / 80 (max)
Feature scaling: robust
Class weighting: dynamic
Oversampling: seizure
Seed: 42
