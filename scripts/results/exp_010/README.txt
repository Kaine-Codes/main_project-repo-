============================================================
  Experiment: exp_010
============================================================

Selection metric: preictal_f1
Best preictal_f1: 0.5379 (epoch 17)

--- Validation ---
  Macro F1:          0.6556
  Balanced Accuracy: 0.6770
  Preictal      P=0.406  R=0.796  F1=0.538  (support=900, predicted=1762)
  Ictal         P=1.000  R=0.825  F1=0.904  (support=80, predicted=66)
  Interictal    P=0.746  R=0.428  F1=0.544  (support=2077, predicted=1192)
  Postictal     P=0.615  R=0.659  F1=0.636  (support=511, predicted=548)

--- Test ---
  Macro F1:          0.7584
  Balanced Accuracy: 0.7361
  Preictal      P=0.894  R=0.740  F1=0.810  (support=900, predicted=745)
  Ictal         P=0.611  R=0.583  F1=0.597  (support=132, predicted=126)
  Interictal    P=0.812  R=0.928  F1=0.866  (support=1910, predicted=2182)
  Postictal     P=0.843  R=0.693  F1=0.761  (support=626, predicted=515)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 17 (best) / 80 (max)
Feature scaling: none
Class weighting: dynamic
Oversampling: seizure
Seed: 42
