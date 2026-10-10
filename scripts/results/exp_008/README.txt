============================================================
  Experiment: exp_008
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6547 (epoch 23)

--- Validation ---
  Macro F1:          0.6547
  Balanced Accuracy: 0.6505
  Preictal      P=0.421  R=0.592  F1=0.492  (support=900, predicted=1266)
  Ictal         P=1.000  R=0.800  F1=0.889  (support=80, predicted=64)
  Interictal    P=0.692  R=0.558  F1=0.618  (support=2077, predicted=1675)
  Postictal     P=0.591  R=0.652  F1=0.620  (support=511, predicted=563)

--- Test ---
  Macro F1:          0.7519
  Balanced Accuracy: 0.7127
  Preictal      P=0.887  R=0.660  F1=0.757  (support=900, predicted=670)
  Ictal         P=0.758  R=0.568  F1=0.649  (support=132, predicted=99)
  Interictal    P=0.785  R=0.934  F1=0.853  (support=1910, predicted=2274)
  Postictal     P=0.821  R=0.688  F1=0.749  (support=626, predicted=525)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 23 (best) / 60 (max)
Feature scaling: none
Class weighting: focal
Oversampling: none
Seed: 42
