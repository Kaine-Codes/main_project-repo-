============================================================
  Experiment: exp_001
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6769 (epoch 13)

--- Validation ---
  Macro F1:          0.6769
  Balanced Accuracy: 0.6599
  Preictal      P=0.438  R=0.601  F1=0.507  (support=900, predicted=1236)
  Ictal         P=1.000  R=0.787  F1=0.881  (support=80, predicted=63)
  Interictal    P=0.713  R=0.617  F1=0.662  (support=2077, predicted=1796)
  Postictal     P=0.685  R=0.634  F1=0.659  (support=511, predicted=473)

--- Test ---
  Macro F1:          0.7257
  Balanced Accuracy: 0.6831
  Preictal      P=0.886  R=0.697  F1=0.780  (support=900, predicted=708)
  Ictal         P=0.747  R=0.538  F1=0.626  (support=132, predicted=95)
  Interictal    P=0.769  R=0.928  F1=0.841  (support=1910, predicted=2303)
  Postictal     P=0.773  R=0.570  F1=0.656  (support=626, predicted=462)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 13 (best) / 60 (max)
Feature scaling: none
Class weighting: none
Oversampling: none
Seed: 42
