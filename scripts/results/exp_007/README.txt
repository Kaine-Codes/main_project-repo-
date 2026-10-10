============================================================
  Experiment: exp_007
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6864 (epoch 11)

--- Validation ---
  Macro F1:          0.6864
  Balanced Accuracy: 0.6714
  Preictal      P=0.443  R=0.636  F1=0.522  (support=900, predicted=1291)
  Ictal         P=1.000  R=0.812  F1=0.897  (support=80, predicted=65)
  Interictal    P=0.721  R=0.611  F1=0.662  (support=2077, predicted=1761)
  Postictal     P=0.710  R=0.626  F1=0.665  (support=511, predicted=451)

--- Test ---
  Macro F1:          0.7172
  Balanced Accuracy: 0.6631
  Preictal      P=0.882  R=0.673  F1=0.764  (support=900, predicted=687)
  Ictal         P=0.860  R=0.561  F1=0.679  (support=132, predicted=86)
  Interictal    P=0.744  R=0.942  F1=0.832  (support=1910, predicted=2418)
  Postictal     P=0.790  R=0.476  F1=0.594  (support=626, predicted=377)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 11 (best) / 60 (max)
Feature scaling: none
Class weighting: none
Oversampling: seizure
Seed: 42
