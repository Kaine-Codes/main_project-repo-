============================================================
  Experiment: exp_002
============================================================

Selection metric: macro_f1
Best macro_f1: 0.6628 (epoch 11)

--- Validation ---
  Macro F1:          0.6628
  Balanced Accuracy: 0.6370
  Preictal      P=0.455  R=0.481  F1=0.468  (support=900, predicted=951)
  Ictal         P=1.000  R=0.825  F1=0.904  (support=80, predicted=66)
  Interictal    P=0.674  R=0.686  F1=0.680  (support=2077, predicted=2114)
  Postictal     P=0.650  R=0.556  F1=0.599  (support=511, predicted=437)

--- Test ---
  Macro F1:          0.3267
  Balanced Accuracy: 0.4568
  Preictal      P=0.395  R=0.204  F1=0.269  (support=900, predicted=466)
  Ictal         P=0.056  R=0.674  F1=0.104  (support=132, predicted=1579)
  Interictal    P=0.397  R=0.156  F1=0.224  (support=1910, predicted=751)
  Postictal     P=0.642  R=0.792  F1=0.710  (support=626, predicted=772)

Model parameters: 36,684
Channel sizes: [32, 32, 48, 48, 64, 64]
Epochs trained: 11 (best) / 60 (max)
Feature scaling: robust
Class weighting: none
Oversampling: none
Seed: 42
