# CHB-MIT Patient chb15 — Dataset Analysis

## 1. Patient Overview

| Property | Value |
|---|---|
| Patient ID | chb15 |
| Age | ~16 years old |
| Sampling Rate | 256 Hz |
| Recording Format | EDF (European Data Format) |
| Total EDF Files | 40 |
| Seizure-Containing Files | 14 |
| Non-Seizure Files | 26 |
| Total Seizures | 20 |
| Total Ictal Duration | 1992 seconds (~33.2 minutes) |
| Approx. Total Recording | ~40 hours |

## 2. Channel Configuration

The pipeline selects 5 bipolar channels from the available montage:

| Channel | Location | Rationale |
|---|---|---|
| FP1-F7 | Left frontopolar → left frontal | Left frontal seizure activity |
| F7-T7 | Left frontal → left temporal | Left frontotemporal propagation |
| FP2-F8 | Right frontopolar → right frontal | Right frontal seizure activity |
| F8-T8 | Right frontal → right temporal | Right frontotemporal propagation |
| FZ-CZ | Midline frontal → midline central | Midline / generalized activity |

**Note:** chb15_01 uses an initial montage with 23 channels. After chb15_01, the montage changes to 38 channels. All 5 selected channels are present in both montages by name, so `load_edf()` (which selects by name) handles this correctly.

## 3. Complete EDF Inventory

| # | File | Seizures | Onset(s) | Offset(s) | Duration(s) | Start Time | End Time |
|---|---|---|---|---|---|---|---|
| 1 | chb15_01 | 0 | — | — | — | 18:23:13 | 19:23:15 |
| 2 | chb15_02 | 0 | — | — | — | 19:23:43 | 20:23:43 |
| 3 | chb15_03 | 0 | — | — | — | 20:23:50 | 21:23:50 |
| 4 | chb15_04 | 0 | — | — | — | 21:23:57 | 22:23:57 |
| 5 | chb15_05 | 0 | — | — | — | 22:24:03 | 23:24:03 |
| 6 | **chb15_06** | **1** | 272 | 397 | **125** | 23:24:10 | 24:24:10 |
| 7 | chb15_07 | 0 | — | — | — | 00:24:17 | 01:24:17 |
| 8 | chb15_08 | 0 | — | — | — | 01:24:24 | 02:24:24 |
| 9 | chb15_09 | 0 | — | — | — | 02:24:31 | 03:24:31 |
| 10 | **chb15_10** | **1** | 1082 | 1113 | **31** | 03:24:38 | 04:24:38 |
| 11 | chb15_11 | 0 | — | — | — | 04:24:45 | 05:24:45 |
| 12 | chb15_12 | 0 | — | — | — | 05:24:52 | 06:24:52 |
| 13 | chb15_13 | 0 | — | — | — | 06:24:59 | 07:24:59 |
| 14 | chb15_14 | 0 | — | — | — | 07:25:06 | 08:25:06 |
| 15 | **chb15_15** | **1** | 1591 | 1748 | **157** | 08:25:13 | 09:25:13 |
| 16 | chb15_16 | 0 | — | — | — | 09:25:20 | 10:25:20 |
| 17 | **chb15_17** | **1** | 1925 | 1960 | **35** | 10:25:27 | 11:25:27 |
| 18 | chb15_19 | 0 | — | — | — | 12:25:41 | 13:25:41 |
| 19 | **chb15_20** | **1** | 607 | 662 | **55** | 13:25:48 | 14:25:48 |
| 20 | **chb15_22** | **1** | 760 | 965 | **205** | 15:26:02 | 16:26:02 |
| 21 | chb15_26 | 0 | — | — | — | 19:27:44 | 20:27:44 |
| 22 | **chb15_28** | **1** | 876 | 1066 | **190** | 21:27:58 | 22:27:58 |
| 23 | chb15_29 | 0 | — | — | — | 22:28:05 | 23:28:05 |
| 24 | chb15_30 | 0 | — | — | — | 23:28:12 | 24:28:12 |
| 25 | **chb15_31** | **1** | 1751 | 1871 | **120** | 00:28:18 | 01:28:18 |
| 26 | chb15_32 | 0 | — | — | — | 01:28:25 | 02:28:25 |
| 27 | chb15_33 | 0 | — | — | — | 02:28:32 | 03:28:32 |
| 28 | chb15_35 | 0 | — | — | — | 04:28:46 | 05:28:46 |
| 29 | chb15_37 | 0 | — | — | — | 06:29:00 | 07:29:00 |
| 30 | **chb15_40** | **3** | 834/2378/3362 | 894/2497/3425 | **60+119+63=242** | 09:29:21 | 10:29:21 |
| 31 | chb15_45 | 0 | — | — | — | 14:30:00 | 15:30:00 |
| 32 | **chb15_46** | **1** | 3322 | 3429 | **107** | 15:30:07 | 16:30:07 |
| 33 | **chb15_49** | **1** | 1108 | 1248 | **140** | 18:38:05 | 19:38:30 |
| 34 | chb15_50 | 0 | — | — | — | 19:38:56 | 20:38:56 |
| 35 | chb15_51 | 0 | — | — | — | 20:39:03 | 21:39:03 |
| 36 | **chb15_52** | **1** | 778 | 849 | **71** | 21:39:10 | 22:39:19 |
| 37 | **chb15_54** | **5** | 263/843/1524/2179/3428 | 318/1020/1595/2250/3460 | **55+177+71+71+32=406** | 23:39:54 | 24:39:54 |
| 38 | chb15_61 | 0 | — | — | — | 06:41:03 | 07:41:03 |
| 39 | **chb15_62** | **1** | 751 | 859 | **108** | 07:41:10 | 08:41:10 |
| 40 | chb15_63 | 0 | — | — | — | 08:41:17 | 09:41:17 |

## 4. Labeling Configuration

| Class | Label | Definition | Duration Parameter |
|---|---|---|---|
| Interictal | 0 | All EEG not covered by other classes | — |
| Preictal | 1 | 30 minutes before seizure onset | `PREICTAL_SEC = 1800` |
| Ictal | 2 | Seizure onset to offset (inclusive) | From annotations |
| Postictal | 3 | 30 minutes after seizure offset | `POSTICTAL_SEC = 1800` |

**Label assignment rule:** Each 1-second window is labeled by its center timestamp.

**Priority rule:** Ictal always wins. When postictal of seizure N overlaps preictal of seizure N+1, the later seizure's preictal label wins (processed in chronological order).

## 5. Preictal Truncation Analysis

With a 30-minute (1800s) preictal window, seizures occurring less than 1800 seconds into their EDF have truncated preictal periods because EEG data before the EDF start does not exist.

| Seizure | EDF | Onset (s) | Full 30 min? | Actual Preictal Available | % Captured |
|---|---|---|---|---|---|
| 1 | chb15_06 | 272 | ❌ | 272s (4.5 min) | 15% |
| 2 | chb15_10 | 1082 | ❌ | 1082s (18.0 min) | 60% |
| 3 | chb15_15 | 1591 | ❌ | 1591s (26.5 min) | 88% |
| 4 | chb15_17 | 1925 | ✅ | 1800s (30.0 min) | 100% |
| 5 | chb15_20 | 607 | ❌ | 607s (10.1 min) | 34% |
| 6 | chb15_22 | 760 | ❌ | 760s (12.7 min) | 42% |
| 7 | chb15_28 | 876 | ❌ | 876s (14.6 min) | 49% |
| 8 | chb15_31 | 1751 | ❌ | 1751s (29.2 min) | 97% |
| 9 | chb15_40.S1 | 834 | ❌ | 834s (13.9 min) | 46% |
| 10 | chb15_40.S2 | 2378 | ⚠️ overlap | See Section 6 | — |
| 11 | chb15_40.S3 | 3362 | ⚠️ overlap | See Section 6 | — |
| 12 | chb15_46 | 3322 | ✅ | 1800s (30.0 min) | 100% |
| 13 | chb15_49 | 1108 | ❌ | 1108s (18.5 min) | 61% |
| 14 | chb15_52 | 778 | ❌ | 778s (13.0 min) | 43% |
| 15 | chb15_54.S1 | 263 | ❌ | 263s (4.4 min) | 15% |
| 16-19 | chb15_54.S2-S5 | — | ⚠️ overlap | See Section 6 | — |
| 20 | chb15_62 | 751 | ❌ | 751s (12.5 min) | 42% |

**Summary:** Only 2 out of 20 seizures (chb15_17, chb15_46) have a full 30-minute preictal period captured within their EDF. 11 seizures have less than 50% of the intended preictal period.

## 6. Within-EDF Postictal/Preictal Overlap Analysis

When two seizures occur within the same EDF and are less than 60 minutes apart, the postictal period of the first seizure overlaps with the preictal period of the second. The labeling code resolves this by giving priority to the later seizure's preictal label.

### chb15_40.edf (3 seizures, 60 min apart = all overlap)

```
Time (s):  0        834===894              2378=======2497        3362===3425   3600
                     S1                      S2                     S3

S1 postictal: 894 → 2694     S2 preictal: 578 → 2378
                 OVERLAP: 894-2378 (1484s, ~24.7 min) → labeled PREICTAL

S2 postictal: 2497 → 4297   S3 preictal: 1562 → 3362
                 OVERLAP: 2497-3362 (865s, ~14.4 min) → labeled PREICTAL

Result: S1 loses ALL its postictal data (overwritten by S2 preictal/ictal/postictal)
        S2 loses most of its postictal data (overwritten by S3 preictal)
        S3 postictal truncated at EDF end (only ~175s of 1800s)
```

### chb15_54.edf (5 seizures, very dense — worst case)

```
Time (s): 0  263=318  843===1020  1524=1595  2179=2250      3428=3460  3600
             S1         S2          S3          S4              S5

S1→S2: gap = 525s  (8.75 min) → OVERLAP: 318-843   (525s)  → Preictal wins
S2→S3: gap = 504s  (8.4 min)  → OVERLAP: 1020-1524 (504s)  → Preictal wins
S3→S4: gap = 584s  (9.7 min)  → OVERLAP: 1595-2179 (584s)  → Preictal wins
S4→S5: gap = 1178s (19.6 min) → OVERLAP: 2250-3428 (1178s) → Preictal wins

Result: Seizures S1-S4 lose ALL postictal data to the next seizure's preictal.
        S5 postictal truncated at EDF end (only ~140s of 1800s).
        This file contributes almost exclusively Ictal and Preictal labels,
        with very little clean Postictal or Interictal data.
```

### Complete Overlap Table

| Location | Gap Between Seizures | Overlap Duration | Effect |
|---|---|---|---|
| chb15_40: S1→S2 | 1484s (24.7 min) | 1484s | S1 postictal → S2 preictal |
| chb15_40: S2→S3 | 865s (14.4 min) | 865s | S2 postictal → S3 preictal |
| chb15_54: S1→S2 | 525s (8.75 min) | 525s | S1 postictal → S2 preictal |
| chb15_54: S2→S3 | 504s (8.4 min) | 504s | S2 postictal → S3 preictal |
| chb15_54: S3→S4 | 584s (9.7 min) | 584s | S3 postictal → S4 preictal |
| chb15_54: S4→S5 | 1178s (19.6 min) | 1178s | S4 postictal → S5 preictal |

**Total overlap: 5,140 seconds (~85.7 minutes) of postictal data relabeled as preictal.**

## 7. Postictal Truncation at EDF End

| Seizure | EDF | Offset (s) | Postictal End (s) | EDF End (~s) | Actual Postictal |
|---|---|---|---|---|---|
| chb15_17 | chb15_17 | 1960 | 3760 | ~3600 | ~1640s (27.3 min) |
| chb15_31 | chb15_31 | 1871 | 3671 | ~3600 | ~1729s (28.8 min) |
| chb15_40.S3 | chb15_40 | 3425 | 5225 | ~3600 | **~175s (2.9 min)** |
| chb15_46 | chb15_46 | 3429 | 5229 | ~3600 | **~171s (2.9 min)** |
| chb15_54.S5 | chb15_54 | 3460 | 5260 | ~3600 | **~140s (2.3 min)** |

## 8. Cross-EDF Boundary Mislabeling

Each EDF is processed independently. The labeling code does not know about seizures in adjacent EDFs. This causes mislabeling at file boundaries.

### Preictal bleeding into previous EDF

If a seizure occurs early in an EDF, its preictal period should extend into the previous EDF, but does not.

| Seizure EDF | Previous EDF | Missing Preictal in Previous EDF |
|---|---|---|
| chb15_06 (onset 272s) | chb15_05 | Last 1528s mislabeled Interictal |
| chb15_10 (onset 1082s) | chb15_09 | Last 718s mislabeled Interictal |
| chb15_20 (onset 607s) | chb15_19 | Last 1193s mislabeled Interictal |
| chb15_52 (onset 778s) | chb15_51 | Last 1022s mislabeled Interictal |
| chb15_62 (onset 751s) | chb15_61 | Last 1049s mislabeled Interictal |

### Postictal bleeding into next EDF

If a seizure's postictal extends beyond the EDF end, early samples in the next EDF are mislabeled.

| Seizure EDF | Next EDF | Missing Postictal in Next EDF |
|---|---|---|
| chb15_17 (postictal end 3760) | chb15_19 | First ~160s mislabeled Interictal |
| chb15_31 (postictal end 3671) | chb15_32 | First ~71s mislabeled Interictal |
| chb15_40 S3 (postictal end 5225) | chb15_45 | First ~1625s mislabeled Interictal |
| chb15_46 (postictal end 5229) | chb15_49 | First ~1629s mislabeled Interictal |
| chb15_54 S5 (postictal end 5260) | chb15_61 | First ~1660s mislabeled Interictal |

**Known limitation:** This is inherent to per-EDF independent processing. A cross-file label propagation step could fix this but is not implemented for the baseline.

## 9. Proposed Train / Val / Test Split (70 / 15 / 15)

Chronological split with seizure guarantee in each partition:

### TRAIN (28 EDFs)

| # | File | Seizures | Ictal Duration |
|---|---|---|---|
| 1-5 | chb15_01 – chb15_05 | 0 | 0s |
| 6 | **chb15_06** | 1 | 125s |
| 7-9 | chb15_07 – chb15_09 | 0 | 0s |
| 10 | **chb15_10** | 1 | 31s |
| 11-14 | chb15_11 – chb15_14 | 0 | 0s |
| 15 | **chb15_15** | 1 | 157s |
| 16 | chb15_16 | 0 | 0s |
| 17 | **chb15_17** | 1 | 35s |
| 18 | chb15_19 | 0 | 0s |
| 19 | **chb15_20** | 1 | 55s |
| 20 | **chb15_22** | 1 | 205s |
| 21 | chb15_26 | 0 | 0s |
| 22 | **chb15_28** | 1 | 190s |
| 23-24 | chb15_29 – chb15_30 | 0 | 0s |
| 25 | **chb15_31** | 1 | 120s |
| 26-28 | chb15_32, chb15_33, chb15_35 | 0 | 0s |

**Train total: 8 seizures, 918s (15.3 min) ictal, 20 non-seizure EDFs**

### VAL (6 EDFs)

| # | File | Seizures | Ictal Duration |
|---|---|---|---|
| 29 | chb15_37 | 0 | 0s |
| 30 | **chb15_40** | 3 | 242s |
| 31 | chb15_45 | 0 | 0s |
| 32 | **chb15_46** | 1 | 107s |
| 33 | **chb15_49** | 1 | 140s |
| 34 | chb15_50 | 0 | 0s |

**Val total: 5 seizures, 489s (8.2 min) ictal, 3 non-seizure EDFs**

### TEST (6 EDFs)

| # | File | Seizures | Ictal Duration |
|---|---|---|---|
| 35 | chb15_51 | 0 | 0s |
| 36 | **chb15_52** | 1 | 71s |
| 37 | **chb15_54** | 5 | 406s |
| 38 | chb15_61 | 0 | 0s |
| 39 | **chb15_62** | 1 | 108s |
| 40 | chb15_63 | 0 | 0s |

**Test total: 7 seizures, 585s (9.8 min) ictal, 3 non-seizure EDFs**

### Split Summary

| Partition | EDFs | Seizure EDFs | Seizures | Ictal (s) | Ictal (min) |
|---|---|---|---|---|---|
| Train | 28 (70%) | 8 | 8 | 918 | 15.3 |
| Val | 6 (15%) | 3 | 5 | 489 | 8.2 |
| Test | 6 (15%) | 3 | 7 | 585 | 9.8 |
| **Total** | **40** | **14** | **20** | **1992** | **33.2** |

## 10. Comparison with chb06 (Reference Patient)

| Property | chb06 | chb15 |
|---|---|---|
| Age | ~1.5 years | ~16 years |
| Total EDFs | 18 | 40 |
| Seizure-containing EDFs | 7 | 14 |
| Total seizures | 10 | 20 |
| Total ictal duration | 153s (2.6 min) | 1992s (33.2 min) |
| Avg seizure duration | 15.3s | 99.6s |
| Multi-seizure EDFs | 1 (chb06_01: 3), 1 (chb06_04: 2) | 2 (chb15_40: 3, chb15_54: 5) |

**chb15 is a significantly better experiment target** — 13× more ictal data, longer seizures, and more total recordings.
