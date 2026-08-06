# Experiment Results

Experiment date: 2026-08-06

Positive label: a pedestrian begins interacting with the robot within four seconds after the observation window. All thresholds and checkpoints were selected using validation F1. The final test dates were evaluated only after model, feature-set, observation-length, checkpoint, and threshold selection were frozen.

## Validation model and feature comparison

All entries below use a two-second observation window and the same recording-day split.

| Model | Feature set | Parameters | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Majority class | — | 0 | 0.000 | 0.000 | 0.000 |
| Logistic regression | Motion | 25 coefficients | 0.202 | 0.302 | 0.242 |
| Logistic regression | Orientation | 49 coefficients | 0.232 | 0.366 | 0.284 |
| Logistic regression | Centered pose | 133 coefficients | 0.156 | 0.419 | 0.227 |
| LSTM | Motion | 51,777 | 0.333 | 0.256 | 0.289 |
| LSTM | Orientation | 53,313 | 0.214 | 0.445 | 0.289 |
| **LSTM** | **Centered pose** | **58,689** | **0.311** | **0.387** | **0.345** |

The richer pose representation improved validation F1 only for the nonlinear sequence model. It hurt logistic regression, suggesting that the benefit was not simply caused by adding more coordinates.

## Validation observation-length comparison

The selected centered-pose LSTM was compared at the planned early-observation lengths.

| Observation | Frames at 5 Hz | Validation samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 second | 5 | 8,808 | 0.212 | 0.388 | 0.274 |
| **2 seconds** | **10** | **7,618** | **0.311** | **0.387** | **0.345** |
| 3 seconds | 15 | 6,619 | 0.239 | 0.350 | 0.284 |

Two seconds was selected. Longer observation did not monotonically improve performance, partly because longer windows reduce the number of eligible pre-engagement examples.

## Final held-out recording-day evaluation

The test set consists entirely of 2022-08-08 (Lobby 1) and 2022-08-23 (Lobby 2): 5,587 windows, including 479 positives. These dates did not contribute to imputation, scaling, training, checkpoint selection, feature selection, observation-length selection, or thresholds.

| Selected model | Threshold | Accuracy | Precision | Recall | F1 | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Orientation logistic regression | 0.80 | 0.861 | 0.239 | 0.284 | **0.260** | 0.694 |
| Centered-pose LSTM | 0.83 | 0.862 | 0.115 | 0.092 | 0.102 | 0.597 |

Confusion matrices use `[[TN, FP], [FN, TP]]` ordering:

- Logistic regression: `[[4675, 433], [343, 136]]`.
- LSTM: `[[4771, 337], [435, 44]]`.

The pose LSTM did not generalize its validation advantage to unseen recording days. Its training loss continued falling while validation loss rose sharply, and its validation-selected threshold yielded only 9.2% test recall. This is evidence of overfitting and temporal/location shift, not a reason to alter the held-out result. The simpler orientation baseline was more robust.

Subgroup F1 further illustrates the shift:

| Model | Lobby 1 / Aug. 8 | Lobby 2 / Aug. 23 |
| --- | ---: | ---: |
| Logistic regression | 0.155 | 0.321 |
| Pose LSTM | 0.099 | 0.105 |

## Qualitative examples

Representative cases were selected deterministically from the LSTM test predictions:

- True positive: distance to the cart fell from approximately 0.87 m to 0.54 m while speed fell from about 1.04 m/s to nearly zero.
- False positive: distance similarly fell from about 0.70 m to 0.47 m and speed approached zero, but no interaction began within the four-second horizon. This behavior is visually similar to a genuine approach and demonstrates label/behavior ambiguity.
- False negative: the person remained approximately 0.96 m away and almost stationary during the observation, providing little approach motion even though interaction began within four seconds.
- True negative: the person remained close and nearly stationary, showing that proximity and low speed alone are insufficient.

The complete ignored artifact directory is `outputs/final_evaluation_20260806T150542Z_9933a63f/`. It contains test predictions, selected example sequences, confusion matrices, learning curves, and a metric-comparison figure.

## Reproducibility identifiers

- Best validation logistic run: `baseline_20260806T144151Z_8449420c`.
- Selected LSTM run: `lstm_20260806T145923Z_e770e3a7`.
- Frozen final evaluation: `final_evaluation_20260806T150542Z_9933a63f`.
- Final evaluator code revision: `f6d0816`.

Generated run directories remain ignored because they contain large checkpoints and prediction files. The small reviewed summary in this document is tracked intentionally.

