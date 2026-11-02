# Model Card: Engagement-Intent Classifiers

## Overview

| | |
|---|---|
| **Task** | Binary classification: will a pedestrian begin interacting with a public robot within 4 s? |
| **Input** | 10 frames (2 s at 5 Hz) of robot-relative motion, orientation, and optionally pelvis-centered pose features |
| **Output** | One raw logit per window; sigmoid gives P(interact) |
| **Recommended model** | Orientation logistic regression (49 coefficients, threshold 0.80) |
| **Comparison models** | Centered-pose LSTM (58,689 params), centered-pose Transformer (69,505 params, validation only) |
| **Training data** | PAR-D Shutter Interaction Dataset, 9 of 13 recording days, two university lobbies |

## Intended use

- Research and prototyping of *proactive greeting* behavior for social robots.
- Low-stakes, reversible actions only: turn toward someone, offer a neutral greeting, wait, or stay quiet.
- A human-robot interaction component that can abstain when uncertain.

## Out-of-scope uses

- Surveillance, tracking, or identifying individuals.
- Access control, security screening, employment, or any consequential decision about a person.
- Deployment in a new environment without collecting local validation data first.

## Performance

Held-out recording days, evaluated once after all selection was frozen (positive class = interacts):

| Model | Precision | Recall | F1 | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Orientation logistic regression | 0.239 | 0.284 | 0.260 | 0.197 |
| Centered-pose LSTM | 0.115 | 0.092 | 0.102 | 0.119 |

Chance-level average precision equals the test prevalence, 0.086. The details are in [RESULTS.md](RESULTS.md).

## Known failure modes

- **Earliness:** recall drops from 51% (engagement < 1 s away) to 14–17% (2–4 s away) for the logistic model.
- **Ambiguous behavior:** a person who approaches and stops can still walk away, and a stationary person can still start interacting. Both are common errors.
- **Day and site shift:** F1 differed sharply between the two test days/lobbies (0.155 vs 0.321 for logistic regression). The LSTM overfit to training days.
- **Sensor noise:** Kinect skeletons suffer from occlusion and jitter, which hurts pose features most.

## Ethical considerations

Public-space sensing raises privacy and consent concerns even with de-identified skeleton data. Errors may fall unevenly on people whose behavior or body differs from the two-lobby training population. A false positive means unwanted attention, and a false negative means an unresponsive robot. The system should store no identifying data, act only reversibly, and abstain when uncertain.
