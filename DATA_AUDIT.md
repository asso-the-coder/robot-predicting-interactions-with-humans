# PAR-D Shutter Data Audit

Audit date: 2026-08-06

## Source and integrity

- Source: [Shutter Interaction Dataset, Yale Dataverse](https://doi.org/10.60600/YU/KFFQPF), version 1.1.
- License: CC0 1.0; citation is still required by the dataset authors.
- Raw archive: `doi-10.60600-yu-kffqpf.zip`, 295,680,400 bytes.
- SHA-256: `a3d02afb9c8277971807c02de3c8290e09b7405c6e980418334704ce672358b8`.
- The raw archive remains unchanged and ignored under `data/raw/par-d/`.

## Observed release structure

The release has two 5 Hz datasets collected in different physical lobbies:

| Dataset | Scenario CSVs |
| --- | ---: |
| Lobby 1 | 348 |
| Lobby 2 | 711 |
| Total | 1,059 |

Across the release there are 139,092 rows, 337 columns, and 2,429 unique `(scenario, person)` tracks. Both lobbies have the same columns. The paper reports that collection occurred on 13 days and that 393 of the 2,429 observed people interacted with the robot.

Each CSV represents one recording during which at least one person was visible. A scenario can contain multiple people, and a person ID alone is not globally unique: 21 numeric IDs occur in both lobbies. The stable person-track key is therefore `(lobby, scenario_id, pid)`.

## Timing, target, and leakage controls

- `resampled_timestamp` is normally spaced by 0.2 seconds, confirming 5 Hz sampling.
- `engaged_smoothed` marks present interaction/engagement.
- `feng_time` is the time until engagement; `-1` means no future engagement is annotated.
- `future_interaction_4_sec` exactly equals `0 <= feng_time <= 4` across all 139,092 rows.
- The model target is `future_interaction_4_sec` at the final frame of an observation window.
- Any input window containing `engaged_smoothed = true` is rejected so the model cannot observe an interaction that has already begun.
- Windows cannot bridge timestamp gaps larger than 0.31 seconds.

At a two-second observation length (10 frames), the raw window candidates that survive these rules contain 2,516 positive and 47,641 negative examples before train/validation/test assignment. Severe imbalance will be handled using training-split class weighting, not by modifying validation or test distributions.

The following released columns are explicitly prohibited as model inputs because they reveal labels, interaction state, or robot response: button values, FSM state, robot actions or speech, current/past/future interaction columns, `feng_time`, `looking_at_robot`, `slow_walking_speed`, and `within_min_distance`.

## Feature sets

The data confirms the modalities questioned in the proposal feedback are actually available:

- **Motion/proxemics:** distance from cart to pelvis, pedestrian velocity, planar velocity, and relative bearing.
- **Orientation:** motion features plus relative head direction and head/pelvis orientation encodings.
- **Pose:** orientation features plus selected head, shoulder, wrist, and ankle coordinates centered on the pedestrian pelvis.

Centered pose features are used instead of raw global joint positions because the two lobbies have different coordinate frames.

## Missingness

The main numeric motion/orientation/pose columns are present throughout the release except for 2,429 missing values in each velocity column, corresponding primarily to the first observation of a track where velocity cannot yet be estimated. Imputation means and scaling statistics are fitted using training windows only and persisted with processed artifacts.

Robot behavior fields have substantial missingness but are excluded from model inputs in any case.

## Split policy

The split is assigned by entire recording date before constructing overlapping windows. This is stronger than splitting frames or individual scenarios and prevents windows from the same collection day entering multiple splits.

- Validation: 2022-08-11 and 2022-08-18.
- Final test: 2022-08-08 and 2022-08-23.
- Training: the remaining nine recording days.

Both validation and test contain recordings from both lobbies. Test dates are excluded from imputation, scaling, checkpoint selection, hyperparameter selection, and threshold selection.

## Known limitations

- Interaction intent is not directly observable; it is approximated using future interaction behavior.
- Labels are derived partly from proximity, speed, orientation, and button use, so raw label-construction flags must not be used as inputs.
- Overlapping windows are correlated even though recording-day separation prevents cross-split leakage.
- Skeleton observations originate from two Kinect sensors and can contain real-world tracking noise and occlusion.
- The held-out test uses unseen recording days from the same two locations, not independently collected data from a third environment.
