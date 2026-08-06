from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engagement_intent.data.preprocess import (
    ENGAGED_COLUMN,
    MOTION_FEATURES,
    PERSON_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    apply_normalizer,
    assign_split,
    contiguous_segments,
    fit_normalizer,
    make_windows,
)


def synthetic_track() -> pd.DataFrame:
    rows = 6
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: [0.0, 0.2, 0.4, 1.0, 1.2, 1.4],
            PERSON_COLUMN: [7] * rows,
            ENGAGED_COLUMN: [False, False, False, False, False, True],
            TARGET_COLUMN: [False, False, True, False, True, True],
        }
    )
    for index, feature in enumerate(MOTION_FEATURES):
        frame[feature] = np.arange(rows, dtype=float) + index
    return frame


def test_assign_split_rejects_overlapping_recording_dates() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assign_split("2022-08-08", {"2022-08-08"}, {"2022-08-08"})


def test_contiguous_segments_do_not_bridge_tracking_gaps() -> None:
    segments = list(contiguous_segments(synthetic_track(), max_gap_seconds=0.31))
    assert [len(segment) for segment in segments] == [3, 3]


def test_make_windows_uses_end_label_and_rejects_engaged_inputs() -> None:
    segment = synthetic_track().iloc[:3].copy()
    windows, targets, metadata = make_windows(segment, "motion", window_frames=2, stride_frames=1)

    assert np.stack(windows).shape == (2, 2, len(MOTION_FEATURES))
    assert targets == [0, 1]
    assert metadata[-1]["window_end_timestamp"] == pytest.approx(0.4)

    segment.loc[segment.index[-1], ENGAGED_COLUMN] = True
    _, engaged_targets, _ = make_windows(segment, "motion", window_frames=2, stride_frames=1)
    assert engaged_targets == [0]


def test_normalizer_uses_training_values_and_imputes_missing() -> None:
    train = np.array([[[1.0, np.nan], [3.0, 4.0]]], dtype=np.float32)
    means, stds = fit_normalizer(train)
    transformed = apply_normalizer(np.array([[[5.0, np.nan]]], dtype=np.float32), means, stds)

    np.testing.assert_allclose(means, [2.0, 4.0])
    np.testing.assert_allclose(transformed, [[[3.0, 0.0]]])
    assert np.isfinite(transformed).all()

