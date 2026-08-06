from __future__ import annotations

import numpy as np
import pytest

from engagement_intent.training.baseline import summarize_sequences


def test_summarize_sequences_has_final_mean_std_and_change() -> None:
    features = np.array([[[1.0, 2.0], [3.0, 6.0]]], dtype=np.float32)
    summary = summarize_sequences(features)

    assert summary.shape == (1, 8)
    np.testing.assert_allclose(summary[0, :2], [3.0, 6.0])
    np.testing.assert_allclose(summary[0, 2:4], [2.0, 4.0])
    np.testing.assert_allclose(summary[0, 4:6], [1.0, 2.0])
    np.testing.assert_allclose(summary[0, 6:8], [2.0, 4.0])


def test_summarize_sequences_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Expected"):
        summarize_sequences(np.zeros((4, 3), dtype=np.float32))

