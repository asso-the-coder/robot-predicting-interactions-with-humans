from __future__ import annotations

import numpy as np
import pytest

from engagement_intent.evaluation.metrics import binary_metrics, select_f1_threshold


def test_binary_metrics_uses_documented_confusion_order() -> None:
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.2]), 0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_threshold_selection_uses_validation_f1() -> None:
    threshold, metrics = select_f1_threshold(
        np.array([0, 1, 1]), np.array([0.2, 0.4, 0.9]), np.array([0.3, 0.5])
    )
    assert threshold == pytest.approx(0.3)
    assert metrics["f1"] == pytest.approx(1.0)

