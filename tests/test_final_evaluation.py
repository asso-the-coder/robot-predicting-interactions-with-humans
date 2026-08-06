from __future__ import annotations

import numpy as np

from engagement_intent.evaluation.final import select_qualitative_indices


def test_select_qualitative_indices_returns_each_confusion_category() -> None:
    targets = np.array([1, 0, 0, 1, 1, 0])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.4])

    selected = select_qualitative_indices(targets, probabilities, threshold=0.5)

    assert selected == {
        "true_positive": 0,
        "true_negative": 1,
        "false_positive": 2,
        "false_negative": 3,
    }

