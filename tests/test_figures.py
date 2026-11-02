from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engagement_intent.evaluation.figures import attach_lead_time, collect_validation_runs


def test_collect_validation_runs_reads_neural_and_baseline_runs(tmp_path: Path) -> None:
    lstm = tmp_path / "lstm_20260101T000000Z_aaaa"
    lstm.mkdir()
    (lstm / "run_metadata.json").write_text(
        json.dumps(
            {
                "processed_dir": "data\\processed\\windows\\pose_2s",
                "seed": 7,
                "parameter_count": 10,
                "validation_metrics": {"f1": 0.5, "precision": 0.4, "recall": 0.6},
            }
        )
    )
    baseline = tmp_path / "baseline_20260101T000000Z_bbbb"
    baseline.mkdir()
    (baseline / "run_metadata.json").write_text(
        json.dumps({"processed_dir": "data/processed/windows/motion_2s", "seed": 7})
    )
    (baseline / "metrics.json").write_text(
        json.dumps({"logistic_regression": {"f1": 0.3, "precision": 0.2, "recall": 0.4}})
    )
    (tmp_path / "lstm_incomplete").mkdir()

    runs = collect_validation_runs(tmp_path).set_index("model")

    assert set(runs.index) == {"lstm", "logistic"}
    assert runs.loc["lstm", "windows"] == "pose_2s"
    assert runs.loc["logistic", "windows"] == "motion_2s"
    assert runs.loc["logistic", "f1"] == pytest.approx(0.3)


def test_attach_lead_time_matches_window_end_and_rejects_missing(tmp_path: Path) -> None:
    pd.DataFrame(
        {"pid": [1, 1], "resampled_timestamp": [10.0, 10.2], "feng_time": [3.0, 2.8]}
    ).to_csv(tmp_path / "scene_a.csv", index=False)
    windows = pd.DataFrame(
        {"scenario_id": ["scene_a"], "person_id": [1], "window_end_timestamp": [10.2]}
    )

    merged = attach_lead_time(windows, tmp_path)
    assert merged["feng_time"].tolist() == [pytest.approx(2.8)]

    windows.loc[0, "window_end_timestamp"] = 11.0
    with pytest.raises(ValueError, match="could not be matched"):
        attach_lead_time(windows, tmp_path)
