"""Train majority-class and logistic-regression baselines."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression

from engagement_intent.evaluation.metrics import binary_metrics, select_f1_threshold


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def summarize_sequences(features: np.ndarray) -> np.ndarray:
    """Convert [N, T, F] sequences into interpretable summary features."""

    if features.ndim != 3:
        raise ValueError(f"Expected [samples, time, features], got {features.shape}")
    return np.concatenate(
        (
            features[:, -1, :],
            features.mean(axis=1),
            features.std(axis=1),
            features[:, -1, :] - features[:, 0, :],
        ),
        axis=1,
    ).astype(np.float32)


def load_split(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    archive = np.load(processed_dir / f"{split}.npz")
    metadata = pd.read_csv(processed_dir / f"{split}_metadata.csv")
    features = archive["features"]
    targets = archive["targets"]
    if len(features) != len(targets) or len(targets) != len(metadata):
        raise ValueError(f"Feature, target, and metadata counts disagree for {split}")
    return features, targets, metadata


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def create_run_dir(output_root: Path) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"baseline_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def prediction_frame(
    metadata: pd.DataFrame, targets: np.ndarray, probabilities: np.ndarray, threshold: float, model: str
) -> pd.DataFrame:
    result = metadata.copy()
    result["target"] = targets.astype(np.uint8)
    result["probability"] = probabilities
    result["prediction"] = (probabilities >= threshold).astype(np.uint8)
    result["model"] = model
    return result


def train_baselines(config: dict[str, Any]) -> Path:
    processed_dir = Path(config["processed_dir"])
    train_features, train_targets, _ = load_split(processed_dir, "train")
    validation_features, validation_targets, validation_metadata = load_split(
        processed_dir, "validation"
    )
    train_summary = summarize_sequences(train_features)
    validation_summary = summarize_sequences(validation_features)

    run_id, run_dir = create_run_dir(Path(config["output_root"]))
    positive_rate = float(train_targets.mean())
    majority_probabilities = np.full(len(validation_targets), positive_rate, dtype=float)
    majority_metrics = binary_metrics(validation_targets, majority_probabilities, threshold=0.5)

    settings = config["logistic_regression"]
    model = LogisticRegression(
        C=float(settings["regularization_c"]),
        class_weight=settings["class_weight"],
        max_iter=int(settings["max_iter"]),
        random_state=int(config["seed"]),
    )
    model.fit(train_summary, train_targets)
    probabilities = model.predict_proba(validation_summary)[:, 1]
    threshold_config = config["threshold_selection"]
    thresholds = np.arange(
        float(threshold_config["minimum"]),
        float(threshold_config["maximum"]) + float(threshold_config["step"]) / 2,
        float(threshold_config["step"]),
    )
    threshold, logistic_metrics = select_f1_threshold(validation_targets, probabilities, thresholds)

    metrics = {
        "run_id": run_id,
        "selection_split": "validation",
        "majority_class": majority_metrics,
        "logistic_regression": logistic_metrics,
    }
    run_metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "processed_dir": str(processed_dir),
        "seed": int(config["seed"]),
        "summary_features": ["final", "mean", "standard_deviation", "change"],
        "input_sequence_shape": list(train_features.shape[1:]),
        "summary_feature_count": int(train_summary.shape[1]),
        "train_samples": int(len(train_targets)),
        "train_positive_rate": positive_rate,
        "selected_threshold": threshold,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    joblib.dump(model, run_dir / "logistic_regression.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prediction_frame(
        validation_metadata, validation_targets, majority_probabilities, 0.5, "majority_class"
    ).to_csv(run_dir / "validation_majority_predictions.csv", index=False)
    prediction_frame(
        validation_metadata, validation_targets, probabilities, threshold, "logistic_regression"
    ).to_csv(run_dir / "validation_logistic_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train validation-selected baseline models.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = train_baselines(load_config(args.config))
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

