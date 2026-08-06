"""One-shot final test evaluation for validation-selected models."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from engagement_intent.evaluation.metrics import binary_metrics
from engagement_intent.models import LSTMClassifier
from engagement_intent.training.baseline import summarize_sequences


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_processed_split(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    archive = np.load(processed_dir / f"{split}.npz")
    metadata = pd.read_csv(processed_dir / f"{split}_metadata.csv")
    features = archive["features"].astype(np.float32)
    targets = archive["targets"].astype(np.uint8)
    if len(features) != len(targets) or len(targets) != len(metadata):
        raise ValueError(f"Feature, target, and metadata counts disagree for {split}")
    return features, targets, metadata


def select_qualitative_indices(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, int]:
    """Select confident representative TP/TN/FP/FN examples deterministically."""

    targets = np.asarray(targets, dtype=np.uint8)
    predictions = (np.asarray(probabilities) >= threshold).astype(np.uint8)
    masks = {
        "true_positive": (targets == 1) & (predictions == 1),
        "true_negative": (targets == 0) & (predictions == 0),
        "false_positive": (targets == 0) & (predictions == 1),
        "false_negative": (targets == 1) & (predictions == 0),
    }
    selectors = {
        "true_positive": np.argmax,
        "true_negative": np.argmin,
        "false_positive": np.argmax,
        "false_negative": np.argmin,
    }
    result: dict[str, int] = {}
    for category, mask in masks.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        local = int(selectors[category](probabilities[indices]))
        result[category] = int(indices[local])
    return result


def plot_confusion_matrices(metrics: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for axis, (name, label) in zip(
        axes, (("logistic_regression", "Logistic regression"), ("lstm", "Pose LSTM"))
    ):
        matrix = np.asarray(metrics[name]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_title(label)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_learning_curves(lstm_run: Path, output_path: Path) -> None:
    curves = pd.read_csv(lstm_run / "learning_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(curves.epoch, curves.train_weighted_bce_loss, label="Train")
    axes[0].plot(curves.epoch, curves.validation_weighted_bce_loss, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted BCE loss")
    axes[0].legend()
    axes[1].plot(curves.epoch, curves.validation_f1, color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation F1")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(metrics: dict[str, Any], output_path: Path) -> None:
    names = ["precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    axis.bar(x - width / 2, [metrics["logistic_regression"][key] for key in names], width, label="Logistic")
    axis.bar(x + width / 2, [metrics["lstm"][key] for key in names], width, label="Pose LSTM")
    axis.set_xticks(x, [name.capitalize() for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_final(baseline_run: Path, lstm_run: Path, output_root: Path) -> Path:
    baseline_metadata = load_json(baseline_run / "run_metadata.json")
    lstm_metadata = load_json(lstm_run / "run_metadata.json")
    baseline_processed = Path(baseline_metadata["processed_dir"])
    lstm_processed = Path(lstm_metadata["processed_dir"])

    baseline_features, baseline_targets, baseline_rows = load_processed_split(baseline_processed, "test")
    lstm_features, lstm_targets, lstm_rows = load_processed_split(lstm_processed, "test")
    if not np.array_equal(baseline_targets, lstm_targets):
        raise ValueError("Selected model test labels differ")
    if baseline_rows["sample_id"].tolist() != lstm_rows["sample_id"].tolist():
        raise ValueError("Selected model test sample identities differ")

    logistic = joblib.load(baseline_run / "logistic_regression.joblib")
    logistic_probabilities = logistic.predict_proba(summarize_sequences(baseline_features))[:, 1]
    logistic_threshold = float(baseline_metadata["selected_threshold"])

    checkpoint = torch.load(lstm_run / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    lstm = LSTMClassifier(
        input_size=int(checkpoint["input_size"]), **checkpoint["model_config"]
    )
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm.eval()
    probability_parts = []
    with torch.no_grad():
        for start in range(0, len(lstm_features), 1024):
            logits = lstm(torch.from_numpy(lstm_features[start : start + 1024]))
            probability_parts.append(torch.sigmoid(logits).numpy())
    lstm_probabilities = np.concatenate(probability_parts)
    lstm_threshold = float(checkpoint["selected_threshold"])

    metrics = {
        "evaluation_split": "test",
        "test_read_utc": datetime.now(timezone.utc).isoformat(),
        "logistic_regression": binary_metrics(
            baseline_targets, logistic_probabilities, logistic_threshold
        ),
        "lstm": binary_metrics(lstm_targets, lstm_probabilities, lstm_threshold),
    }
    run_id = f"final_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    for name, rows, targets, probabilities, threshold in (
        ("logistic_regression", baseline_rows, baseline_targets, logistic_probabilities, logistic_threshold),
        ("lstm", lstm_rows, lstm_targets, lstm_probabilities, lstm_threshold),
    ):
        predictions = rows.copy()
        predictions["target"] = targets
        predictions["probability"] = probabilities
        predictions["prediction"] = (probabilities >= threshold).astype(np.uint8)
        predictions["model"] = name
        predictions.to_csv(output_dir / f"test_{name}_predictions.csv", index=False)

    example_indices = select_qualitative_indices(lstm_targets, lstm_probabilities, lstm_threshold)
    qualitative_rows = []
    preprocessing = load_json(lstm_processed / "preprocessing.json")
    means = np.asarray(preprocessing["imputation_means"], dtype=np.float32)
    stds = np.asarray(preprocessing["scaling_standard_deviations"], dtype=np.float32)
    qualitative_sequences: dict[str, Any] = {}
    for category, index in example_indices.items():
        row = lstm_rows.iloc[index].to_dict()
        row.update(
            {
                "category": category,
                "target": int(lstm_targets[index]),
                "probability": float(lstm_probabilities[index]),
                "prediction": int(lstm_probabilities[index] >= lstm_threshold),
            }
        )
        qualitative_rows.append(row)
        sequence = lstm_features[index] * stds + means
        qualitative_sequences[category] = {
            "sample_id": row["sample_id"],
            "target": row["target"],
            "probability": row["probability"],
            "feature_names": preprocessing["feature_names"],
            "sequence": np.round(sequence, 4).tolist(),
        }
    pd.DataFrame(qualitative_rows).to_csv(output_dir / "qualitative_examples.csv", index=False)
    (output_dir / "qualitative_sequences.json").write_text(
        json.dumps(qualitative_sequences, indent=2) + "\n", encoding="utf-8"
    )

    selection = {
        "baseline_run": str(baseline_run),
        "lstm_run": str(lstm_run),
        "baseline_processed_dir": str(baseline_processed),
        "lstm_processed_dir": str(lstm_processed),
        "baseline_threshold": logistic_threshold,
        "lstm_threshold": lstm_threshold,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    plot_confusion_matrices(metrics, output_dir / "confusion_matrices.png")
    plot_learning_curves(lstm_run, output_dir / "learning_curves.png")
    plot_metric_comparison(metrics, output_dir / "test_metric_comparison.png")
    print(json.dumps(metrics, indent=2))
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate selected models on the final test split once.")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--lstm-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = evaluate_final(args.baseline_run, args.lstm_run, args.output_root)
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
