"""Regenerate the README/report figures from saved run artifacts.

Every figure is drawn from files written by the training and final-evaluation
commands; no metric is recomputed from scratch or typed in by hand. Test-split
plots describe the *already frozen* models and are never used for selection.

Example::

    python -m engagement_intent.evaluation.figures \
        --final-run outputs/final_evaluation_20260806T150542Z_9933a63f \
        --output-dir assets/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

# Validated categorical slots (blue, orange, aqua) plus neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    }
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_meta(run_dir: Path) -> dict:
    return _read_json(run_dir / "run_metadata.json")


def _processed_name(meta_or_config: dict) -> str:
    return Path(str(meta_or_config["processed_dir"]).replace("\\", "/")).name


def collect_validation_runs(outputs: Path) -> pd.DataFrame:
    """Gather validation F1 for every completed run directory."""
    rows = []
    for run_dir in sorted(outputs.iterdir()):
        if run_dir.name.startswith(("lstm_", "transformer_")) and (run_dir / "run_metadata.json").exists():
            meta = _run_meta(run_dir)
            metrics = meta["validation_metrics"]
            rows.append(
                {
                    "run_id": run_dir.name,
                    "model": run_dir.name.split("_")[0],
                    "windows": _processed_name(meta),
                    "seed": meta["seed"],
                    "parameters": meta["parameter_count"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )
        elif run_dir.name.startswith("baseline_") and (run_dir / "metrics.json").exists():
            metrics = _read_json(run_dir / "metrics.json")
            meta = _run_meta(run_dir)
            lr = metrics["logistic_regression"]
            rows.append(
                {
                    "run_id": run_dir.name,
                    "model": "logistic",
                    "windows": _processed_name(meta) if "processed_dir" in meta else "",
                    "seed": meta["seed"],
                    "parameters": np.nan,
                    "f1": lr["f1"],
                    "precision": lr["precision"],
                    "recall": lr["recall"],
                }
            )
    return pd.DataFrame(rows)


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )


def plot_feature_ablation(runs: pd.DataFrame, output: Path) -> None:
    """Validation F1 by feature set for logistic regression and seed-360 LSTM."""
    feature_sets = ["motion", "orientation", "pose"]
    labels = ["Motion\n(6 features)", "Orientation\n(12)", "Centered pose\n(33)"]
    seed = runs[runs["seed"] == 360]

    def lookup(model: str, feature: str) -> float:
        match = seed[(seed["model"] == model) & (seed["windows"] == f"{feature}_2s")]
        if match.empty:
            raise ValueError(f"No seed-360 {model} run found for {feature}_2s")
        return float(match.iloc[0]["f1"])

    logistic = [lookup("logistic", f) for f in feature_sets]
    lstm = [lookup("lstm", f) for f in feature_sets]

    x = np.arange(len(feature_sets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - width / 2 - 0.01, logistic, width, color=BLUE, label="Logistic regression")
    b2 = ax.bar(x + width / 2 + 0.01, lstm, width, color=ORANGE, label="LSTM (seed 360)")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation F1 (positive class)")
    ax.set_ylim(0, max(lstm + logistic) * 1.25)
    ax.set_title("Feature ablation: richer input helps only the sequence model")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(output)
    plt.close(fig)


def plot_seed_stability(runs: pd.DataFrame, output: Path) -> None:
    """Per-seed validation F1 for the pose LSTM and pose Transformer."""
    logistic = runs[(runs["model"] == "logistic") & (runs["windows"] == "orientation_2s")]["f1"].max()
    pose = runs[runs["windows"] == "pose_2s"]
    lstm = pose[(pose["model"] == "lstm") & (pose["parameters"] == 58689)]
    lstm = lstm.sort_values("run_id").drop_duplicates("seed", keep="first")
    transformer = pose[(pose["model"] == "transformer") & (pose["parameters"] == 69505)]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for row, (name, frame, color) in enumerate(
        [("Pose LSTM", lstm, ORANGE), ("Pose Transformer", transformer, AQUA)]
    ):
        scores = frame["f1"].to_numpy()
        ax.scatter(scores, np.full_like(scores, row), s=70, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        order = np.argsort(scores)
        for rank, index in enumerate(order):
            offset = 10 if rank % 2 == 0 else -16
            ax.annotate(f"seed {frame['seed'].iloc[index]}", (scores[index], row), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
        mean = scores.mean()
        ax.plot([mean, mean], [row - 0.18, row + 0.18], color=INK, linewidth=2)
        ax.annotate(f"mean {mean:.3f} ± {scores.std(ddof=1):.3f}", (1.0, row), xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.axvline(logistic, color=BLUE, linestyle="--", linewidth=1.5)
    ax.annotate(f"orientation logistic {logistic:.3f}", (logistic, 1.55), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=BLUE)
    ax.set_yticks([0, 1], ["Pose LSTM", "Pose Transformer"])
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlabel("Validation F1 (positive class)")
    ax.set_title("Seed stability: the best LSTM seed was not representative")
    ax.grid(axis="y", visible=False)
    fig.savefig(output)
    plt.close(fig)


def plot_observation_length(runs: pd.DataFrame, output: Path) -> None:
    """Validation F1 versus observation-window length for the pose LSTM."""
    seconds, scores = [], []
    for length in (1, 2, 3):
        match = runs[(runs["model"] == "lstm") & (runs["seed"] == 360) & (runs["windows"] == f"pose_{length}s")]
        if match.empty:
            raise ValueError(f"No seed-360 LSTM run for pose_{length}s")
        seconds.append(length)
        scores.append(float(match.sort_values("run_id").iloc[0]["f1"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(seconds, scores, color=ORANGE, linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    for s, f in zip(seconds, scores):
        ax.annotate(f"{f:.3f}", (s, f), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(seconds, [f"{s} s\n({5 * s} frames)" for s in seconds])
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(0.2, max(scores) * 1.15)
    ax.set_ylabel("Validation F1")
    ax.set_title("How much history is needed?")
    fig.savefig(output)
    plt.close(fig)


def plot_test_results(final_run: Path, output: Path) -> None:
    """Frozen held-out metrics for both selected models."""
    metrics = _read_json(final_run / "metrics.json")
    names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    lr = [metrics["logistic_regression"][n] for n in names]
    lstm = [metrics["lstm"][n] for n in names]
    majority = metrics["logistic_regression"]["negative"] / metrics["logistic_regression"]["samples"]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - width / 2 - 0.01, lr, width, color=BLUE, label="Orientation logistic regression")
    b2 = ax.bar(x + width / 2 + 0.01, lstm, width, color=ORANGE, label="Centered-pose LSTM")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.text(0.0, -0.2, f"Reference: always predicting 'no' scores {majority:.3f} accuracy and 0.000 F1.", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_title("Held-out recording days (frozen, evaluated once)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.savefig(output)
    plt.close(fig)


def plot_confusion_matrices(final_run: Path, output: Path) -> None:
    metrics = _read_json(final_run / "metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"wspace": 0.35})
    for ax, key, title, color in [
        (axes[0], "logistic_regression", "Orientation logistic", BLUE),
        (axes[1], "lstm", "Centered-pose LSTM", ORANGE),
    ]:
        matrix = np.asarray(metrics[key]["confusion_matrix"], dtype=float)
        row_norm = matrix / matrix.sum(axis=1, keepdims=True)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("one_hue", ["#ffffff", color])
        ax.imshow(row_norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = "white" if row_norm[i, j] > 0.6 else INK
                ax.text(j, i, f"{int(matrix[i, j]):,}\n{row_norm[i, j]:.1%}", ha="center", va="center", fontsize=9, color=ink)
        ax.set_xticks([0, 1], ["Pred. no", "Pred. yes"])
        ax.set_yticks([0, 1], ["True no", "True yes"])
        ax.set_title(title)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Test confusion matrices (row-normalized)", fontweight="bold", fontsize=11, y=1.03)
    fig.savefig(output)
    plt.close(fig)


def plot_precision_recall(final_run: Path, output: Path) -> dict[str, float]:
    """Threshold-free test precision-recall curves for the frozen models."""
    metrics = _read_json(final_run / "metrics.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    summary: dict[str, float] = {}
    prevalence = None
    for key, file, label, color in [
        ("logistic_regression", "test_logistic_regression_predictions.csv", "Orientation logistic", BLUE),
        ("lstm", "test_lstm_predictions.csv", "Centered-pose LSTM", ORANGE),
    ]:
        predictions = pd.read_csv(final_run / file)
        precision, recall, _ = precision_recall_curve(predictions["target"], predictions["probability"])
        precision, recall = precision[:-1], recall[:-1]  # drop sklearn's synthetic (recall 0, precision 1) end point
        ap = average_precision_score(predictions["target"], predictions["probability"])
        summary[key] = float(ap)
        prevalence = predictions["target"].mean()
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{label} (AP {ap:.3f})")
        chosen = metrics[key]
        ax.scatter([chosen["recall"]], [chosen["precision"]], s=60, color=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1.5)
    ax.annotate(f"chance = prevalence {prevalence:.3f}", (0.01, prevalence), xytext=(0, -11), textcoords="offset points", ha="left", fontsize=8, color=MUTED)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.4)
    ax.set_title("Test precision-recall\n(dots mark the validation-chosen thresholds)")
    ax.legend(loc="upper right")
    fig.savefig(output)
    plt.close(fig)
    summary["prevalence"] = float(prevalence)
    return summary


def plot_learning_curves(lstm_run: Path, output: Path) -> None:
    curves = pd.read_csv(lstm_run / "learning_curves.csv")
    best_epoch = _run_meta(lstm_run)["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    axes[0].plot(curves["epoch"], curves["train_weighted_bce_loss"], color=BLUE, linewidth=2, label="Train")
    axes[0].plot(curves["epoch"], curves["validation_weighted_bce_loss"], color=ORANGE, linewidth=2, label="Validation")
    axes[0].set_title("Weighted BCE loss")
    axes[0].legend()
    axes[1].plot(curves["epoch"], curves["validation_f1"], color=ORANGE, linewidth=2)
    axes[1].set_title("Validation F1")
    for ax in axes:
        ax.axvline(best_epoch, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[1].annotate(f"selected epoch {best_epoch}", (best_epoch, axes[1].get_ylim()[0]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    fig.suptitle("Pose LSTM overfits within a few epochs", fontweight="bold", fontsize=11, y=1.03)
    fig.savefig(output)
    plt.close(fig)


def attach_lead_time(predictions: pd.DataFrame, source_root: Path) -> pd.DataFrame:
    """Join each window with the released time-to-engagement at its final frame."""
    paths = {p.stem: p for p in source_root.rglob("*.csv")}
    lookups = []
    for scenario_id in predictions["scenario_id"].unique():
        if scenario_id not in paths:
            raise FileNotFoundError(f"Staged CSV not found for scenario {scenario_id}")
        frame = pd.read_csv(paths[scenario_id], usecols=["pid", "resampled_timestamp", "feng_time"])
        frame["scenario_id"] = scenario_id
        lookups.append(frame)
    lookup = pd.concat(lookups, ignore_index=True)
    lookup["end_key"] = lookup["resampled_timestamp"].round(3)
    merged = predictions.assign(end_key=predictions["window_end_timestamp"].round(3)).merge(
        lookup[["scenario_id", "pid", "end_key", "feng_time"]],
        left_on=["scenario_id", "person_id", "end_key"],
        right_on=["scenario_id", "pid", "end_key"],
        how="left",
        validate="many_to_one",
    )
    if merged["feng_time"].isna().any():
        raise ValueError(f"{int(merged['feng_time'].isna().sum())} windows could not be matched to feng_time")
    return merged


def plot_lead_time(final_run: Path, source_root: Path, output: Path) -> pd.DataFrame:
    """Test recall of positive windows, bucketed by seconds until engagement."""
    bins = [0, 1, 2, 3, 4.0001]
    labels = ["0–1 s", "1–2 s", "2–3 s", "3–4 s"]
    rows = []
    for file, name in [
        ("test_logistic_regression_predictions.csv", "Orientation logistic"),
        ("test_lstm_predictions.csv", "Centered-pose LSTM"),
    ]:
        predictions = pd.read_csv(final_run / file)
        positives = attach_lead_time(predictions[predictions["target"] == 1], source_root)
        positives["bucket"] = pd.cut(positives["feng_time"], bins=bins, labels=labels, right=False)
        grouped = positives.groupby("bucket", observed=False)["prediction"].agg(["mean", "size"])
        for bucket, values in grouped.iterrows():
            rows.append({"model": name, "lead_time": bucket, "recall": values["mean"], "windows": int(values["size"])})
    table = pd.DataFrame(rows)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for offset, (name, color) in zip((-1, 1), [("Orientation logistic", BLUE), ("Centered-pose LSTM", ORANGE)]):
        values = table[table["model"] == name]["recall"].to_numpy()
        bars = ax.bar(x + offset * (width / 2 + 0.01), values, width, color=color, label=name)
        _bar_labels(ax, bars, "{:.2f}")
    counts = table[table["model"] == "Orientation logistic"]["windows"].to_numpy()
    ax.set_xticks(x, [f"{label}\n(n={n})" for label, n in zip(labels, counts)])
    ax.set_xlabel("Time from end of observation until engagement begins")
    ax.set_ylabel("Test recall")
    ax.set_ylim(0, max(table["recall"].max() * 1.3, 0.1))
    ax.set_title("Earlier predictions are harder")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.savefig(output)
    plt.close(fig)
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--final-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("data/processed/par-d-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/figures"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = _read_json(args.final_run / "selection.json")
    lstm_run = Path(str(selection["lstm_run"]).replace("\\", "/"))

    runs = collect_validation_runs(args.outputs)
    plot_feature_ablation(runs, args.output_dir / "feature_ablation.png")
    plot_seed_stability(runs, args.output_dir / "seed_stability.png")
    plot_observation_length(runs, args.output_dir / "observation_length.png")
    plot_test_results(args.final_run, args.output_dir / "test_results.png")
    plot_confusion_matrices(args.final_run, args.output_dir / "confusion_matrices.png")
    ap = plot_precision_recall(args.final_run, args.output_dir / "precision_recall.png")
    plot_learning_curves(lstm_run, args.output_dir / "lstm_learning_curves.png")
    summary: dict = {"average_precision": ap, "final_run": args.final_run.name}
    if args.source_root.exists():
        lead = plot_lead_time(args.final_run, args.source_root, args.output_dir / "lead_time_recall.png")
        summary["lead_time_recall"] = lead.to_dict(orient="records")
    else:
        print(f"Skipping lead-time figure: {args.source_root} not found")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
