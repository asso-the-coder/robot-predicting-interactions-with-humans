"""Create leakage-safe fixed-length windows from the PAR-D Shutter CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

TIMESTAMP_COLUMN = "resampled_timestamp"
PERSON_COLUMN = "pid"
ENGAGED_COLUMN = "engaged_smoothed"
TARGET_COLUMN = "future_interaction_4_sec"

MOTION_FEATURES = (
    "cart_to_pelvis_dist",
    "person_velocity",
    "x_vel",
    "y_vel",
    "cart_to_pelvis_cos",
    "cart_to_pelvis_sin",
)
ORIENTATION_FEATURES = (
    *MOTION_FEATURES,
    "head_to_cart_cos",
    "head_to_cart_sin",
    "pelvis_y_cos",
    "pelvis_y_sin",
    "head_y_cos",
    "head_y_sin",
)
POSE_JOINTS = (
    "head",
    "shoulder_left",
    "shoulder_right",
    "wrist_left",
    "wrist_right",
    "ankle_left",
    "ankle_right",
)
FEATURE_SETS = {
    "motion": MOTION_FEATURES,
    "orientation": ORIENTATION_FEATURES,
    "pose": ORIENTATION_FEATURES
    + tuple(f"{joint}_relative_{axis}" for joint in POSE_JOINTS for axis in "xyz"),
}

DATE_PATTERN = re.compile(r"csv_(\d{4}-\d{2}-\d{2})_")


@dataclass(frozen=True)
class SourceFile:
    """Identity and split metadata for one source recording."""

    path: Path
    relative_path: str
    lobby: str
    recording_date: str
    scenario_id: str
    split: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def assign_split(recording_date: str, validation_dates: set[str], test_dates: set[str]) -> str:
    if validation_dates & test_dates:
        raise ValueError("Validation and test recording dates overlap")
    if recording_date in test_dates:
        return "test"
    if recording_date in validation_dates:
        return "validation"
    return "train"


def discover_sources(
    source_root: Path, validation_dates: set[str], test_dates: set[str]
) -> list[SourceFile]:
    source_root = source_root.resolve()
    csv_paths = sorted(source_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {source_root}")

    sources: list[SourceFile] = []
    for path in csv_paths:
        relative = path.relative_to(source_root)
        lobby = next((part for part in relative.parts if part in {"Lobby_1", "Lobby_2"}), None)
        match = DATE_PATTERN.search(path.name)
        if lobby is None or match is None:
            raise ValueError(f"Cannot parse lobby/date from source path: {relative}")
        recording_date = match.group(1)
        sources.append(
            SourceFile(
                path=path,
                relative_path=relative.as_posix(),
                lobby=lobby,
                recording_date=recording_date,
                scenario_id=path.stem,
                split=assign_split(recording_date, validation_dates, test_dates),
            )
        )
    return sources


def raw_columns_for(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    columns = {TIMESTAMP_COLUMN, PERSON_COLUMN, ENGAGED_COLUMN, TARGET_COLUMN}
    columns.update(ORIENTATION_FEATURES if feature_set == "pose" else FEATURE_SETS[feature_set])
    if feature_set == "pose":
        columns.update(f"pelvis_{axis}" for axis in "xyz")
        columns.update(f"{joint}_{axis}" for joint in POSE_JOINTS for axis in "xyz")
    return sorted(columns)


def build_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Build one documented feature tensor frame without label-derived inputs."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    result = frame.loc[:, MOTION_FEATURES].copy()
    if feature_set in {"orientation", "pose"}:
        for column in ORIENTATION_FEATURES[len(MOTION_FEATURES) :]:
            result[column] = frame[column]
    if feature_set == "pose":
        for joint in POSE_JOINTS:
            for axis in "xyz":
                result[f"{joint}_relative_{axis}"] = frame[f"{joint}_{axis}"] - frame[
                    f"pelvis_{axis}"
                ]
    return result.loc[:, FEATURE_SETS[feature_set]]


def contiguous_segments(frame: pd.DataFrame, max_gap_seconds: float) -> Iterable[pd.DataFrame]:
    """Yield time-ordered segments that do not bridge tracking gaps."""

    ordered = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    segment_ids = ordered[TIMESTAMP_COLUMN].diff().fillna(0).gt(max_gap_seconds).cumsum()
    for _, segment in ordered.groupby(segment_ids, sort=False):
        yield segment.reset_index(drop=True)


def make_windows(
    segment: pd.DataFrame,
    feature_set: str,
    window_frames: int,
    stride_frames: int,
) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    """Create fixed windows ending before engagement with a future-intent label."""

    if window_frames <= 0 or stride_frames <= 0:
        raise ValueError("Window and stride sizes must be positive")
    if len(segment) < window_frames:
        return [], [], []

    features = build_features(segment, feature_set).to_numpy(dtype=np.float32)
    engaged = segment[ENGAGED_COLUMN].fillna(False).astype(bool).to_numpy()
    labels = segment[TARGET_COLUMN].fillna(False).astype(bool).to_numpy()
    timestamps = segment[TIMESTAMP_COLUMN].to_numpy(dtype=float)

    windows: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    for end in range(window_frames - 1, len(segment), stride_frames):
        start = end - window_frames + 1
        if engaged[start : end + 1].any():
            continue
        windows.append(features[start : end + 1])
        targets.append(int(labels[end]))
        metadata.append(
            {
                "window_start_timestamp": float(timestamps[start]),
                "window_end_timestamp": float(timestamps[end]),
            }
        )
    return windows, targets, metadata


def fit_normalizer(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature imputation means and scaling standard deviations."""

    flattened = train_features.reshape(-1, train_features.shape[-1]).astype(np.float64)
    means = np.nanmean(flattened, axis=0)
    if np.isnan(means).any():
        missing = np.flatnonzero(np.isnan(means)).tolist()
        raise ValueError(f"Training features are entirely missing at indices: {missing}")
    stds = np.nanstd(flattened, axis=0)
    stds[stds < 1e-8] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def apply_normalizer(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    imputed = np.where(np.isnan(features), means, features)
    normalized = (imputed - means) / stds
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized features contain non-finite values")
    return normalized.astype(np.float32)


def stable_sample_id(parts: Sequence[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def preprocess(config: dict[str, Any], feature_set_override: str | None = None, observation_override: float | None = None) -> Path:
    source_root = Path(config["source_root"])
    output_root = Path(config["output_root"])
    feature_set = feature_set_override or str(config["feature_set"])
    observation_seconds = observation_override or float(config["observation_seconds"])
    sample_rate_hz = int(config["sample_rate_hz"])
    window_frames = int(round(observation_seconds * sample_rate_hz))
    if not np.isclose(window_frames / sample_rate_hz, observation_seconds):
        raise ValueError("Observation duration must map to a whole number of frames")

    validation_dates = {str(value) for value in config["validation_dates"]}
    test_dates = {str(value) for value in config["test_dates"]}
    sources = discover_sources(source_root, validation_dates, test_dates)
    output_dir = output_root / f"{feature_set}_{observation_seconds:g}s"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_by_split: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "validation", "test")}
    targets_by_split: dict[str, list[int]] = {key: [] for key in features_by_split}
    metadata_by_split: dict[str, list[dict[str, Any]]] = {key: [] for key in features_by_split}
    manifest: list[dict[str, Any]] = []

    for source in sources:
        manifest.append(
            {
                "relative_path": source.relative_path,
                "lobby": source.lobby,
                "recording_date": source.recording_date,
                "scenario_id": source.scenario_id,
                "split": source.split,
            }
        )
        frame = pd.read_csv(source.path, usecols=raw_columns_for(feature_set), low_memory=False)
        for pid, track in frame.groupby(PERSON_COLUMN, sort=False):
            segment_index = 0
            for segment in contiguous_segments(track, float(config["max_gap_seconds"])):
                windows, targets, window_metadata = make_windows(
                    segment,
                    feature_set=feature_set,
                    window_frames=window_frames,
                    stride_frames=int(config["stride_frames"]),
                )
                for window, target, item_metadata in zip(windows, targets, window_metadata):
                    sample_id = stable_sample_id(
                        [source.lobby, source.scenario_id, str(pid), str(segment_index), str(item_metadata["window_end_timestamp"])]
                    )
                    features_by_split[source.split].append(window)
                    targets_by_split[source.split].append(target)
                    metadata_by_split[source.split].append(
                        {
                            "sample_id": sample_id,
                            "split": source.split,
                            "lobby": source.lobby,
                            "recording_date": source.recording_date,
                            "scenario_id": source.scenario_id,
                            "person_id": int(pid),
                            "segment_index": segment_index,
                            "label": target,
                            **item_metadata,
                        }
                    )
                segment_index += 1

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in features_by_split:
        if not features_by_split[split]:
            raise ValueError(f"No windows created for split: {split}")
        arrays[split] = (
            np.stack(features_by_split[split]).astype(np.float32),
            np.asarray(targets_by_split[split], dtype=np.uint8),
        )

    means, stds = fit_normalizer(arrays["train"][0])
    stats: dict[str, Any] = {}
    for split, (features, targets) in arrays.items():
        normalized = apply_normalizer(features, means, stds)
        np.savez_compressed(output_dir / f"{split}.npz", features=normalized, targets=targets)
        pd.DataFrame(metadata_by_split[split]).to_csv(output_dir / f"{split}_metadata.csv", index=False)
        positives = int(targets.sum())
        stats[split] = {
            "samples": int(len(targets)),
            "positive": positives,
            "negative": int(len(targets) - positives),
            "positive_rate": positives / len(targets),
        }

    split_dates = {
        split: sorted({item["recording_date"] for item in manifest if item["split"] == split})
        for split in arrays
    }
    if any(set(split_dates[left]) & set(split_dates[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Recording dates overlap across splits")

    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    preprocessing_metadata = {
        "feature_set": feature_set,
        "feature_names": list(FEATURE_SETS[feature_set]),
        "observation_seconds": observation_seconds,
        "window_frames": window_frames,
        "sample_rate_hz": sample_rate_hz,
        "prediction_horizon_seconds": int(config["prediction_horizon_seconds"]),
        "target_column": TARGET_COLUMN,
        "engagement_exclusion_column": ENGAGED_COLUMN,
        "stride_frames": int(config["stride_frames"]),
        "max_gap_seconds": float(config["max_gap_seconds"]),
        "split_dates": split_dates,
        "imputation_means": means.tolist(),
        "scaling_standard_deviations": stds.tolist(),
        "stats": stats,
    }
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing_metadata, indent=2) + "\n", encoding="utf-8"
    )

    examples = []
    for split in ("train", "validation", "test"):
        features, targets = arrays[split]
        for wanted in (0, 1):
            indices = np.flatnonzero(targets == wanted)
            if len(indices):
                index = int(indices[0])
                examples.append(
                    {
                        "sample_id": metadata_by_split[split][index]["sample_id"],
                        "split": split,
                        "label": wanted,
                        "feature_names": list(FEATURE_SETS[feature_set]),
                        "unnormalized_sequence": np.round(features[index], 4).tolist(),
                    }
                )
    (output_dir / "deidentified_examples.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-safe PAR-D temporal windows.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS))
    parser.add_argument("--observation-seconds", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = preprocess(load_config(args.config), args.feature_set, args.observation_seconds)
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

