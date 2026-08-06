"""Train and validation-select the primary LSTM intent classifier."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from engagement_intent.evaluation.metrics import binary_metrics, select_f1_threshold
from engagement_intent.models import LSTMClassifier, TransformerClassifier


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_split(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    archive = np.load(processed_dir / f"{split}.npz")
    metadata = pd.read_csv(processed_dir / f"{split}_metadata.csv")
    features = archive["features"].astype(np.float32)
    targets = archive["targets"].astype(np.float32)
    if len(features) != len(targets) or len(targets) != len(metadata):
        raise ValueError(f"Feature, target, and metadata counts disagree for {split}")
    return features, targets, metadata


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            logits_parts.append(logits.detach().cpu().numpy())
            target_parts.append(targets.detach().cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.concatenate(logits_parts),
        np.concatenate(target_parts),
    )


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def create_run_dir(output_root: Path, model_type: str = "lstm") -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{model_type}_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_sequence_model(
    model_type: str, input_size: int, settings: dict[str, Any]
) -> nn.Module:
    """Construct a configured temporal classifier without binding it to a device."""
    if model_type == "lstm":
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=int(settings["hidden_size"]),
            num_layers=int(settings["num_layers"]),
            dropout=float(settings["dropout"]),
        )
    if model_type == "transformer":
        return TransformerClassifier(
            input_size=input_size,
            model_size=int(settings["model_size"]),
            num_heads=int(settings["num_heads"]),
            num_layers=int(settings["num_layers"]),
            feedforward_size=int(settings["feedforward_size"]),
            dropout=float(settings["dropout"]),
            max_sequence_length=int(settings["max_sequence_length"]),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def train_sequence_model(config: dict[str, Any], model_type: str) -> Path:
    """Train one temporal model and select its checkpoint and threshold on validation F1."""
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config["device"]))
    processed_dir = Path(config["processed_dir"])
    train_features, train_targets, _ = load_split(processed_dir, "train")
    validation_features, validation_targets, validation_metadata = load_split(
        processed_dir, "validation"
    )
    settings = config["training"]
    train_loader = make_loader(
        train_features,
        train_targets,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(settings["num_workers"]),
    )
    validation_loader = make_loader(
        validation_features,
        validation_targets,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(settings["num_workers"]),
    )
    model_settings = config["model"]
    model = build_sequence_model(
        model_type, input_size=int(train_features.shape[-1]), settings=model_settings
    ).to(device)
    positive_weight_setting = settings["positive_weight"]
    if positive_weight_setting == "auto":
        positive_weight = float((len(train_targets) - train_targets.sum()) / train_targets.sum())
    else:
        positive_weight = float(positive_weight_setting)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    threshold_config = config["threshold_selection"]
    thresholds = np.arange(
        float(threshold_config["minimum"]),
        float(threshold_config["maximum"]) + float(threshold_config["step"]) / 2,
        float(threshold_config["step"]),
    )

    run_id, run_dir = create_run_dir(Path(config["output_root"]), model_type=model_type)
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    best_probabilities: np.ndarray | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(settings["epochs"]) + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            gradient_clip_norm=float(settings["gradient_clip_norm"]),
        )
        validation_loss, validation_logits, observed_targets = run_epoch(
            model, validation_loader, loss_function, device
        )
        probabilities = torch.sigmoid(torch.from_numpy(validation_logits)).numpy()
        threshold, metrics = select_f1_threshold(observed_targets, probabilities, thresholds)
        metrics["weighted_bce_loss"] = validation_loss
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce_loss": train_loss,
                "validation_weighted_bce_loss": validation_loss,
                "validation_f1": metrics["f1"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "threshold": threshold,
            }
        )
        print(json.dumps(history[-1]))

        if metrics["f1"] > best_f1 + 1e-9:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = metrics
            best_probabilities = probabilities.copy()
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "input_size": int(train_features.shape[-1]),
                    "model_config": model_settings,
                    "selected_threshold": best_threshold,
                    "best_epoch": best_epoch,
                    "validation_metrics": best_metrics,
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(settings["early_stopping_patience"]):
                break

    if best_metrics is None or best_probabilities is None:
        raise RuntimeError("Training did not produce a checkpoint")
    predictions = validation_metadata.copy()
    predictions["target"] = validation_targets.astype(np.uint8)
    predictions["probability"] = best_probabilities
    predictions["prediction"] = (best_probabilities >= best_threshold).astype(np.uint8)
    predictions["model"] = model_type
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "learning_curves.csv", index=False)

    run_metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(),
        "model_type": model_type,
        "model_config": model_settings,
        "processed_dir": str(processed_dir),
        "seed": seed,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "parameter_count": parameter_count(model),
        "input_sequence_shape": list(train_features.shape[1:]),
        "train_samples": int(len(train_targets)),
        "positive_weight": positive_weight,
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "validation_metrics": best_metrics,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps(run_metadata, indent=2))
    return run_dir


def train_lstm(config: dict[str, Any]) -> Path:
    return train_sequence_model(config, model_type="lstm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the primary LSTM classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.processed_dir is not None:
        config["processed_dir"] = str(args.processed_dir)
    if args.seed is not None:
        config["seed"] = args.seed
    run_dir = train_lstm(config)
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
