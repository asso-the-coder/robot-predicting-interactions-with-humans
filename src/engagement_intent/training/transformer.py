"""Train and validation-select the Transformer intent classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from engagement_intent.training.lstm import load_config, train_sequence_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Transformer classifier.")
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
    run_dir = train_sequence_model(config, model_type="transformer")
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
