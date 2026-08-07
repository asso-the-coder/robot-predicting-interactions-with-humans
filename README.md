# Early Prediction of Human Engagement Intent

This APS360 project studies supervised binary classification of pedestrian intent to interact with a public social robot from short temporal observation sequences.

## Current status

The data audit, leakage-safe preprocessing, baseline models, LSTM, positional Transformer, feature/window ablations, and one frozen held-out evaluation are implemented. The primary source is the CC0-licensed [PAR-D Shutter Interaction Dataset](https://doi.org/10.60600/YU/KFFQPF). Raw data must remain unchanged and must not be committed.

## Results at a glance

The positive label means that a pedestrian begins interacting within four seconds after a two-second observation window. Positive-class F1 is the primary metric because only 5% of all processed windows are positive; an always-negative predictor still obtains 91.4% test accuracy but zero F1.

| Selected model | Held-out accuracy | Precision | Recall | Positive-class F1 |
| --- | ---: | ---: | ---: | ---: |
| Orientation logistic regression | 0.861 | 0.239 | 0.284 | **0.260** |
| Centered-pose LSTM | 0.862 | 0.115 | 0.092 | 0.102 |

The Transformer was added only after the frozen test evaluation and was evaluated on validation data only. The complete reviewed results, chronology, and run identifiers are in [`RESULTS.md`](RESULTS.md).

## Repository layout

- `configs/`: preprocessing and experiment configurations.
- `src/engagement_intent/`: data inspection/staging, preprocessing, models, training, and final evaluation.
- `tests/`: unit and smoke tests for data handling, models, metrics, and evaluation.
- `DATA_AUDIT.md`: verified dataset schema, target semantics, leakage controls, and limitations.
- `RESULTS.md`: reviewed validation and frozen test results.
- `report/`: LaTeX report source, bibliography, figures, and build instructions.

## Local setup

Create a virtual environment and install the package with development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On an NVIDIA system compatible with CUDA 11.8, replace the default CPU wheel with the official CUDA build:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Place the original PAR-D download under `data/raw/par-d/`. Inventory it without extracting or modifying the archive:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.data.inspect `
  --data-dir data/raw/par-d `
  --output data/processed/raw_inventory.json
```

Stage the release and its nested lobby archives under ignored processed storage. The command refuses to overwrite a non-empty destination:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.data.stage `
  --archive data/raw/par-d/doi-10.60600-yu-kffqpf.zip `
  --output-dir data/processed/par-d-v1
```

Build leakage-safe, normalized temporal windows after staging:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.data.preprocess `
  --config configs/preprocess.yaml `
  --feature-set orientation `
  --observation-seconds 2
```

The verified schema, target semantics, exclusions, and split policy are documented in [`DATA_AUDIT.md`](DATA_AUDIT.md). Reviewed validation and final test results are recorded in [`RESULTS.md`](RESULTS.md).

Train the majority-class and logistic-regression baselines. This command selects a logistic-regression decision threshold using validation F1 and does not read the test split:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.training.baseline `
  --config configs/baseline.yaml
```

Train the primary LSTM. Checkpoints and thresholds are selected using validation F1; the test split is not read:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.training.lstm `
  --config configs/lstm.yaml
```

Train the positional Transformer comparison through the same validation-only selection path:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.training.transformer `
  --config configs/transformer.yaml
```

After all model, feature, window, checkpoint, and threshold choices are frozen, evaluate the selected baseline and LSTM on the final test split:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.evaluation.final `
  --baseline-run outputs/BASELINE_RUN_ID `
  --lstm-run outputs/LSTM_RUN_ID
```

Run the current tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Generated datasets, checkpoints, predictions, and plots remain ignored. Each run records its complete configuration, code revision, device, selected checkpoint, threshold, and validation metrics.

## Final report

The tracked report source is [`report/main.tex`](report/main.tex), with build instructions in [`report/README.md`](report/README.md). The generated PDF is intentionally ignored and submitted separately through the course submission system.
