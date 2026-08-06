# Early Prediction of Human Engagement Intent

This APS360 project studies supervised binary classification of pedestrian intent to interact with a public social robot from short temporal observation sequences.

## Current status

The project is in the data-audit stage. The primary source is the CC0-licensed [PAR-D Shutter Interaction Dataset](https://doi.org/10.60600/YU/KFFQPF). Raw data must remain unchanged and must not be committed.

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
  --archive data/raw/par-d/DOWNLOADED_ARCHIVE.zip `
  --output-dir data/processed/par-d-v1
```

Build leakage-safe, normalized temporal windows after staging:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.data.preprocess `
  --config configs/preprocess.yaml `
  --feature-set orientation `
  --observation-seconds 2
```

The verified schema, target semantics, exclusions, and split policy are documented in [`DATA_AUDIT.md`](DATA_AUDIT.md).

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

Training and evaluation commands will be documented only after the real dataset schema and label semantics have been audited.
