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

Place the original PAR-D download under `data/raw/par-d/`. Inventory it without extracting or modifying the archive:

```powershell
.\.venv\Scripts\python.exe -m engagement_intent.data.inspect `
  --data-dir data/raw/par-d `
  --output data/processed/raw_inventory.json
```

Run the current tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Training and evaluation commands will be documented only after the real dataset schema and label semantics have been audited.
