# Repository Guidance

## Project Purpose

- Build an independent APS360 implementation for **early prediction of human engagement intent for social robot interaction**.
- Treat the task as supervised binary classification: from a fixed-length pedestrian observation sequence, predict `interact` or `not_interact` before the interaction decision.
- Study two questions: how early intent can be predicted and whether richer orientation or pose features improve over simple motion features.
- Use the PAR-D paper and release as cited source material, not as code or prose to reproduce blindly. The project must include its own data audit, preprocessing, models, experiments, and interpretation.
- Keep the application low risk: greeting, waiting, or avoiding interruption. Do not frame it for surveillance or consequential decisions.

## Repository Map

Current tracked files:

- `README.md`: minimal project title and one-line description; it does not yet document setup or usage.
- `.gitignore`: excludes the local virtual environment, course documents, raw/processed data, checkpoints, run outputs, caches, and editor files.
- `AGENTS.md`: this contributor guide.

Current local-only files and directories:

- `docs/`: ignored PDFs containing the syllabus, rubrics, guidelines, examples, and annotated proposal, plus an ignored `project_brief.md`. Use these as requirements; do not stage or push them.
- `.venv/`: ignored local environment. It is not a reproducible dependency specification.
- `.agents/`: currently empty and untracked.

No notebooks, Python packages, configuration files, dependency manifests, data scripts, training scripts, evaluation scripts, tests, datasets, or experiment outputs exist yet.

Proposed structure (create incrementally, not all at once):

```text
configs/                         # Tracked experiment and preprocessing configs
src/engagement_intent/
  data/                          # Inspection, splitting, preprocessing, datasets
  models/                        # Baseline interfaces, LSTM, optional Transformer
  training/                      # Training loops and checkpoint selection
  evaluation/                    # Metrics, predictions, and plots
tests/                           # Fast unit and smoke tests
notebooks/                       # Optional exploration; no hidden core logic
data/raw/                        # Immutable local PAR-D release; ignored
data/processed/                  # Reproducible derived artifacts; ignored
outputs/                         # Unique experiment directories; ignored
report/                          # Tracked LaTeX source and bibliography when started
```

## Intended System Architecture

1. Inspect the downloaded PAR-D release, license, documentation, identifiers, sampling rate, labels, missingness, and feature availability. Record real de-identified input and output examples.
2. Preserve `data/raw/` unchanged. Put every transformed artifact and its metadata under `data/processed/`.
3. Define a stable grouping unit from actual PAR-D identifiers. Assign groups to train, validation, and test before constructing overlapping windows.
4. Convert pedestrian tracks into fixed-length temporal windows anchored before a documented decision/event time. Support early-prediction windows such as 1, 2, and 3 seconds when the sampling and annotation semantics allow it.
5. Derive only justified features. Candidate sets are motion (relative position, distance, velocity, movement direction, distance change), orientation, and pose. Do not claim PAR-D provides a feature until verified.
6. Fit imputation, scaling, feature selection, class weighting, and any learned preprocessing using the training split only. Persist these parameters with the split definition.
7. Train a majority-class baseline, then an interpretable logistic-regression baseline on summary features.
8. Train the primary LSTM sequence classifier. Add a Transformer encoder with positional information only after the data pipeline and LSTM comparison are reliable.
9. Select hyperparameters and checkpoints using validation F1. Evaluate the untouched test set once for the selected model/comparisons.
10. Save metrics, predictions, curves, confusion matrices, qualitative examples, and plot metadata so report figures can be regenerated.

## Data Handling Rules

- Never commit raw or restricted data. Confirm PAR-D redistribution and privacy terms before sharing any derivative.
- Never assume file names, columns, coordinate frames, units, frame rates, labels, or event timing. Inspect and document them.
- Preserve raw files byte-for-byte. Make preprocessing deterministic and rerunnable.
- Split by scenario, participant, recording, trajectory, sequence, or the strongest available grouping unit. Assert that no group overlaps across splits.
- Perform grouping before window creation so adjacent or overlapping observations cannot leak across splits.
- Fit normalization and all data-dependent preprocessing on training data only. Apply the saved transform unchanged to validation and test data.
- Report class counts and rates per split. Check missing values, invalid values, duplicate identifiers, sequence lengths, and label consistency.
- Document filtering, exclusions, label mapping, feature derivation, coordinate transforms, imputation, padding, masks, truncation, and prediction anchors.
- Keep a versioned split manifest and preprocessing metadata. A seed alone is not a sufficient split record.
- Show concrete, de-identified sample inputs and their actual labels in data documentation and reports, as requested in proposal feedback.
- Treat final test data and any separately collected new-data evaluation as untouchable during model selection.

## Modeling Rules

- Establish both a majority-class predictor and logistic regression before interpreting deep-model gains.
- Keep the LSTM as the primary deep model. Treat the Transformer as an optional, serious comparison motivated by spatiotemporal inputs and proposal feedback.
- Define each feature tensor explicitly. Prefer sequence input shape `[batch, time, features]`; document masks/lengths and keep output logits consistently `[batch]` or `[batch, 1]`.
- Return raw logits. Use `torch.nn.BCEWithLogitsLoss` for binary training and do not apply sigmoid inside a model trained with that loss.
- Apply sigmoid only when converting logits to probabilities for metrics or inference. Select any non-default threshold on validation data only.
- Encode temporal order explicitly. LSTMs encode order recurrently; Transformers require positional or temporal encoding.
- Keep feature-set, hidden size, layers, dropout, pooling, loss weighting, and decision threshold configurable.
- Support CPU and CUDA without embedding a device in saved tensors or model code.
- Handle imbalance only after measuring it. Record class weights or sampling behavior in the run configuration.
- Keep data loading, feature construction, models, optimization, and evaluation in separate modules.

## Experiment Rules

- Give every run a unique output directory; never overwrite an earlier run.
- Store the following with every run: seed, code revision, split version, preprocessing version, feature set, observation-window length, model type, full hyperparameters, checkpoint path, validation metrics, and artifact paths.
- Record package versions and device information. Track configuration files in Git and generated results under ignored output directories unless a small summary is intentionally reviewed for inclusion.
- Prefer one configuration-driven entry point per operation over copied training scripts.
- Choose checkpoints using validation performance, primarily F1. Never choose a checkpoint from test performance.
- Reserve final test metrics until model and threshold selection are complete. Label exploratory validation and final test results unmistakably.
- Include early-prediction comparisons across feasible observation lengths and ablations from motion-only to orientation/pose features.
- Compare models on the same split and preprocessing version. Report parameter counts or training cost when architectural complexity differs materially.
- Never invent, estimate, or backfill metrics. Missing experiments must remain marked as pending.

## Evaluation Rules

- Report positive-class F1, precision, recall, accuracy, confusion matrix, and loss for required splits.
- State the positive label and confusion-matrix ordering. Include per-class metrics when imbalance makes a single score incomplete.
- Inspect the majority-class rate before interpreting accuracy.
- Save prediction-level records with identifiers, labels, probabilities/logits, split, run ID, and observation-window metadata where privacy terms permit.
- Include representative true positives, true negatives, false positives, and false negatives. Use them to discuss motion, orientation, ambiguity, and likely failure modes.
- Evaluate selected models on genuinely unseen data as required by the final rubric. Clearly state whether this means a held-out PAR-D group or a separately sourced/collected set.
- Compute metrics through one shared evaluation path to avoid notebook/script discrepancies.

## Coding Conventions

- Use Python and PyTorch for neural modeling unless a documented repository decision changes the framework. Logistic regression may use a standard library such as scikit-learn.
- Use descriptive names, type hints where practical, and docstrings for nontrivial public functions/classes.
- Prefer `pathlib.Path`; never commit machine-specific absolute paths.
- Prefer clear vectorized operations over unnecessary Python loops, while keeping temporal and masking behavior readable.
- Centralize random seeding and configuration loading. Do not scatter unexplained constants through scripts.
- Validate inputs and fail with actionable errors. Do not silently catch exceptions or silently discard malformed samples.
- Document every new dependency in the chosen dependency manifest and explain unusual dependencies in the README.
- Keep notebooks exploratory. Move reusable or submission-critical logic into tested modules.
- Keep code understandable enough that the student can explain every submitted component.

## Testing and Validation

No test framework or test commands currently exist. Add lightweight tests as each component appears:

- Parse real filenames/identifiers and reject malformed cases.
- Verify grouped splitting is deterministic and has zero group overlap.
- Verify sequence windows, prediction anchors, padding/truncation, masks, and labels on tiny synthetic tracks.
- Verify normalization uses training statistics and can be serialized/reloaded.
- Verify each model accepts documented shapes and emits one finite logit per sample.
- Verify loss computation for both classes and any class weighting.
- Verify checkpoint save/load reproduces logits for a fixed batch.
- Run a tiny overfit test showing the LSTM can learn a small deterministic dataset.
- Add an end-to-end CPU smoke test from processed samples through evaluation.

## Commands

There are currently no reproducible project commands: no dependency file, package, CLI, preprocessing entry point, trainer, evaluator, plot generator, or test suite has been implemented. The existing `.venv/` is local-only and must not be treated as setup documentation.

The following are **proposed examples, not working commands**. Implement and update them as the package is created:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m engagement_intent.data.inspect --data-dir data/raw
.\.venv\Scripts\python.exe -m engagement_intent.data.preprocess --config configs/preprocess.yaml
.\.venv\Scripts\python.exe -m engagement_intent.training.baseline --config configs/baseline.yaml
.\.venv\Scripts\python.exe -m engagement_intent.training.lstm --config configs/lstm.yaml
.\.venv\Scripts\python.exe -m engagement_intent.evaluation.evaluate --config configs/evaluate.yaml --checkpoint checkpoints/model.pt
.\.venv\Scripts\python.exe -m engagement_intent.evaluation.plots --run-dir outputs/RUN_ID
.\.venv\Scripts\python.exe -m pytest
```

Do not advertise a proposed command as operational until it is implemented and smoke-tested.

## Agent Workflow

- Read `AGENTS.md`, relevant code/configs, `README.md`, and Git status before editing.
- Consult ignored course documents when assignment semantics matter, but never stage them.
- Inspect actual dataset examples before deciding schema, labels, grouping, windows, or supported features. Ask for clarification if these remain ambiguous.
- Make the smallest coherent change that completes the requested milestone. Do not rewrite unrelated files.
- Preserve user changes and working outputs. Do not delete or regenerate material without understanding ownership and cost.
- Add or update focused tests with behavioral changes, then run relevant tests and a smoke check.
- Review diffs for data leakage, hard-coded paths, accidental artifacts, secrets, and unsupported claims.
- Summarize changed files, commands run, results, and unresolved issues. Never invent experimental results.
- Commit coherent milestones with meaningful messages and push periodically when authentication is available.

## Git and Artifact Hygiene

- Check `git status --short --ignored` before staging. Explicitly stage intended paths; do not use broad staging when local data or documents are present.
- Do not commit `docs/` course PDFs/brief, `.venv/`, raw or processed datasets, secrets, cached tensors, checkpoints, run directories, or generated plots unless explicitly approved and appropriate.
- Keep small configuration, split-generation code, schemas, tests, and reproducibility metadata tracked.
- Update `.gitignore` when a new generated artifact class appears, but do not hide source, configuration, or tests.
- Do not force-push, rewrite history, reset destructively, or remove another contributor's work without explicit permission.
- Use meaningful commits aligned with working milestones. Do not commit known-broken intermediate states merely to create activity.

## Academic Integrity and Attribution

- Cite the PAR-D dataset/paper and every external dataset, pretrained component, algorithm implementation, or adapted script.
- Do not copy paper or sample-project prose. Write original explanations grounded in this implementation and its results.
- Understand borrowed/generated code before retaining it; preserve license notices and make attribution explicit.
- Keep an experiment trail sufficient to support every table, figure, and claim in the report.
- Never fabricate outputs, metrics, experiments, dataset properties, citations, or qualitative examples.

## Known Uncertainties

- The PAR-D release is not present, so its actual schema, formats, labels, feature modalities, sampling rate, coordinate system, grouping identifiers, license, and missing-data patterns are unverified.
- The precise positive-label definition and prediction anchor are not yet established from actual annotations. "First 1/2/3 seconds" must be reconciled with what information is available at inference time.
- It is unknown whether PAR-D directly provides head/body orientation and pose or whether those features must be derived. Do not promise unavailable modalities.
- The safest leakage-prevention group (scenario, participant, recording, trajectory, or combination) depends on identifiers in the release.
- The rubric's new-data evaluation requirement still needs an exact plan that remains feasible, ethical, and untouched by tuning.
- Dependencies, supported Python version, configuration format, experiment tracker, and deterministic-computation policy have not been selected.

## Next Priorities

1. Obtain and audit PAR-D, then document its license, schema, label semantics, grouping keys, class balance, missingness, sampling, and concrete de-identified input/output examples.
2. Scaffold the Python package, dependency manifest, typed configuration format, README setup instructions, and a fast test suite.
3. Implement deterministic grouped splitting and temporal-window preprocessing with leakage, normalization, and serialization tests.
4. Implement majority-class and logistic-regression baselines with the shared evaluation pipeline.
5. Implement and smoke-test the configurable LSTM, then run early-window and feature-ablation experiments before deciding whether the Transformer adds useful scope.
