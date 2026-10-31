# Archived pre-clarification experiments

These experiments target same-message log risk and are superseded by the official final-tier forecast pipeline in `src/pipeline.py`. They are not deployed or used for current metrics. Their saved outputs are historical only. The archived pipeline writes to `experiments/legacy_artifacts`, never the production artifact directory.

# Exploratory model studies

These studies are offline research tools. They do not update the production dashboard, model or prediction bundle, and none reads the supplied test file's `risk` labels. Each uses the training-file event split in `artifacts/split_events.json`. The historical holdout and some validation results were inspected during development, so none is a fresh blind evaluation. This is a selected, high-risk-enriched dataset, not a random operational population.

The target throughout is the risk **reported in the same message**, not whether a collision actually occurred or the final risk at closest approach.

## Reproducing

From the project root in PowerShell:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& .\.venv312\Scripts\python.exe experiments/compare_models.py
& .\.venv312\Scripts\python.exe experiments/conformal_study.py --output experiments/conformal_results.json
& .\.venv312\Scripts\python.exe experiments/critical_review_study.py
```

### `compare_models.py`

Refits the original unweighted estimator using the saved training and validation event groups, and compares it with the current model on the same already-inspected holdout. It saves the reconstructed reference model as `unweighted_reference.joblib` and the comparison as `model_comparison.json`. The reconstructed reference reproduces the earlier saved holdout metrics; it is not a different independent test.

### `conformal_study.py`

Fits a separate weighted estimator on one event group and calibrates symmetric log-risk residual intervals on a disjoint event group. It compares row-level calibration with an event-maximum method, then measures coverage and interval width on the previously inspected holdout. The intervals are extremely wide on this data. Row-level calibration does not imply whole-event coverage; even event-level empirical coverage on a selected historical sample is not a deployment guarantee. This experiment is not connected to the dashboard.

### `critical_review_study.py`

Fits a separate class-balanced classifier on training events and selects a threshold using validation misses from a separately fitted point estimator. The score is uncalibrated and would only suggest an additional review; it does not change a challenge tier. The 95% validation-miss target generated excessive review burden in the inspected holdout. An additional threshold sweep is explicitly post-hoc and hypothesis-generating; it is not a validated operating policy. The dashboard does not use this experiment.
