# Asteria: conjunction intelligence

An interactive, local-first prototype for Airbus Challenge 1. Explore historical conjunction messages, inspect a relative encounter geometry, review challenge-rule alerts, and download one risk estimate per supplied message. The evidence screen reports event-grouped validation results and a visible failure case.

## Run the MVP

The project has been prepared with a local Python 3.12 environment. In PowerShell, from this directory:

```powershell
& .\.venv312\Scripts\python.exe -m streamlit run app.py
```

Open the local URL Streamlit prints. If the artifacts have not been generated, train and generate predictions first:

```powershell
& .\.venv312\Scripts\python.exe train.py
```

On a different machine, create a Python 3.12 virtual environment and install `requirements.txt`; then run `python train.py` and `python -m streamlit run app.py`. The default input paths are `Challenge 1 Airbus_/train_data.csv` and `Challenge 1 Airbus_/test_data.csv`; use `python train.py --help` to change paths. Model and reports are written to `artifacts/`.

## How to use it

1. Choose **Mission control** and explore the test events. Filter by tier, object type, time or review status. Select an event, drag the 3D encounter view and use the replay slider to inspect successive message estimates. The queue itself always represents the latest available message.
2. Switch to held-out training events for historical 24-hour examples. The bucket chart and counts refer to each selected message's own time to closest approach, not a synchronized or live forecast.
3. Inspect the selected message's review reasons, rule IDs and source values. Download the assessment PDF, or record a local analyst review with a reason. The record documents an assessment; it does not authorize a maneuver. A sidebar guided tour points to the main controls and charts.
4. Choose **Model evidence** for baseline comparisons, probability-driven critical metrics, uncertainty in those measurements, and examples of model errors.
5. Download all row predictions, or upload a telemetry CSV using the dataset selector. Uploaded data needs `event_id` and all 12 permitted feature columns. Any `risk` column is ignored at inference. Review warnings describe data quality; they are not calibrated model uncertainty.

The default prediction download escapes formula-like text for spreadsheet use. The separate exact CSV preserves IDs for machine scoring; avoid opening untrusted exact CSV text in a spreadsheet.

The local server accepts uploads up to 256 MB. CSV parsing reads the permitted telemetry fields and optional source-row ID; unrelated columns, including `risk`, are excluded from inference. Larger files must be split before upload.

Run the checks with `python -m pytest -q` using the project environment. The local review records live in `artifacts/analyst_decisions.sqlite` and can be exported as CSV from the app; they are not a tamper-evident audit system.

## Exploratory interval study

`experiments/conformal_study.py` examines uncertainty intervals separately from the production model. In one previously inspected historical holdout, nominal 80% row-calibrated intervals covered 80.2% of messages but only 44.4% of whole events; event-max calibration covered 80.9% of whole events with a median width of 24.8 log10 units. These intervals are too broad for the MVP's decision view and are not deployed. The experiment uses different event groups for fitting and calibration, does not read test labels, and does not establish operational coverage. Results and caveats are in `experiments/conformal_results.json`.

The previous unweighted model is reconstructed in `experiments/unweighted_reference.joblib`. `experiments/model_comparison.json` compares it with the current model on the same previously inspected holdout. The comparison is exploratory and is not independent validation.

`experiments/critical_review_study.py` also evaluates an experimental classifier as a supplementary review flag. On the inspected holdout it identified all 13 point-model misses in its evaluated subset, but flagged another 13,877 messages; it is not integrated into the dashboard. The threshold was selected on a small validation sample and the experiment does not establish performance on new events.

## Scope and limitations

- The model estimates the **same-message** `risk` value in log10(Pc). It is not a physical Pc calculation, a forecast of the last message's risk, or a calibrated flight-qualified system. The original ESA competition's final-risk prediction target is different.
- The supplied test file contains a `risk` column, but it is excluded from inference and model selection. Training events are split by event ID; event IDs are scoped to their respective input files.
- The artifact manifest checks whether the model, predictions and evaluation files match recorded checksums. Matching hashes establish file integrity relative to that manifest; they do not by themselves prove data or model lineage.
- Thresholds are the challenge's rules, applied with critical precedence. They do not authorize a maneuver. A qualified operator retains the decision.
- Relative R/T/N coordinates describe the reported geometry at closest approach; the visual is not an orbital trajectory. There are no common observation timestamps, and no test messages fall within 24 hours of closest approach.
- The data is a selected research sample with high-risk events overrepresented. Its class frequencies and evaluation do not establish real-world performance. The `max_risk_estimate` feature is already risk-related; the model evidence includes an ablation without it.
- The current exploratory held-out run reports **67.1% risk-only critical recall** (51 of 155 critical-risk messages missed). The distance rule raises combined-tier recall, but does not repair missed probability estimates. The held-out data has been inspected during development; these are not fresh blind-test results. Do not use this model for operational decisions.
- Confirm the organisers' desired prediction target and output format. The included CSV preserves every input row and includes additional context for inspection.

ESA data documentation: https://kelvins.esa.int/collision-avoidance-challenge/data/ . NASA conjunction assessment context: https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/ .
