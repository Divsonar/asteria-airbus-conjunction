# Asteria: next-build plan

This updates the original six-hour plan using the two teammate contributions. It builds on the Streamlit/Plotly dashboard, row-level scikit-learn risk surrogate, deterministic tier rules, CSV upload/export, grouped holdout evaluation and in-context guided tour.

## Progress in this build

- Added Pc-driven recall and precision, a distance-only baseline, relevant-risk error, time-to-TCA breakdowns, an event-cluster bootstrap interval and visible historical failure cases. A risk-weighted candidate was selected on grouped validation; prior holdout exposure is disclosed.
- Added upload validation and data-quality review flags, rule IDs, event evidence, dataset and model fingerprints, and an assessment PDF. A checksum manifest checks bundle integrity but does not prove lineage.
- Added object-type and review filters, retrospective 24-hour-wide buckets, a message replay slider and a local analyst review log. Spreadsheet-safe and exact machine exports are separate.
- Investigated uncertainty intervals and a supplemental review classifier offline. Both are retained as experiments, not used for operational decisions; results and limitations are in `experiments/`.
- Still needed: organiser confirmation of scoring/target, fresh independent evaluation, and a timed pitch/backup recording. More work is required before calibrated uncertainty or any claim of operational readiness.

## Product boundary

- Keep Python, Streamlit, Plotly, pandas and scikit-learn; extend the existing app and pipeline rather than replacing them with React, FastAPI or a new model framework.
- The model estimates the risk reported on the **same historical message**. It does not forecast the final risk or calculate a physical Pc. Keep the analyst in control and retain the prominent model-performance warning.
- Keep the Airbus challenge's direct tier rules and precedence. A review reason is an overlay, not a new tier or an automatic clearance.
- The test file's risk labels remain out of fitting and model selection. Confirm the organiser's intended target, scoring and output format before using those labels for a frozen evaluation.
- The existing holdout has already been inspected. If we tune in response to its results, describe subsequent checks on it as exploratory, not an untouched final evaluation. Seek genuinely unseen events or organiser scoring for independent evidence.

## Prioritized additions

### 1. Make the failure evidence more useful

- Add critical recall, precision and missed counts for the **Pc-driven subset**: recorded log-risk ≥ −4 and miss distance ≥ 1 km. Report row and latest-event results with denominators; distinguish these from the existing risk-only and combined-rule metrics.
- Add a distance-only baseline, error for true log-risk ≥ −6, and results by time-to-TCA bucket. Compare `max_risk_estimate` as a proxy only after verifying its encoding. Retain the existing median baseline and feature ablation.
- Show a small gallery of actual held-out failure examples, including a missed Pc-driven critical and a false alarm when present. Show the relevant input, predicted value and rule, with clear labels.
- Trial improvements within scikit-learn, selected on event-grouped training/validation data for both error and Pc-driven misses. Compare against the existing model and track false alarms. Keep the current model if the candidate does not improve the agreed validation measures; never promise zero misses.

**Done when:** the evidence page shows the new subset metrics and denominators, compares like-for-like data, and displays real failures without making a deployment claim.

### 2. Validate uploads and flag cases for review

- Add a visible validation summary for missing columns, invalid numeric values, missing fields, negative distances and inconsistent RTN magnitude versus recorded miss distance. Keep a stable source-row ID and do not silently drop rows.
- Flag unusually large radial sigmas for inspection. Treat the teammate's suggested placeholder threshold as a hypothesis until the dataset documentation or organisers confirm it; do not silently replace such values as fact.
- Add clear reason-coded **Analyst review** warnings for invalid, missing or questionable inputs. Preserve a critical flag if a valid critical condition is already met. Explain that these are data-quality checks, not calibrated uncertainty or proof that unflagged rows are safe.

**Done when:** a valid upload still predicts and exports, a malformed upload receives actionable feedback, and questionable rows retain their identity and warning.

### 3. Make each decision traceable

- Add stable rule IDs to the existing deterministic rules, showing the input value, units and why each rule fired. Link to the Airbus brief's risk-classification section; only add NASA/ESA section references after checking the actual source and revision.
- Expand the event detail with the source dataset and row, message time to TCA, model version, predicted log-risk/Pc, miss distance, rule triggers and data-quality warnings.
- Add a downloadable **event-assessment PDF** using a lightweight Python PDF library if time allows. Include generation time separately from message time, evidence, thresholds, historical/surrogate labels and limitations. It is a review artifact, not a maneuver authorization, execution record or official report.

**Done when:** an analyst can reproduce the displayed flag from its inputs and rule, and the PDF matches the selected event without invented values.

### 4. Add clear retrospective warning views

- Add object-type filtering and 24-hour-wide time-to-TCA buckets with counts for <1 km and 1–5 km, deduplicating events using a clearly stated message-selection rule.
- Keep the existing 24-hour example and clearly state that these are historical per-message relative times, not a simultaneous live forecast. The supplied test messages are all more than two days from TCA.
- Add concise, computed insight cards only where the underlying counts are shown. Preserve the existing 3D RTN geometry and history plots.

**Done when:** filters, counts and exports reconcile on a small checked sample, and no view suggests that historical messages share an observation timestamp.

### 5. Optional, after the core is verified

- A replay scrubber through an event's successive **estimates of position at TCA**, explicitly not an animated orbital trajectory. Use genuinely out-of-sample predictions if replay is presented as model evidence.
- A manual analyst review note or decision log, clearly local to the demo, with timestamp and event reference. Do not call it tamper-evident without implementing and testing that property.
- Opt-in alert audio and restrained visual polish. Keep the tour usable and avoid automatic sound.
- Calibrated intervals or threshold probabilities only if there is enough time for separate event-grouped calibration and coverage testing; otherwise retain the point estimate and its explicit limitation.

## Suggested six-hour schedule

Assumes two or three contributors working in parallel; for a solo build, take items 1–3 in order and treat the rest as stretch work.

| Time | Main work | Checkpoint |
|---|---|---|
| 0:00–0:30 | Confirm organiser scoring question; agree rule IDs, report fields and test fixtures. | Freeze the existing MVP and a working demo path. |
| 0:30–2:00 | One person adds risk-focused metrics and evaluates a small model experiment; another adds upload validation and review reasons. | Compare candidate to current model on grouped validation; demonstrate valid and malformed uploads. |
| 2:00–3:15 | Add rule provenance and event evidence; surface failure cases. | A selected event's values and triggered rule are traceable. |
| 3:15–4:15 | Add retrospective buckets/type filter; build the assessment PDF if core work is stable. | Reconcile sample counts and inspect a generated report. |
| 4:15–5:00 | Run targeted rule, upload, metric, export and dashboard checks; update the guided tour where needed. | Retain a runnable, verified demo and honest results. |
| 5:00–6:00 | Rehearse the three-minute presentation, record a backup and submit according to the event rules. | Show the analyst workflow, a failure/review case and measured evidence. |

## Keep out of the core MVP

Do not add automatic burn execution, simulated claims of a completed maneuver, unsupported separate orbital tracks, or LLM-generated numeric justifications. Defer a framework rewrite, live feeds, fleet-scale 3D, a full handbook assistant and hash-chained multi-user audit until the core is validated. The scope's numerical targets are goals to test, not results already achieved.
