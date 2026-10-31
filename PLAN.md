# Asteria: updated build and verification plan

The 31 October clarification changes the supervised target to an event's final recorded tier, predicted from a message available at least two days before TCA. Preserve the existing Python, Streamlit, Plotly, pandas, scikit-learn and SQLite stack, interactive dashboard, guided tour and analyst workflow.

## Fixed data contract

- Use the official workbook's 13,154 final-tier labels with the unchanged training CSV.
- Select the latest eligible CDM per event: minimum time-to-TCA ≥ 2 days, with a stable source-row tie-break.
- Train and evaluate on 11,942 eligible events: 834 Critical, 2,162 Elevated and 8,946 Safe. Exclude 1,212 events with no eligible message.
- Use 12 permitted telemetry features. Exclude event IDs, recorded risk, final tier and other final-label fields from model inputs.
- Keep event IDs scoped to their source file. Never join training outcomes to demonstration test events.
- Interpret the target as the final recorded tier. There are 389 negative final time-to-TCA values and 3,050 final records still at least two days before TCA; neither implies verified collision truth.

## Migration sequence

1. **Audit the update.** Retain the updated challenge brief, clarification PDF and label workbook under docs/. Record source checksums and confirm the original CSVs are unchanged.
2. **Rebuild the supervised task.** Make a 60/20/20 event-stratified split with seed 42. Compare unweighted and balanced HistGradientBoostingClassifier candidates on validation macro-F1, using Critical recall as the tie-break. Refit the selected configuration on training plus validation events, then evaluate once on the holdout.
3. **Regenerate one consistent artifact bundle.** Replace the obsolete regressor, metrics, predictions, examples and integrity manifest together. Preserve source-row IDs. Retain every inference row, but withhold class probabilities for missing time-to-TCA or values below two days.
4. **Update the product.** Label model outputs as predicted final tiers and uncalibrated class probabilities. Keep recorded-message tier evidence separate. Update the evidence screen, error examples, report text, exports, guided tour and all claims that depended on the old target.
5. **Verify the contract.** Check the two-day boundary, stable row selection, event separation, train/test namespace isolation, label exclusion, probability columns, units, exact distance/Pc rules, report generation and artifact integrity. Inspect the app's queue, replay, 3D view and model evidence locally.
6. **Publish the reviewed bundle.** Sync source, artifacts, documentation, banner and showcase video into the private deployment checkout. Commit and push, then verify Streamlit's rebuild, the displayed target, current metrics and artifact checksums.

The regenerated classifier has 93.01% holdout accuracy and macro-F1 0.8708 across 2,389 events. The distance-only baseline is slightly stronger overall (93.22%, 0.8717); the classifier's higher Critical recall trades against precision. Preserve that comparison in the presentation.

## Units and rules

Time-to-TCA is in days; miss distance and relative-position components are in metres. Recorded risk is log10(Pc), with −30 treated as approximately zero. Existing unit conversions were already correct.

Apply Critical first when log10(Pc) ≥ −4 or miss distance < 1,000 m. Otherwise apply Elevated when log10(Pc) ≥ −6 or miss distance ≤ 5,000 m, then Safe for valid inputs. Exactly 1,000 m is not distance-only Critical; exactly 5,000 m is Elevated. These recorded-message rules do not convert classifier probabilities into physical Pc.

## Product boundaries

- Headline holdout metrics use one latest eligible message per event. Replay predictions across earlier eligible messages are inspection views, not additional independent evaluation samples.
- Test demonstration events have no official final outcomes. Do not show cross-file matches as labels.
- Class probabilities are uncalibrated. Data-quality flags describe input checks, not calibrated prediction uncertainty.
- Historical buckets are not synchronized live forecasts. The 3D view is relative encounter geometry, not orbit propagation.
- The final tier is a recorded label, not a collision outcome. Analyst reviews and reports do not authorize maneuvers.
- Prior exploration of this dataset must remain disclosed. A new target and event split do not make these data an independent operational validation.
- The existing showcase is an interface recording captured before the clarification. Label it accordingly and direct viewers to the live app for current model results.
- Earlier same-message, Pc-driven and conformal studies remain archived under experiments/; their scores are obsolete for the current target.

## Follow-up work after the migration

- Validate against independent labelled events or organiser scoring, then assess class-probability calibration and error trade-offs.
- Confirm the organiser's exact submission schema before creating a scoring-specific export.
- Add durable review storage and user authentication if the prototype becomes a shared operational service.
- Re-record the showcase with the new target and current evidence when time permits.

Live feeds, automated maneuvers, orbit propagation and framework rewrites remain outside this MVP.
