# Asteria: three-minute demonstration

The task is to predict the **final recorded event tier from a message available at least two days before TCA**. All data is historical. Model class probabilities are not physical collision probabilities, and the product does not make maneuver decisions.

## Before presenting

1. Open the current app and check that artifact verification succeeds and the model target is final event tier.
2. Read the regenerated Model evidence results and class denominators. Choose a current held-out error example; do not reuse old same-message regression scores, Pc-driven subset scores or hard-coded case IDs from a previous model.
3. Keep the updated challenge PDFs, this script and a fallback recording available. The existing 30-second showcase predates the target clarification and demonstrates the interface only.
4. Reset the dataset and filters between rehearsals. Avoid creating misleading analyst-review records during practice.
5. Confirm that the test demonstration does not display training-file final labels.

## Suggested presentation

| Time | What to show | What to say |
|---|---|---|
| 0:00–0:25 | Mission control, event queue and predicted-tier cards. | “We help an analyst prioritize events by predicting the final recorded tier from information available at least two days before closest approach.” |
| 0:25–0:50 | Select an eligible event, inspect message evidence, and rotate the 3D view. | “These are the source measurements and relative encounter geometry. The distance guides are not orbit trajectories.” |
| 0:50–1:15 | Move the replay slider and inspect the message history. | “Replay shows how the historical evidence changes. Only eligible early messages receive forecasts; our headline evaluation uses one message per event.” |
| 1:15–1:40 | Show the predicted final tier beside recorded-message rule evidence; open an assessment PDF. | “The model forecasts a final tier. The explicit rules describe this recorded message. The class probabilities are uncalibrated model estimates, not physical Pc.” |
| 1:40–2:15 | Open Model evidence: current event counts, class results and confusion matrix. | “On 2,389 held-out events, the model detects 143 of 167 Critical events and misses 24, with 36 false Critical alarms. Its Critical recall is slightly higher than the distance baseline, but its overall accuracy and macro-F1 are slightly lower.” |
| 2:15–2:40 | Open a current held-out error example. | “This is where the forecast failed. These data were explored during development; we need independent events before claiming operational reliability.” |
| 2:40–3:00 | Show a review form and the limitations. | “The analyst makes the decision. Our target is the last recorded tier, not a verified collision outcome, and a review does not authorize a maneuver.” |

## Selecting a useful error example

Choose an example from the regenerated holdout predictions, comparing the official true_final_tier and the model's predicted_final_tier. Prefer a missed Critical event, then a false Critical alarm, if present. Show its latest eligible early message and the final recorded label separately. Do not imply that a test-demo event has a known final outcome.

Recheck counts against the current evidence screen after any rebuild. Do not quote scores from the pre-clarification recording or archived experiments.

## Likely judge questions

**What is the prediction target?**

The official workbook's final recorded tier for the event, predicted using its latest message still at least two days before TCA. The target is excluded from the input features.

**How much labelled data is usable?**

11,942 of 13,154 labelled events: 834 Critical, 2,162 Elevated and 8,946 Safe. The other 1,212 events have no eligible early message.

**Does “final” mean a collision was observed?**

No. It means the final recorded message. Of the official final records, 389 have negative time-to-TCA and 3,050 are still at least two days before TCA.

**Are the displayed probabilities Pc?**

The classifier's probabilities describe the three final-tier classes and are uncalibrated. Recorded risk is log10(Pc); it is separate from model output.

**What are the exact tier rules?**

Critical takes precedence at log10(Pc) ≥ −4 or distance < 1,000 m. Otherwise Elevated applies at log10(Pc) ≥ −6 or distance ≤ 5,000 m; valid remaining inputs are Safe. Time is in days and distance is in metres.

**Can it recommend or execute a maneuver?**

No. This prototype supports inspection and review. It does not propagate physical orbits, establish collision truth, or authorize a maneuver.
