# Asteria: three-minute demonstration

This script uses the generated artifacts currently in the workspace. All data is historical. The model estimates the risk reported in the same message; it does not forecast the final event risk or make operational decisions. Confirm the organiser's scoring target and export schema separately.

## Before presenting

1. Start the app with the command in `README.md`, and open Mission control.
2. Check that the sidebar says the artifact checksums match and that the research-only warning is visible.
3. Keep this document and a fallback screen recording available. Recheck the examples if the model or artifacts have changed.
4. Reset the data source and filters between rehearsals. The dashboard stores analyst reviews locally; avoid creating a misleading real record during practice.

## Suggested presentation

| Time | What to show | What to say |
|---|---|---|
| 0:00–0:20 | The event queue and critical/elevated cards. | “An on-call conjunction analyst needs to prioritize incoming messages. We make the evidence and the screening rules visible.” |
| 0:20–0:50 | Select the default test event, open Message evidence, and rotate the RTN encounter plot. | “The selected point is the relative position estimated for closest approach. The colored circles are distance guides, not orbital trajectories.” |
| 0:50–1:15 | Move the message replay slider and show the history chart and rule IDs change with the selected message. Open the assessment PDF. | “These are successive historical estimates. The report records the selected message, rule, data-quality flags and limitations; it does not authorize a maneuver.” |
| 1:15–1:35 | Select **Held-out training events · retrospective** in Dataset. Search `173` and select event `173`; show its latest message and the 24-hour card. | “This message was recorded about 0.077 days from closest approach, with a reported 164 m miss distance. These are separate historical snapshots, not a live synchronized 24-hour feed.” |
| 1:35–2:20 | Open **Model evidence**. Show the probability-driven recall and precision, the distance-only comparison and a visible missed case. | “We evaluate probability-driven critical cases separately, because the distance rule can conceal missed risk estimates. Our evaluation is exploratory; the holdout was previously inspected.” |
| 2:20–2:45 | Show the model comparison, the 95% event-resampled interval, and the latest-event sample count. | “The current model detects 48 of 60 probability-driven critical messages in this selected holdout, but only one qualifying latest-event case exists and it was missed. These are not independent operational results.” |
| 2:45–3:00 | Open a review form or point to the review indicator. | “The analyst makes the decision. We flag questionable inputs, show limitations, and need independent event data before relying on this model.” |

## A concrete missed case

On the current held-out predictions, event `37`, source row `521`, is a probability-driven critical message missed by the point model: the recorded same-message log-risk is about −3.456, the model estimate is about −4.355, and the miss distance is 2,103 m. The challenge rules therefore show the model-estimated tier as Elevated. This is a historical reference label, not proof of an actual collision or a final-event outcome.

To inspect it, select the held-out dataset, search `37`, select event `37`, and use the replay slider to locate source row `521`. The Model evidence page also presents this failure; verify the displayed row if artifacts are regenerated.
