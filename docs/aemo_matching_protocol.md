# T2 AEMO matching protocol — aemo_t1_7day_one_to_one_v2

At the user's request, this revision reuses the project's existing T1
seven-day tolerance and first-unused detection rule. The reference is
`run_synthetic_detection_evaluation_absnorm_v2.py` (336 half-hour observations)
and the synthetic matcher in `evaluation.py`. T1 and T2 now use the shared
`_first_unused_in_window` primitive and `DEFAULT_DETECTION_TOLERANCE`.
The JSON beside this file records the rule being reused, rather than defining
a new temporal tolerance. The frozen Tier 1/Tier 2 procedure still applies.

This is an explicit user-requested correction after the earlier v1 evaluation,
not a claim that v2 was selected before any results were seen. Prior outputs
and protocol are retained in `results/tables/detection/archive/aemo_context_v1/`.
Log identities distinguish v2 from that superseded evaluation.

## Frozen inputs

Use results from Task 120,`results/tables/detection/aemo_detection_changepoints.csv`
and the existing `data/events.csv` without editing either file. Preserve all
five Tier 1 IDs specified in `event_tiering_criteria.md` and all 29 Tier 2
events. Record SHA-256 hashes of inputs, protocol, and implementation.
No forecasting or detector rerun is part of T2.

## Time and applicability

Interpret naive SETTLEMENTDATE and catalogue dates in fixed AEST (UTC+10),
as used by the AEMO market-time series, without local daylight-saving shifts.
Catalogue dates have day-level or coarser precision; delays from midnight are
reference-date offsets, not estimates of exact physical onset.
NEM events apply to both SA1 and NSW1; regional events apply only to their
listed region. EVT-2023-14 applies to both evaluated regions.

Matching windows are inclusive [start, end], exactly as in the T1 matcher.
These matching boundaries differ from the half-open F1 regime boundaries.

- `date_precision=day`: match in [start_date, start_date + 7 days].
  In particular, EVT-2022-03 uses the 15 June suspension commencement as
  its point anchor; its 24 June end date is duration context, consistent with
  the frozen catalogue's supporting note.
- `date_precision=date_range` or `month`: reuse T1 gradual matching in
  [start_date, end_date + 7 days]. Catalogue date anchors are at midnight:
  the COVID end anchor is 17 May 00:00, so its final matching instant is
  24 May 00:00. Do not add a day from the F1 regime definition to this anchor.
- Detections before the event start cannot match that event.

## Assignment, delay, and precision

Evaluate each detector and region independently. Sort applicable events by
tier, then start, then event ID. Complete the entire Tier 1 pass before the
Tier 2 pass. For each event, assign the first unused chronological detection
in its inclusive window and remove that detection from the unused set.
Each event and detection can be matched at most once. Remaining detections,
including additional detections in an already matched Tier 1 window, are
then checked against Tier 2 before being classified as unmatched.

Candidate IDs in the audit represent all temporal candidates, even if those
events have already used an earlier detection. Thus a detection can have
candidate IDs and still be unmatched under the one-to-one rule. Exact
duplicate detector/region indices or timestamps are rejected, not dropped.

For each Tier 1 event, delay_days = (earliest assigned detection - event
start) / 24 hours. Keep fractional days in machine-readable metrics; round
only the display table. Use `not detected` if an applicable, covered Tier 1
event has no assigned detection, and `not applicable` for another region.
Reject incomplete Tier 1 coverage instead of silently calling it not detected.

Unmatched = number of detections assigned to neither tier.
Precision = (Tier 1 matched detections + Tier 2 matched detections) /
total detections. With zero detections, precision is undefined (`N/A`).
Under one-to-one matching the matched detection and matched event counts
are equal within each detector-region run. This is a catalogue-match
proportion, not ground-truth classification precision or
proof of causation. AEMO events are contextual historical references;
unmatched detections must not automatically be called false positives.

For reporting, Tier 1 unmatched is the number of detections that do not
match a Tier 1 event. Tier 2 contextual is the subset of those detections
that are subsequently assigned to Tier 2 events. Final unmatched is the
number assigned to neither tier. Therefore:

Tier 1 unmatched = Tier 2 contextual + unmatched.

This reporting preserves Tier 1 as the primary evidence layer while using
Tier 2 as contextual support, consistent with the agreed event-tiering
procedure.

## Regimes and reproducible logging

Reuse the F1 event-relative definitions from
`experiments/degradation/run_baseline_degradation_evaluation.py`:

- pre-drift = [start_date - 7 days, start_date)
- drift = [start_date, end_date + 1 day)
- post-drift = [end_date + 1 day, end_date + 1 day + 7 days)

These descriptive regimes use the full documented duration even for the
suspension point anchor. They are not the matching tolerance windows and
are not a claim of true drift. Log event-window exposure, detection counts,
Tier 1/Tier 2 counts, unmatched counts and catalogue-match proportion through
`run_logger.log_run`. Event windows may overlap; do not add them together to
calculate overall T2 counts.

For reconstructing T2 from `results/tables/runs.csv`, also partition the full test
stream into mutually exclusive accounting regimes: drift if in any applicable
catalogue event duration; otherwise pre-drift if in any event's preceding
7 days; otherwise post-drift if after any applicable event's start; otherwise
pre-drift. Drift takes precedence over overlapping pre/post windows. Thus
quiet time between events is retained. Log these separately using `t2.global.*`
metric names, once per detector/region/regime. Sum their counts to calculate
T2 unmatched and precision. Log each Tier 1 event's applicability and detected
status, plus its delay in the matched detection's event-relative regime.
The event ID is embedded in metric names so the shared logger schema stays
unchanged. Do not log `full_test` rows for this evaluation.

Use the actual Task 120 detector configuration and seed (KSWIN seed 12;
forecaster seed 0 for deterministic detectors), fixed split, training count,
and measured evaluation time. This is evaluation of the stored Task 120 run,
not a new multi-seed experiment. Do not invent standard deviations.

Append T2 metrics to the project's existing central log at
`results/tables/runs.csv`.  A manifest
records the source hashes and protocol, and duplicate run identities are
not appended again. The display table is reconstructed from logged metrics.
The current split ID is `aemo_fixed_train2018_2019_test202003_task120_t2_t1rule_v2`.
Earlier v1 rows remain in the log for traceability; select the manifest's
configuration hashes and split ID when producing the current T2 table.
