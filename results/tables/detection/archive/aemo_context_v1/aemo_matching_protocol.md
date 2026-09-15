# T2 AEMO matching protocol — aemo_context_v1

This protocol is specified for this T2 evaluation before calculating matches
from Task 120. It adds the temporal and accounting rules left unspecified by
`event_tiering_criteria.md`; it is not a claim that these temporal rules were
already present in the repository. The JSON beside this file is the executable
configuration. Do not tune it after inspecting matches.

## Frozen inputs

Use `results/tables/detection/aemo_detection_changepoints.csv` from Task 120,
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

All windows are [start, end): the start is included and the end excluded.

- `date_precision=day`: match in [start_date, start_date + 7 days).
  In particular, EVT-2022-03 uses the 15 June suspension commencement as
  its point anchor; its 24 June end date is duration context, consistent with
  the frozen catalogue's supporting note.
- `date_precision=date_range` or `month`: match in [start_date,
  end_date + 1 day + 7 days), including the full documented last day.
- Detections before the event start cannot match that event.

## Assignment, delay, and precision

Evaluate each detector and region independently. Every detection is checked
against all applicable Tier 1 windows first, then against Tier 2 windows if
no Tier 1 window contains it. Resolve overlaps within a tier by earliest
event start, then lexicographic event ID. Retain the candidate IDs in the
audit output so overlaps remain visible. A detection has at most one assigned
event; an event may explain multiple detections. Assignments never consume
an event or change its tier. Exact duplicate detector/region indices or
timestamps are rejected, not silently dropped.

For each Tier 1 event, delay_days = (earliest assigned detection - event
start) / 24 hours. Keep fractional days in machine-readable metrics; round
only the display table. Use `not detected` if an applicable, covered Tier 1
event has no assigned detection, and `not applicable` for another region.
Reject incomplete Tier 1 coverage instead of silently calling it not detected.

Unmatched = number of detections assigned to neither tier.
Precision = (Tier 1 matched detections + Tier 2 matched detections) /
total detections. With zero detections, precision is undefined (`N/A`).
Count matched detections, not unique events, in the numerator. This is a
catalogue-match proportion, not ground-truth classification precision or
proof of causation. AEMO events are contextual historical references;
unmatched detections must not automatically be called false positives.

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

For reconstructing T2 from `results/runs.csv`, also partition the full test
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

Append to `results/runs.csv` as explicitly required by T2. The older
`results/tables/runs.csv` remains a separate historical source. A manifest
records the source hashes and protocol, and duplicate run identities are
not appended again. The display table is reconstructed from logged metrics.
