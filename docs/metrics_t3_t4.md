# T3 and T4 metric definitions

How the numbers in **T3 (forecast accuracy by regime)** and **T4 (adaptation
cost)** are computed. Every definition below points at the code that
implements it; if the two ever disagree, the code is authoritative and this
page is wrong.

| What | Code |
| --- | --- |
| MAE, pinball loss, adaptation gain, regime assignment | [`src/drift_lab/evaluation.py`](../src/drift_lab/evaluation.py) |
| T3 generator | [`experiments/adaptation/generate_t3_forecast_accuracy.py`](../experiments/adaptation/generate_t3_forecast_accuracy.py) |
| T4 generator | [`experiments/adaptation/generate_t4_adaptation_cost.py`](../experiments/adaptation/generate_t4_adaptation_cost.py) |
| Retrain counts, training samples, wall-clock timing | [`src/drift_lab/adaptation.py`](../src/drift_lab/adaptation.py) (`adaptive_forecast_stream`) |
| Four-arm runner (`--forecaster`) | [`experiments/adaptation/run_four_arm_adaptation.py`](../experiments/adaptation/run_four_arm_adaptation.py) |
| Cross-forecaster summary | [`experiments/adaptation/generate_cross_forecaster_summary.py`](../experiments/adaptation/generate_cross_forecaster_summary.py) |
| Seed-level numbers in the shared log | [`experiments/adaptation/log_t3_t4_to_runs.py`](../experiments/adaptation/log_t3_t4_to_runs.py) → `results/runs.csv` |

Setting: regions SA1 and NSW1, four adaptation arms (A never retrain, B retrain
every 30 days, C retrain on drift with full history, D retrain on drift on a
recent 60/120/180-day window — Arms C and D use ADWIN, Page-Hinkley or KSWIN,
giving 17 configurations), forecasting seeds 1–5. Nothing in T3/T4 re-runs a
forecaster: both tables are computed from the saved forecast streams
(`results/tables/adaptation/revised/pgs242_128_five_seed_final/seed_*/`).

Notation: for one configuration, region and seed, the test stream has
half-hourly observations `y_t` with point forecast `ŷ_t`, `t = 1 … N`
(N = 67,249 per region).

## Aggregation over seeds

Every metric is first computed **per seed** (one forecaster run per seed), then
reported as `mean ± sd` over the 5 seeds, where `sd` is the **sample** standard
deviation (`ddof = 1`). The `sd` therefore measures seed-to-seed variability of
the model fit, not uncertainty in the data. Where a metric is derived from a
comparison (gain, gain per retrain) the comparison is made inside each seed
and only the result is aggregated.

## MAE

```
MAE(S) = (1 / |S|) · Σ_{t ∈ S}  | y_t − ŷ_t |
```

`S` is the set of test timestamps being scored — one regime (T3), or the drift
regime (T4 gain). Implemented by `calculate_mae`. MAE is in MW.

T3 reports `MAE pre-drift`, `MAE drift` and `MAE post-drift`, i.e. `S` = the
timestamps labelled with that regime (see [Regime windows](#regime-windows)).

## Pinball loss

For a single quantile level `q ∈ (0, 1)`, the pinball (quantile) loss of a
quantile forecast `ŷ` for outcome `y` is

```
L_q(y, ŷ) = max( q·(y − ŷ),  (q − 1)·(y − ŷ) )
```

(`calculate_pinball_loss` takes the mean of `L_q` over timestamps.)

T3's `pinball loss` scores the 90 % prediction interval (`alpha = 0.10`) as
implemented in `calculate_interval_pinball_loss`: the lower bound is scored as
the `q = alpha/2 = 0.05` quantile, the upper bound as the `q = 1 − alpha/2 =
0.95` quantile, and the two mean losses are averaged:

```
pinball = ½ · [ mean_t L_0.05(y_t, lower_t)  +  mean_t L_0.95(y_t, upper_t) ]
```

- The interval is `[ŷ_t − r, ŷ_t + r]` with `r` the fixed, seed- and
  region-specific conformal radius from the PGS242-141 calibration
  (`seed_<n>/uncertainty/aemo_prediction_intervals_90_summary.csv`), applied to
  the point forecasts of every adaptation arm. The radius is not re-calibrated
  after retraining.
- Pinball loss is computed over **all** test timestamps (full-stream scope),
  not per regime. It is therefore identical under both buffer modes.
- Lower is better; units are MW.

## Adaptation gain (T4)

```
MAE gain vs arm A  =  MAE_drift(arm)  −  MAE_drift(arm A)        (same region, same seed)
```

`MAE_drift` is the drift-regime MAE of T3 (full-stream scope, the buffer mode
being reported). **Negative = better than never retraining.** Implemented by
`calculate_adaptation_gain`; the generator pairs each arm with Arm A on
`(region, seed)` before differencing, so Arm A's own gain is exactly 0.

Worked example (from the supervisor's output specification): Arm D with drift
MAE 54.3, Arm A with 88.3 and 2 retrains gives a gain of −34.0 and −17.0 per
retrain (covered by `tests/test_adaptation_gain.py`).

## Gain per retrain (T4)

```
gain per retrain  =  MAE gain vs arm A  /  n_retrains            (n_retrains > 0)
```

Computed per seed, then averaged (`mean ± sd` over seeds). It is undefined
(`NaN`, shown as `—`) for Arm A, which never retrains. It is not the mean gain
divided by the mean retrain count; because retrain counts are identical across
seeds for a given configuration (the change points are frozen), the two
coincide in practice, but the generator uses the per-seed definition.

The ratio rewards cheap, modest improvements and, for arms that are *worse* than
A, turns into a large positive number (a large loss per retrain). Read it
together with the gain itself.

## Cost columns (T4)

| Column | Definition |
| --- | --- |
| `retrains` | Number of model refits during the test stream (`n_retrains`). B: every 30 days; C/D: one per accepted drift alarm. |
| `cumulative train samples` | Initial training rows plus the rows used by every refit (`train_samples_total`). Full-history arms re-train on everything seen so far; D re-trains on the last 60/120/180 days only. |
| `wall clock (s)` | `retraining_wall_clock_s`, see below. |

### How wall-clock is measured

`wall clock (s)` is the **sum of the model-fitting time of the refits**. In
`adaptive_forecast_stream` ([`src/drift_lab/adaptation.py`](../src/drift_lab/adaptation.py)),
for each retrain event the timer wraps exactly one statement:

```python
started = perf_counter()
updated_model.fit(training)          # <- timed
fit_s = perf_counter() - started
```

and `retraining_wall_clock_s = Σ fit_s` over the arm's retrain events.

Not included:

- the initial fit on the training split (`initial_fit_s`; measured once per
  region and shared by all arms, so it is not an adaptation cost);
- `deepcopy` of the previous model (`model_copy_s`, recorded separately);
- selecting the training window, forecasting/inference, and drift detection
  (the detectors run upstream; their alarms are frozen inputs).

The runner also records `arm_wall_clock_s` (the whole `adaptive_forecast_stream`
call, including forecasting). It is *not* used in T4 and is what the shared log
stores in its `wall_clock_s` column; the `retraining_wall_clock_s` value is the
one logged as a metric. Timings come from `time.perf_counter` on whichever
machine produced the saved runs, so they are comparable between arms and seeds
of this study but not across hardware. Arm A is 0 by definition.

## Forecasters

T3/T4 were first produced for `gradient_boosted` only. They now also exist for
`seasonal_naive`, so a statement like "model X is better" can be checked against
the other forecaster. Both use the same change points, 30-day schedule, D windows
(30/60/120/180 days, so the same 17 configurations), refit-data rules (no
calibration or future rows in any refit) and seeds 1–5.

| Forecaster | Four-arm run | T3 / T4 tables | `forecaster` column |
| --- | --- | --- | --- |
| `gradient_boosted` | `revised/pgs242_128_five_seed_final/` | `t3/<mode>/`, `t4/<mode>/` | no (original schema, unchanged) |
| `seasonal_naive` | `revised/seasonal_naive_five_seed/` | `t3/seasonal_naive/<mode>/`, `t4/seasonal_naive/<mode>/` | yes, first column |

`--forecaster` (default `gradient_boosted`) is accepted by the runner and by both
generators. The default writes the original files in the original schema, so the
gradient_boosted tables are byte-identical to before; any other forecaster gets its
own folder and a leading `forecaster` column in its tables, in both buffer modes.

### What "retraining" means for seasonal-naive

`SeasonalNaiveForecaster` (`season = 48`, i.e. the same half-hour one day earlier):

- `fit(train)` only checks that `train` is not empty and returns `self`. It stores
  nothing, so there is no seasonal reference to refresh from the training window.
- `predict(data, history_df)` returns `TOTALDEMAND` shifted by 48 rows, taking the
  first 48 values from the last 48 rows of `history_df`. It reads only observations
  that are already known (`adaptive_forecast_stream` passes every row before the
  segment as history), never anything fitted.

A retrain therefore changes no forecast: arms B, C and D produce exactly the same
forecasts as arm A. Consequences for the tables:

- `retrains` and `cumulative train samples` are identical to gradient_boosted's
  (same schedule, same refit-data selection) but only describe what was
  *scheduled*; the refit itself does nothing and `wall clock (s)` is the cost of an
  empty `fit()` (~1e-5 s).
- `MAE gain vs arm A` and `gain per retrain` are exactly 0.
- The model is deterministic, so the five seeds are identical and every `sd` is
  0.0. That is not seed variation to be interpreted: the seed only labels the run
  (`manifest.json` has `seed_affects_model: false`). The generators write this into
  the caption whenever all seeds agree.

### Pinball loss

T3's pinball loss needs a conformal radius. For `gradient_boosted` it comes from the
PGS242-141 runner. For `seasonal_naive` the radius did not exist, so the four-arm
runner calibrates it with the same procedure
(`conformal_radius_summary`; `experiments/uncertainty/run_aemo_uncertainty.py`): fit
on train, absolute residuals on the calibration frame with train as lag history,
`conformal_radius` at `alpha = 0.10`. It uses calibration data only and is written to
`seed_<n>/uncertainty/aemo_prediction_intervals_90_summary.csv`, where T3 reads it.
Because the radius depends on the point forecasts, pinball losses are comparable
within a forecaster but the intervals differ between forecasters.

### Cross-forecaster summary

`results/tables/adaptation/cross_forecaster/cross_forecaster_summary_<mode>.{csv,md}`
has one row per forecaster, region and arm (the same headline configuration per arm as
T3/T4): MAE pre-drift / drift / post-drift, retrains and the drift-regime gain vs
arm A, as `mean ± sd` over seeds. `post_only` is the default; `symmetric` is also
written. It is computed from the seed-level T3/T4 files of both forecasters.

### Regenerate

The forecast streams (`forecasts_<region>.csv.gz`, ~31 MB each, five identical copies)
are not tracked. Re-create them (about 2.5 minutes) and then the tables:

```bash
python experiments/adaptation/run_four_arm_adaptation.py --forecaster seasonal_naive \
    --run-id seasonal_naive_five_seed \
    --detections results/tables/detection/aemo_detection_changepoints.csv \
    --decisions docs/decisions.md
for mode in post_only symmetric; do
  python experiments/adaptation/generate_t3_forecast_accuracy.py --forecaster seasonal_naive --buffer-mode $mode --scopes full
  python experiments/adaptation/generate_t4_adaptation_cost.py --forecaster seasonal_naive --buffer-mode $mode
  python experiments/adaptation/generate_cross_forecaster_summary.py --buffer-mode $mode
done
```

The runner refuses to overwrite an existing run directory.

## Regime windows

The documented events are in `data/events.csv` (NEM-wide events apply to both
regions; SA1/NSW1 events only to their own region; all tiers are used). Each
event has a first and last day (`start_date`, `end_date`, both inclusive), so
the event itself is `[start_date, end_date + 1 day)`. The buffer length is
**7 days** (`--regime-days 7`). All intervals are left-closed, right-open and
evaluated in `Etc/GMT-10` (AEST, no daylight saving, matching the AEMO
timestamps).

The full-stream T3 labels every test timestamp. The buffer mode only changes
where the drift window **starts**; the rest is shared:

| | `post_only` (default) | `symmetric` |
| --- | --- | --- |
| Drift window per event | `[start, end + 1 d + 7 d)` | `[start − 7 d, end + 1 d + 7 d)` |
| Origin | Buffer after the event only (supervisor, 20 Oct). | Earlier ±7-day convention, kept so both versions can be shown and labelled. |
| Code | `buffer_mode="post_only"` in `assign_full_stream_forecast_regimes` | `buffer_mode="symmetric"` |

Common to both modes:

- **pre-drift** — every timestamp before the first drift window;
- **drift** — every timestamp inside the **union** of the drift windows of the
  events whose window overlaps the evaluated stream (overlapping windows merge;
  a second event inside the first event's post-buffer extends the drift
  period instead of creating a post-drift gap);
- **post-drift** — every other timestamp after the first drift window.

There is no "unassigned" regime and no excluded data: pre + drift + post = N.
Because the stream starts on 2020-03-01 and the first applicable event is the
COVID-19 demand change on 2020-03-16, pre-drift is a single block at the start
of the stream. Observations per regime (identical for every configuration):

| Region | Mode | pre-drift | drift | post-drift |
| --- | --- | ---: | ---: | ---: |
| SA1 | `post_only` | 720 | 16,609 | 49,920 |
| SA1 | `symmetric` | 384 | 22,945 | 43,920 |
| NSW1 | `post_only` | 720 | 14,304 | 52,225 |
| NSW1 | `symmetric` | 384 | 19,344 | 47,521 |

(`symmetric` has 336 fewer pre-drift observations — exactly the 7 days before
the first event — and moves the 7 days before every later event from
pre-/post-drift into drift, hence its larger drift counts.)

The earlier **event-local** T3 (`--scopes local`, `t3/t3_local.*`) uses its own
fixed windows — 7 days pre, the event as drift, 7 days post, everything else
excluded — and does not depend on `--buffer-mode`.

### Files

| Mode | T3 | T4 |
| --- | --- | --- |
| `post_only` | `results/tables/adaptation/t3/post_only/t3_full_post_only.{csv,md}` | `results/tables/adaptation/t4/post_only/t4_{representative_per_arm,all_configurations}_post_only.{csv,md}` |
| `symmetric` | `results/tables/adaptation/t3/symmetric/t3_full_symmetric.{csv,md}` | `results/tables/adaptation/t4/symmetric/t4_{representative_per_arm,all_configurations}_symmetric.{csv,md}` |

The un-suffixed files directly under `t3/` and `t4/` (read by the dashboard) are
the `post_only` results: the T3 headline table and the T4 representative table
are byte-identical to the `post_only` files above, the seed-level metrics agree
to ~1e-13 (floating point), and `t4_all_configurations.csv` has the same rows in
a different order. The seed-level numbers for both modes are also in
`results/runs.csv` under the groups `t3_forecast_accuracy_{post_only,symmetric}`
and `t4_adaptation_cost_{post_only,symmetric}`.

The T3 generator also writes a per-timestamp `regime_assignment_full_<mode>.csv`
(one row per half-hour and region) next to each table. These audit files are large
and are not tracked; they are recreated by the commands below.

Regenerate (nothing is re-trained):

```bash
python experiments/adaptation/generate_t3_forecast_accuracy.py --buffer-mode post_only --scopes full
python experiments/adaptation/generate_t3_forecast_accuracy.py --buffer-mode symmetric --scopes full
python experiments/adaptation/generate_t4_adaptation_cost.py --buffer-mode post_only
python experiments/adaptation/generate_t4_adaptation_cost.py --buffer-mode symmetric
python experiments/adaptation/log_t3_t4_to_runs.py --replace   # refresh results/runs.csv
```
