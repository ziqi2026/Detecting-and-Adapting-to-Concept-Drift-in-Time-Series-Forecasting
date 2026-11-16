# Streamlit Dashboard and Results Visualisation

[Project README](../README.md) · [Public dashboard](https://detecting-and-adapting-to-concept-drift-in-time-series-forecas.streamlit.app/) · [Uncertainty and escalation README](../experiments/uncertainty/README.md)

## Purpose and scope

`dashboard/app.py` is a single-file Streamlit app, **Concept Drift Lab — Team 42**. It presents the stored results of every project stage (F1 to F6, T1 to T7) in nine pages, with interactive filters and CSV downloads.

The dashboard **only reads saved files** under `results/`. It does not fit forecasters, run detectors, rebuild intervals or retrain anything when you change a control. The "live" replay in stage 09 is a replay of recorded test-period outputs, not an AEMO feed.

## Files

| File | Responsibility |
| --- | --- |
| [app.py](app.py) | The whole app: page routing, loaders, charts, tables and styling (about 2,000 lines) |
| [requirements.txt](requirements.txt) | `streamlit`, `pandas`, `plotly` (not version-pinned) |

## Run locally

From the repository root:

```bash
python -m pip install -r dashboard/requirements.txt
git lfs pull                     # needed for the interval streams used by stages 05, 07 and 09
streamlit run dashboard/app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`).

The app finds the repository root by walking up from `app.py` until it finds a `results/` folder, so it also works when the file is copied into a lighter deployment repository with the same `results/` layout.

## The nine stages

The sidebar radio **Explore the research** switches between stages. Names match `STAGES` in `app.py`.

| Stage | Outputs | Main controls | Render functions |
| --- | --- | --- | --- |
| **01 · Project Overview** | Research questions, regions, split timeline, methods, four arms | — | inline |
| **02 · Ethical Considerations** | Ethics, security and human-oversight discussion | — | inline |
| **03 · Forecast Degradation** | F1 rolling MAE figures and tables | Region, figure type | `render_0` |
| **04 · Drift Detection** | T1 synthetic benchmark; T2 event matching and F2 panels | Evidence view, region, detector | `render_1`, `render_2` |
| **05 · Forecast Adaptation** | Interactive forecasts, F3, T3, T4, model selection | View, region, detector (default Page-Hinkley), Full/Local scope, seed, Arm D window | `render_6`, `render_5`, `render_3`, `render_4`, `render_model_selection` |
| **06 · Uncertainty & Reliability** | T5 three-method tables; T5 simulation browser; F4 trajectories | Region, T5 dataset, configuration, detector | `render_7`, `render_f4` |
| **07 · Human Escalation** | Five-day escalation explorer; F5 curves; T6 operating points | View, region, seed, UQ method, configuration, review signal, threshold | `render_escalation_explorer`, `render_f5`, `render_t6` |
| **08 · Trigger Comparison** | T7 Team 41 vs Team 42 table; F6 alarm timeline | Region for F6 | `render_t7`, `render_f6` |
| **09 · Integrated System & Research Synthesis** | Historical replay of demand, forecast, interval, alarms, retrains and review flags; figure and table gallery | Seed, region, UQ method, configuration, detector overlay, review signal, replay date | `render_integrated_prototype`, `render_8` |

### Five-day windows (stages 07 and 09)

Both the escalation explorer and the integrated replay show a five-day slice of the TEST stream. Review scores are computed on the **full stream before slicing**, so the first step of a window still uses the real previous observation. The two review signals match the escalation study:

- **Previous 30-minute absolute error**: `|y_(t−1) − ŷ_(t−1)|`
- **Current interval width**: `upper_t − lower_t`

In stage 09 the threshold is a percentile slider over the selected stream's scores. The thresholds in these views are for exploration. They are not the tuned T6 operating points, and no reviewer correction is simulated.

## Data each stage reads

All paths are relative to `results/`. A missing file shows a warning or an info box in that section; the rest of the page still renders.

| Stage | Files |
| --- | --- |
| 03 | `figures/degradation/F1_<region>_*.png`; `tables/degradation/F1_rolling_mae.csv`, `F1_rolling_mae_baseline.csv`, `patchtst_synthetic_degradation_results.csv` |
| 04 | `tables/detection/T1_detection_on_synthetic.csv`, `T2_aemo_detection.csv`, `T2_detection_summary.csv`, `T2_detection_matches.csv`, `T2_tier1_event_details.csv`; `figures/detection/` |
| 05 | `tables/adaptation/t3/`, `tables/adaptation/t4/t4_seed_level_metrics.csv`, `t4_adaptation_cost.csv`; `tables/adaptation/revised/pgs242_128_five_seed_final/summary_all_seeds.csv`; seed-level `forecasts_<region>.csv.gz`; `figures/adaptation/F3_four_arms/`; `model_selection/` |
| 06 | `tables/uncertainty/t5_agaci_full_post7/T5_<region>_<config>.csv` (any folder containing `T5_SA1_*.csv` is offered); `tables/uncertainty/t5_simulation/`; `figures/uncertainty/f4/` |
| 07 | **`tables/uncertainty/agaci_boa_fixed_calibration/seed_<n>/agaci_comparison_intervals_<region>.csv.gz` (Git LFS)**; `figures/uncertainty/f5_width/`, `f5_nonconformity/`; `tables/uncertainty/t6_*/` |
| 08 | `tables/adaptation/t4/t4_seed_level_metrics.csv`; T5 and T6 folders above; `tables/detection/aemo_detection_changepoints.csv`; Team 41 alarms from `tables/cross_team/page_hinkley_standard_half_hourly_post_tune_<region>.csv`, falling back to `team41_accepted_detections.csv` |
| 09 | The Git LFS interval streams above; `tables/detection/aemo_detection_changepoints.csv`; seed-level `retraining_events.csv` from the adaptation run; every PNG under `figures/` and CSV under `tables/` for the gallery |

## How T7 and F6 are built

T7 and F6 are not separate committed tables. The dashboard builds them when the page loads:

- **T7** compares the two teams on Arm D, 180-day window, Page-Hinkley (`D_180d_page_hinkley`), for both regions. Team 42 values are read from the T4, T5 and T6 files above. **Team 41 values are typed into `render_t7()`**, transcribed from Team 41's shared T7 summary. If Team 41 updates their numbers, `app.py` must be edited by hand. A download button exports the assembled table.
- **F6** plots both teams' accepted Page-Hinkley alarm times on one timeline per region, from the two detection files listed for stage 08.

The comparison table on the page lists the differences that remain between the teams (forecaster implementation, AgACI implementation), so read T7 as a like-for-like trigger comparison only where the table says the settings match.

## Deployment

The public app runs on Streamlit Community Cloud from a **separate, lightweight deployment repository**. An earlier deployment from this research repository hit resource limits because of the size of the result data. Large uncertainty artifacts (the interval streams) are published as **GitHub Release** assets for the deployment, rather than committed there.

To update the public app after results change:

1. Copy the changed `dashboard/app.py` and the affected `results/` files into the deployment repository, keeping the same relative paths.
2. If interval streams changed, upload the new `.csv.gz` files to a new Release in the deployment repository.
3. Push; Streamlit Cloud redeploys on push to its tracked branch.

The local app and the public app are related but separate. Check the public app after each deployment, especially stages 07 and 09, which depend on the large files.

## Checking the app after a change

A headless smoke test with Streamlit's `AppTest` opens every stage and reports exceptions, warnings and errors:

```python
from streamlit.testing.v1 import AppTest

for stage in ['01 · Project Overview', '02 · Ethical Considerations', '03 · Forecast Degradation',
              '04 · Drift Detection', '05 · Forecast Adaptation', '06 · Uncertainty & Reliability',
              '07 · Human Escalation', '08 · Trigger Comparison',
              '09 · Integrated System & Research Synthesis']:
    at = AppTest.from_file('dashboard/app.py', default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(stage).run()
    print(stage, len(at.exception), [w.value[:80] for w in at.warning])
```

On 15 Nov 2026 (Streamlit 1.67), all nine stages ran with **zero exceptions**. In a checkout without the Git LFS streams and the large F1/forecast CSVs, stages 03, 05, 07 and 09 show "file unavailable" warnings for exactly those files, as designed.

## Known issues

- `FINAL` at the top of `app.py` points to `results/tables/adaptation/pgs242_128_five_seed_final`, but the five-seed run is stored under `results/tables/adaptation/revised/pgs242_128_five_seed_final`. The interactive-forecast loader falls back to a recursive search, so it still works, but the constant is wrong.
- Streamlit 1.67 logs a deprecation warning for `use_container_width`, which is removed after 2025-12-31 in Streamlit's schedule. Replace it with `width='stretch'` before upgrading further.
- `requirements.txt` is not pinned. A future Streamlit or pandas release could change behaviour; pin the versions used by the deployed app if it needs to stay stable.
- Team 41's T7 values are hard-coded (see above).

## Limitations

- The dashboard shows historical SA1 and NSW1 results only. It is a research demonstration, not an operational forecasting or alerting tool.
- Event markers and regimes come from an incomplete catalogue. An alarm with no matching event is not automatically a false alarm.
- Review thresholds and flags in stages 07 and 09 are exploratory. Coverage shown there is for the forecasts kept, not for a system after human correction.
