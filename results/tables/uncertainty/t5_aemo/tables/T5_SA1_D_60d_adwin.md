# T5 — SA1 / D_60d_adwin

T5 — SA1, D_60d_adwin. Normalised width denominator: 1300.8730537401182 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

Values are mean ± sample standard deviation across seeds 1–5 (ddof=1), not confidence intervals. Worst 24h coverage is computed per seed before aggregation. The minimum of 10 complete overlapping windows is an adjustable reporting threshold, not a statistical reliability guarantee.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89639573 ± 0.00041962 | 990.03086567 ± 2.49081394 | 0.76105110 ± 0.00191472 | 0.37500000 ± 0.00000000 |
| adaptive_conformal | post-drift | 0.89754554 ± 0.00030437 | 911.08827787 ± 2.39519717 | 0.70036678 ± 0.00184122 | 0.38750000 ± 0.01863390 |
| adaptive_conformal | pre-drift | 0.91458333 ± 0.00621736 | 724.81379874 ± 8.63617888 | 0.55717489 ± 0.00663876 | 0.67083333 ± 0.01743042 |
| static_conformal | drift | 0.85181957 ± 0.00660796 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.09166667 ± 0.01863390 |
| static_conformal | post-drift | 0.88632514 ± 0.00604414 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.25416667 ± 0.02282177 |
| static_conformal | pre-drift | 0.96302083 ± 0.01213106 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.80833333 ± 0.06814761 |

## Complete 24-hour windows

Minimum required per seed/regime: 10. Worst coverage is N/A if any seed has too few windows. Counts are identical for static and adaptive intervals. These overlapping rolling windows are not independent samples.

| Regime | Seed | Complete windows | Worst-window status |
| --- | --- | --- | --- |
| drift | 1 | 22099 | sufficient |
| drift | 2 | 22099 | sufficient |
| drift | 3 | 22099 | sufficient |
| drift | 4 | 22099 | sufficient |
| drift | 5 | 22099 | sufficient |
| post-drift | 1 | 43121 | sufficient |
| post-drift | 2 | 43121 | sufficient |
| post-drift | 3 | 43121 | sufficient |
| post-drift | 4 | 43121 | sufficient |
| post-drift | 5 | 43121 | sufficient |
| pre-drift | 1 | 337 | sufficient |
| pre-drift | 2 | 337 | sufficient |
| pre-drift | 3 | 337 | sufficient |
| pre-drift | 4 | 337 | sufficient |
| pre-drift | 5 | 337 | sufficient |
