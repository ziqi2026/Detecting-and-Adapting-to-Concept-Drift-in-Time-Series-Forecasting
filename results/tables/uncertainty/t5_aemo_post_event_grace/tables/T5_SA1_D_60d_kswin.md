# T5 — SA1 / D_60d_kswin

T5 — SA1, D_60d_kswin. Normalised width denominator: 1300.8730537401182 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89975315 ± 0.00044244 | 1003.47061401 ± 2.25854286 | 0.77138243 ± 0.00173617 | 0.40000000 ± 0.00931695 |
| adaptive_conformal | post-drift | 0.89855369 ± 0.00011009 | 953.69809976 ± 2.33512489 | 0.73312157 ± 0.00179504 | 0.42916667 ± 0.01141089 |
| adaptive_conformal | pre-drift | 0.91222222 ± 0.00359505 | 745.69748917 ± 28.58103473 | 0.57322848 ± 0.02197066 | 0.66250000 ± 0.01743042 |
| static_conformal | drift | 0.85037028 ± 0.00628210 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.22916667 ± 0.00000000 |
| static_conformal | post-drift | 0.86310497 ± 0.00674477 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.00000000 ± 0.00000000 |
| static_conformal | pre-drift | 0.95111111 ± 0.01311747 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.71666667 ± 0.03784563 |

## Complete 24-hour windows

Minimum required per seed/regime: 10. Worst coverage is N/A if any seed has too few windows. Counts are identical for static and adaptive intervals. These overlapping rolling windows are not independent samples.

| Regime | Seed | Complete windows | Worst-window status |
| --- | --- | --- | --- |
| drift | 1 | 15716 | sufficient |
| drift | 2 | 15716 | sufficient |
| drift | 3 | 15716 | sufficient |
| drift | 4 | 15716 | sufficient |
| drift | 5 | 15716 | sufficient |
| post-drift | 1 | 49074 | sufficient |
| post-drift | 2 | 49074 | sufficient |
| post-drift | 3 | 49074 | sufficient |
| post-drift | 4 | 49074 | sufficient |
| post-drift | 5 | 49074 | sufficient |
| pre-drift | 1 | 673 | sufficient |
| pre-drift | 2 | 673 | sufficient |
| pre-drift | 3 | 673 | sufficient |
| pre-drift | 4 | 673 | sufficient |
| pre-drift | 5 | 673 | sufficient |
