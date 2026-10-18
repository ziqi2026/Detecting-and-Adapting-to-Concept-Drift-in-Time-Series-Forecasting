# T5 — SA1 / B

T5 — SA1, B. Normalised width denominator: 1300.8730537401182 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89908913 ± 0.00026972 | 673.11646037 ± 3.41381074 | 0.51743439 ± 0.00262425 | 0.45833333 ± 0.00000000 |
| adaptive_conformal | post-drift | 0.90017304 ± 0.00019758 | 631.47947523 ± 1.50572608 | 0.48542744 ± 0.00115747 | 0.38333333 ± 0.04564355 |
| adaptive_conformal | pre-drift | 0.91458333 ± 0.00621736 | 724.81379874 ± 8.63617888 | 0.55717489 ± 0.00663876 | 0.67083333 ± 0.01743042 |
| static_conformal | drift | 0.94547832 ± 0.00424085 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.33750000 ± 0.03423266 |
| static_conformal | post-drift | 0.95235883 ± 0.00380569 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.28333333 ± 0.03159531 |
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
