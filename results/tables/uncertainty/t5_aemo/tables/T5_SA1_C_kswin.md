# T5 — SA1 / C_kswin

T5 — SA1, C_kswin. Normalised width denominator: 1300.8730537401182 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89944650 ± 0.00037415 | 682.88256743 ± 2.74454928 | 0.52494174 ± 0.00210977 | 0.44166667 ± 0.03726780 |
| adaptive_conformal | post-drift | 0.90001366 ± 0.00021152 | 632.74582920 ± 3.21213291 | 0.48640090 ± 0.00246921 | 0.35833333 ± 0.01743042 |
| adaptive_conformal | pre-drift | 0.91458333 ± 0.00621736 | 724.81379874 ± 8.63617888 | 0.55717489 ± 0.00663876 | 0.67083333 ± 0.01743042 |
| static_conformal | drift | 0.94417084 ± 0.00425013 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.34166667 ± 0.01141089 |
| static_conformal | post-drift | 0.95311475 ± 0.00340931 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.30416667 ± 0.02795085 |
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
