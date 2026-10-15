# T5 — NSW1 / A

T5 — NSW1, A. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.90555089 ± 0.00070606 | 1777.56939492 ± 10.34778885 | 0.22269022 ± 0.00129635 | 0.40000000 ± 0.01743042 |
| adaptive_conformal | post-drift | 0.89769651 ± 0.00019071 | 1753.36562960 ± 4.00834432 | 0.21965802 ± 0.00050216 | 0.40000000 ± 0.01743042 |
| adaptive_conformal | pre-drift | 0.95722222 ± 0.00181089 | 1886.18039122 ± 17.28910878 | 0.23629678 ± 0.00216594 | 0.81666667 ± 0.01743042 |
| static_conformal | drift | 0.97105705 ± 0.00195687 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.44166667 ± 0.01743042 |
| static_conformal | post-drift | 0.96826424 ± 0.00192466 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.36666667 ± 0.01141089 |
| static_conformal | pre-drift | 0.98055556 ± 0.00277778 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.83750000 ± 0.02282177 |

## Complete 24-hour windows

Minimum required per seed/regime: 10. Worst coverage is N/A if any seed has too few windows. Counts are identical for static and adaptive intervals. These overlapping rolling windows are not independent samples.

| Regime | Seed | Complete windows | Worst-window status |
| --- | --- | --- | --- |
| drift | 1 | 13599 | sufficient |
| drift | 2 | 13599 | sufficient |
| drift | 3 | 13599 | sufficient |
| drift | 4 | 13599 | sufficient |
| drift | 5 | 13599 | sufficient |
| post-drift | 1 | 51520 | sufficient |
| post-drift | 2 | 51520 | sufficient |
| post-drift | 3 | 51520 | sufficient |
| post-drift | 4 | 51520 | sufficient |
| post-drift | 5 | 51520 | sufficient |
| pre-drift | 1 | 673 | sufficient |
| pre-drift | 2 | 673 | sufficient |
| pre-drift | 3 | 673 | sufficient |
| pre-drift | 4 | 673 | sufficient |
| pre-drift | 5 | 673 | sufficient |
