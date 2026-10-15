# T5 — NSW1 / C_kswin

T5 — NSW1, C_kswin. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.90285235 ± 0.00040343 | 1604.82498509 ± 6.96703915 | 0.20104916 ± 0.00087282 | 0.40000000 ± 0.01743042 |
| adaptive_conformal | post-drift | 0.89861561 ± 0.00009612 | 1624.79948209 ± 4.28823116 | 0.20355152 ± 0.00053722 | 0.40000000 ± 0.00931695 |
| adaptive_conformal | pre-drift | 0.95722222 ± 0.00181089 | 1886.18039122 ± 17.28910878 | 0.23629678 ± 0.00216594 | 0.81666667 ± 0.01743042 |
| static_conformal | drift | 0.97983781 ± 0.00185072 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.43333333 ± 0.00931695 |
| static_conformal | post-drift | 0.97539110 ± 0.00144876 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.35833333 ± 0.00931695 |
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
