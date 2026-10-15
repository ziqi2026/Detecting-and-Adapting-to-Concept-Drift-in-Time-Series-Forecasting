# T5 — NSW1 / D_180d_page_hinkley

T5 — NSW1, D_180d_page_hinkley. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.90569072 ± 0.00038292 | 1972.87694981 ± 5.54901971 | 0.24715795 ± 0.00069517 | 0.33333333 ± 0.00000000 |
| adaptive_conformal | post-drift | 0.89750120 ± 0.00010575 | 2019.56452460 ± 2.75539438 | 0.25300687 ± 0.00034519 | 0.35416667 ± 0.00000000 |
| adaptive_conformal | pre-drift | 0.95722222 ± 0.00181089 | 1886.18039122 ± 17.28910878 | 0.23629678 ± 0.00216594 | 0.81666667 ± 0.01743042 |
| static_conformal | drift | 0.95446029 ± 0.00218888 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.26666667 ± 0.00931695 |
| static_conformal | post-drift | 0.94591096 ± 0.00255129 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.35416667 ± 0.00000000 |
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
