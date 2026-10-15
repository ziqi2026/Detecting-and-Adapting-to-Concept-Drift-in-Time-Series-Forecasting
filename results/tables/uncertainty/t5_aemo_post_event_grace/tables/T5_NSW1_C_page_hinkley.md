# T5 — NSW1 / C_page_hinkley

T5 — NSW1, C_page_hinkley. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.90346756 ± 0.00052176 | 1601.77414822 ± 7.18690553 | 0.20066696 ± 0.00090036 | 0.39166667 ± 0.02282177 |
| adaptive_conformal | post-drift | 0.89842030 ± 0.00016304 | 1627.26535803 ± 4.28221210 | 0.20386044 ± 0.00053647 | 0.39583333 ± 0.01473139 |
| adaptive_conformal | pre-drift | 0.95722222 ± 0.00181089 | 1886.18039122 ± 17.28910878 | 0.23629678 ± 0.00216594 | 0.81666667 ± 0.01743042 |
| static_conformal | drift | 0.97954418 ± 0.00142470 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.43750000 ± 0.00000000 |
| static_conformal | post-drift | 0.97534897 ± 0.00212812 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.36666667 ± 0.01141089 |
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
