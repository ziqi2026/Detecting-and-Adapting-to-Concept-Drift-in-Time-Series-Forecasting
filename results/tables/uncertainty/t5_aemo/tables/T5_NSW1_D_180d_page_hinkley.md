# T5 — NSW1 / D_180d_page_hinkley

T5 — NSW1, D_180d_page_hinkley. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.90089950 ± 0.00018277 | 2030.76309232 ± 5.00677870 | 0.25440981 ± 0.00062724 | 0.33333333 ± 0.00000000 |
| adaptive_conformal | post-drift | 0.89905936 ± 0.00007645 | 1996.13833617 ± 3.50416386 | 0.25007209 ± 0.00043899 | 0.35416667 ± 0.00000000 |
| adaptive_conformal | pre-drift | 0.95052083 ± 0.00184142 | 2365.28108870 ± 32.85174269 | 0.29631753 ± 0.00411560 | 0.82500000 ± 0.01141089 |
| static_conformal | drift | 0.94471671 ± 0.00250513 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.26666667 ± 0.00931695 |
| static_conformal | post-drift | 0.94935292 ± 0.00244290 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.35416667 ± 0.00000000 |
| static_conformal | pre-drift | 0.96354167 ± 0.00520833 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.83750000 ± 0.02282177 |

## Complete 24-hour windows

Minimum required per seed/regime: 10. Worst coverage is N/A if any seed has too few windows. Counts are identical for static and adaptive intervals. These overlapping rolling windows are not independent samples.

| Regime | Seed | Complete windows | Worst-window status |
| --- | --- | --- | --- |
| drift | 1 | 18639 | sufficient |
| drift | 2 | 18639 | sufficient |
| drift | 3 | 18639 | sufficient |
| drift | 4 | 18639 | sufficient |
| drift | 5 | 18639 | sufficient |
| post-drift | 1 | 46816 | sufficient |
| post-drift | 2 | 46816 | sufficient |
| post-drift | 3 | 46816 | sufficient |
| post-drift | 4 | 46816 | sufficient |
| post-drift | 5 | 46816 | sufficient |
| pre-drift | 1 | 337 | sufficient |
| pre-drift | 2 | 337 | sufficient |
| pre-drift | 3 | 337 | sufficient |
| pre-drift | 4 | 337 | sufficient |
| pre-drift | 5 | 337 | sufficient |
