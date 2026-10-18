# T5 — NSW1 / D_180d_adwin

T5 — NSW1, D_180d_adwin. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89827337 ± 0.00025431 | 2551.65230845 ± 4.15432384 | 0.31966573 ± 0.00052045 | 0.44583333 ± 0.01863390 |
| adaptive_conformal | post-drift | 0.90019149 ± 0.00007136 | 2287.68028859 ± 1.98558660 | 0.28659586 ± 0.00024875 | 0.36250000 ± 0.01141089 |
| adaptive_conformal | pre-drift | 0.95052083 ± 0.00184142 | 2365.28108870 ± 32.85174269 | 0.29631753 ± 0.00411560 | 0.82500000 ± 0.01141089 |
| static_conformal | drift | 0.87540323 ± 0.00555330 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.17916667 ± 0.01141089 |
| static_conformal | post-drift | 0.91662633 ± 0.00361005 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.24583333 ± 0.00931695 |
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
