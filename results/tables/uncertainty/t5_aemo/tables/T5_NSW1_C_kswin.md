# T5 — NSW1 / C_kswin

T5 — NSW1, C_kswin. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89879032 ± 0.00047857 | 1622.15677011 ± 4.82238044 | 0.20322045 ± 0.00060414 | 0.32916667 ± 0.00931695 |
| adaptive_conformal | post-drift | 0.90028829 ± 0.00019830 | 1617.83950088 ± 4.01638310 | 0.20267959 ± 0.00050316 | 0.40000000 ± 0.00931695 |
| adaptive_conformal | pre-drift | 0.95052083 ± 0.00184142 | 2365.28108870 ± 32.85174269 | 0.29631753 ± 0.00411560 | 0.82500000 ± 0.01141089 |
| static_conformal | drift | 0.97707816 ± 0.00158395 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.35833333 ± 0.00931695 |
| static_conformal | post-drift | 0.97621683 ± 0.00150650 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.47500000 ± 0.00931695 |
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
