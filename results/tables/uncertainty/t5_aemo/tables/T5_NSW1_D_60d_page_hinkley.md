# T5 — NSW1 / D_60d_page_hinkley

T5 — NSW1, D_60d_page_hinkley. Normalised width denominator: 7982.2516647164603 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

Values are mean ± sample standard deviation across seeds 1–5 (ddof=1), not confidence intervals. Worst 24h coverage is computed per seed before aggregation. The minimum of 10 complete overlapping windows is an adjustable reporting threshold, not a statistical reliability guarantee.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.90218156 ± 0.00032572 | 2337.96703934 ± 4.89070858 | 0.29289568 ± 0.00061270 | 0.35000000 ± 0.00931695 |
| adaptive_conformal | post-drift | 0.89721597 ± 0.00043499 | 2421.97546973 ± 5.38918835 | 0.30342008 ± 0.00067515 | 0.39583333 ± 0.00000000 |
| adaptive_conformal | pre-drift | 0.95052083 ± 0.00184142 | 2365.28108870 ± 32.85174269 | 0.29631753 ± 0.00411560 | 0.82500000 ± 0.01141089 |
| static_conformal | drift | 0.91802109 ± 0.00374987 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.24166667 ± 0.02795085 |
| static_conformal | post-drift | 0.90586478 ± 0.00364629 | 2798.74013270 ± 55.04806249 | 0.35062038 ± 0.00689631 | 0.16666667 ± 0.00000000 |
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
