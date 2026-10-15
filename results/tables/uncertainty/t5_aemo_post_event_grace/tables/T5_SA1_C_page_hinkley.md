# T5 — SA1 / C_page_hinkley

T5 — SA1, C_page_hinkley. Normalised width denominator: 1300.8730537401182 (training_mean_actual_demand); fixed across all methods, seeds and regimes. TRAIN actual demand in MW, [2018-01-01, 2020-01-01) in fixed AEST.

| UQ method | regime | coverage | mean width | normalised width | worst 24h coverage |
| --- | --- | --- | --- | --- | --- |
| adaptive_conformal | drift | 0.89751340 ± 0.00075489 | 694.14381240 ± 3.26528613 | 0.53359842 ± 0.00251007 | 0.47500000 ± 0.03423266 |
| adaptive_conformal | post-drift | 0.90065304 ± 0.00021276 | 637.71749233 ± 2.88324126 | 0.49022269 ± 0.00221639 | 0.39166667 ± 0.02282177 |
| adaptive_conformal | pre-drift | 0.91222222 ± 0.00359505 | 745.69748917 ± 28.58103473 | 0.57322848 ± 0.02197066 | 0.66250000 ± 0.01743042 |
| static_conformal | drift | 0.93796135 ± 0.00502038 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.33750000 ± 0.01743042 |
| static_conformal | post-drift | 0.95383413 ± 0.00309421 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.31666667 ± 0.02716334 |
| static_conformal | pre-drift | 0.95111111 ± 0.01311747 | 884.64572275 ± 22.44919044 | 0.68004001 ± 0.01725702 | 0.71666667 ± 0.03784563 |

## Complete 24-hour windows

Minimum required per seed/regime: 10. Worst coverage is N/A if any seed has too few windows. Counts are identical for static and adaptive intervals. These overlapping rolling windows are not independent samples.

| Regime | Seed | Complete windows | Worst-window status |
| --- | --- | --- | --- |
| drift | 1 | 15716 | sufficient |
| drift | 2 | 15716 | sufficient |
| drift | 3 | 15716 | sufficient |
| drift | 4 | 15716 | sufficient |
| drift | 5 | 15716 | sufficient |
| post-drift | 1 | 49074 | sufficient |
| post-drift | 2 | 49074 | sufficient |
| post-drift | 3 | 49074 | sufficient |
| post-drift | 4 | 49074 | sufficient |
| post-drift | 5 | 49074 | sufficient |
| pre-drift | 1 | 673 | sufficient |
| pre-drift | 2 | 673 | sufficient |
| pre-drift | 3 | 673 | sufficient |
| pre-drift | 4 | 673 | sufficient |
| pre-drift | 5 | 673 | sufficient |
