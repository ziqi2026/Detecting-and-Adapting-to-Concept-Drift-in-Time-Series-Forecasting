# AEMO T5 results

Evaluation of static and adaptive conformal intervals across SA1/NSW1,
17 configurations and seeds 1–5.

- TEST: 2020-03-01 00:00 through 2024-01-01 00:00, inclusive, fixed AEST.
- Regimes: full; seven-day event buffers.
- Rolling coverage: complete (t-24h, t] windows at 30-minute caden within_regime; incomplete and boundary-crossing windows excluded.
- Normalised width: mean interval width / regional TRAIN mean actual demand.
- TRAIN: [2018-01-01, 2020-01-01), fixed AEST.
- SA1 denominator: 1300.8730537401182 MW.
- NSW1 denominator: 7982.2516647164603 MW.
- Each regional denominator is fixed across methods, seeds and regimes.
- Values are mean ± sample standard deviation across five seeds (ddof=1).
- Worst 24h coverage is calculated per seed, then aggregated.
- Minimum complete-window count: 10 per seed/regime, an adjustable
  implementation choice, not a supervisor-specified number.
- Below the threshold, worst coverage is N/A. Overlapping windows are
  not independent samples.

| Region | Pre-drift | Drift | Post-drift |
| --- | ---: | ---: | ---: |
| SA1 | 337 | 22099 | 43121 |
| NSW1 | 337 | 18639 | 46816 |

Counts above apply separately to each seed/configuration/UQ method.

## Files

- [Evaluation protocol](protocol.json)
- [Tables, captions and audit files](tables/)
- [Separate ealuation log](runs.csv)

The simulation outputs remain separate.
