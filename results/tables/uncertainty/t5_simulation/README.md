# T5 simulation — synthetic data only

This directory contains a generated demonstration of Table 5, not AEMO results.
The fixture covers seeds 1–5, SA1/NSW1 and configuration A. Observations are
synthetic zeros; both methods cover every observation by construction. Thus
100% coverage is an arithmetic check, not evidence of model performance.

## Current conventions — provisional and changeable

| Setting | Current simulation rule |
| --- | --- |
| Normalised width | Mean interval width divided by a fixed synthetic scale of **10** for each region. This is neither TRAIN range nor TRAIN standard deviation. |
| Denominator scope | The same scale is used for all methods, seeds and regimes within each region. |
| Regime scope | `full`, using the shared T3 full-stream assignment, fixed AEST (`Etc/GMT-10`) and seven-day buffers. |
| Event boundaries | Drift is the union of applicable `[start - 7d, end-date + 1d + 7d)` windows. Pre-drift precedes the first applicable buffered window; subsequent non-drift times are post-drift. |
| Rolling boundary | `within_regime`: all 48 observations must belong to the same regime. Windows crossing boundaries are excluded. Disjoint segments are never concatenated. |
| Rolling time | `(t - 24h, t]`, 30-minute cadence, 48 contiguous observations including the endpoint. Incomplete windows are excluded. |
| Undefined worst coverage | N/A if a regime has no valid window; no shorter window is substituted. |
| Aggregation | Compute each metric per seed, then mean and sample SD (`ddof=1`). Find worst coverage within each seed before aggregation. |
| Table granularity | Separate table per region/configuration; no pooling across regions or configurations. |

**These conventions can be changed after team feedback.** The fixed scale 10
is only a synthetic fixture parameter and must not be reused as an AEMO
normalisation denominator. For AEMO, the supervisor requires each region's mean actual demand in TRAIN
[2018-01-01, 2020-01-01) as the fixed denominator. Formal runs verify the
regional source/hash and mean, state the numeric value in each table caption,
and publish complete-window counts per seed/regime. An explicitly agreed
minimum window count is required; below-threshold worst coverage is N/A.
Regime/window policy and the numeric support threshold still need agreement.
Changing those rules requires regenerating the affected tables and logging a
new protocol identity; it does not require model retraining.

## Files

- `protocol.json`: exact synthetic settings used for this demonstration.
- `tables/T5_SA1_A.csv` and `tables/T5_NSW1_A.csv`: exact six-column schema.
- `runs.csv`: 240 synthetic metric records (5 seeds × 2 regions × 2 methods ×
  3 regimes × 4 metrics), separate from the canonical project log.

Values in the tables aggregate the matching records in this directory's
`runs.csv`. Width units are synthetic; coverage is a fraction. The synthetic
split ID and protocol ID distinguish these records from real experiments.
The fixture uses the shared frozen event catalogue only to exercise grouping.

## Reproduce

From the repository root, install `numpy`, `pandas` and `pytest`, then run:

```bash
PYTHONPATH=src python - <<'PYTHON'
import importlib.util
import json
from pathlib import Path

root = Path.cwd()
spec = importlib.util.spec_from_file_location(
    "t5_fixture", root / "tests/uncertainty/test_t5_evaluation.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
work = root / "local_t5_simulation"
work.mkdir(exist_ok=True)
run_dir, protocol = fixture.make_inputs(work)
(work / "protocol.json").write_text(json.dumps(protocol, indent=2))
fixture.t5.generate(
    run_dir, root / "data/events.csv", work / "tables",
    work / "runs.csv", protocol
)
PYTHON
```

Choose a fresh output directory for each reproduction. The generator preserves
existing tables rather than replacing them. It also generates source manifests
and window/seed audit files locally.

Validation: the 41 uncertainty/T3 tests passed again on 15 October 2026;
the synthetic table means/SDs and log-to-table mapping were independently checked.
The full repository suite and formal AEMO T5 have not been validated here.
