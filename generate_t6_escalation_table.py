"""Generate T6 from stored PGS242-163 outputs only; never fit models.

Discrete operating points: no interpolation or assumed human corrections.
Budget means <= budget; choose highest retained coverage, then least review,
then smallest tau. Target means minimum achieved review with coverage >= target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["region", "configuration", "uq_method", "seed", "regime"]
METRICS = ["review rate for 0.90", "review rate for 0.95",
           "coverage @ 5%", "coverage @ 10%", "coverage @ 20%"]
TOL = 1e-12


def valid_points(frame):
    x = frame.loc[frame.n_retained.gt(0) & frame.retained_coverage.notna()].copy()
    return x[np.isfinite(x.review_rate) & np.isfinite(x.retained_coverage)]


def select_target(frame, target):
    if not 0 <= target <= 1:
        raise ValueError("Target must be in [0, 1]")
    x = valid_points(frame)
    x = x[x.retained_coverage >= target - TOL]
    if x.empty:
        return None
    return x.sort_values(["review_rate", "tau"]).iloc[0]


def select_budget(frame, budget):
    if not 0 <= budget <= 1:
        raise ValueError("Budget must be in [0, 1]")
    x = valid_points(frame)
    x = x[x.review_rate <= budget + TOL]
    if x.empty:
        return None
    return x.sort_values(["retained_coverage", "review_rate", "tau"],
                         ascending=[False, True, True]).iloc[0]


def same(a, b):
    return (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, atol=1e-10, rtol=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sources(sweep, summary, runs):
    if sweep.duplicated(KEYS + ["tau"]).any() or summary.duplicated(KEYS).any():
        raise ValueError("Duplicate sweep/summary keys")
    for col in ["review_rate", "retained_coverage"]:
        finite = sweep[col].dropna()
        if not np.isfinite(finite).all() or not finite.between(0, 1).all():
            raise ValueError(f"Invalid {col}")
    for col in ["n_total", "n_escalated", "n_retained"]:
        if sweep[col].isna().any() or (sweep[col] < 0).any() or (sweep[col] % 1 != 0).any():
            raise ValueError("Invalid counts")
    if not (sweep.n_total > 0).all():
        raise ValueError("Empty regimes require explicit handling")
    if not (sweep.n_total == sweep.n_retained + sweep.n_escalated).all():
        raise ValueError("Counts do not partition stream")
    if not np.allclose(sweep.review_rate, sweep.n_escalated / sweep.n_total, atol=TOL, rtol=0):
        raise ValueError("Review rate/count mismatch")
    if sweep.loc[sweep.n_retained.eq(0), "retained_coverage"].notna().any():
        raise ValueError("Empty retained set must have undefined coverage")
    if sweep.loc[sweep.n_retained.gt(0), "retained_coverage"].isna().any():
        raise ValueError("Nonempty retained set has missing coverage")
    indexed = summary.set_index(KEYS)
    logs = runs[runs.group.eq("escalation_sweep")].copy()
    logkeys = ["method", "region", "seed", "config_hash", "split_id", "regime", "metric_name"]
    loggroups = {k: g.metric_value.tolist() for k, g in logs.groupby(logkeys)}
    checks = 0
    groups = sweep.groupby(KEYS, dropna=False)
    if len(indexed) != groups.ngroups:
        raise ValueError("Summary group coverage mismatch")
    for key, g in groups:
        z = indexed.loc[key]
        if g.config_hash.nunique() != 1 or g.split_id.nunique() != 1:
            raise ValueError("Mixed provenance")
        if z.config_hash != g.config_hash.iloc[0] or z.split_id != g.split_id.iloc[0]:
            raise ValueError("Summary provenance mismatch")
        if set(g.tau_quantile) != {"-inf", "+inf"} | {f"k{k:g}" for k in np.round(np.arange(.5, 3.001, .05), 2)}:
            raise ValueError("Incomplete upstream threshold grid")
        baseline = g[np.isposinf(g.tau)].iloc[0]
        if not same(z.overall_coverage, baseline.retained_coverage):
            raise ValueError("Baseline coverage mismatch")
        expected = {}
        for budget, label in [(.05, "05pct"), (.10, "10pct"), (.20, "20pct")]:
            # Reproduce upstream nearest-rate convention, not T6 budget policy.
            near = g.assign(distance=(g.review_rate-budget).abs()).sort_values(
                ["distance", "review_rate", "tau"]).iloc[0]
            metric = f"retained_coverage_at_review_{label}"
            expected[metric] = near.retained_coverage
            if not same(z[f"review_rate_at_review_{label}"], near.review_rate):
                raise ValueError("Upstream nearest review selection mismatch")
        target = select_target(g, .90)
        expected["review_rate_for_retained_coverage_90pct"] = np.nan if target is None else target.review_rate
        for metric, value in expected.items():
            if not same(z[metric], value):
                raise ValueError(f"Sweep/summary mismatch: {key} {metric}")
            identity = (key[1]+"|"+key[2], key[0], key[3], z.config_hash,
                        z.split_id, key[4], metric)
            found = loggroups.get(identity, [])
            if not found or not all(same(v, value) for v in found):
                raise ValueError(f"Missing/conflicting runs.csv value: {identity}")
            checks += 1
    return checks


def generate(sweep):
    details = []
    for key, g in sweep.groupby(KEYS, dropna=False):
        if set(g.seed.unique()) != {key[3]}:
            raise ValueError("Mixed seed")
        for metric, kind, value in [(METRICS[0], "target", .90), (METRICS[1], "target", .95),
                                    (METRICS[2], "budget", .05), (METRICS[3], "budget", .10),
                                    (METRICS[4], "budget", .20)]:
            point = select_target(g, value) if kind == "target" else select_budget(g, value)
            row = dict(zip(KEYS, key))
            row.update(metric=metric, request=value, status="unreachable" if point is None and kind=="target"
                       else "no feasible threshold" if point is None else "achieved",
                       value=np.nan, tau=np.nan, tau_quantile="", achieved_review_rate=np.nan,
                       achieved_retained_coverage=np.nan, n_retained=np.nan,
                       config_hash=g.config_hash.iloc[0], split_id=g.split_id.iloc[0])
            if point is not None:
                row.update(value=float(point.review_rate if kind=="target" else point.retained_coverage),
                           tau=float(point.tau), tau_quantile=point.tau_quantile,
                           achieved_review_rate=float(point.review_rate),
                           achieved_retained_coverage=float(point.retained_coverage),
                           n_retained=int(point.n_retained))
            details.append(row)
    return pd.DataFrame(details)


def write_outputs(details, output):
    output.mkdir(parents=True, exist_ok=False)
    details.to_csv(output / "t6_seed_operating_points.csv", index=False)
    aggregate = []
    for key, g in details.groupby(["region", "configuration", "uq_method", "regime", "metric"]):
        if sorted(g.seed.tolist()) != [1,2,3,4,5]:
            raise ValueError(f"Expected exactly seeds 1-5: {key}")
        n = int(g.status.eq("achieved").sum())
        if n == 5:
            display = f"{100*g.value.mean():.3f}% ± {100*g.value.std(ddof=1):.3f} pp"
        elif n == 0:
            display = "unreachable" if key[-1].startswith("review") else "no feasible threshold"
        else:
            display = f"partial: {n}/5 seeds achieved; {5-n}/5 unreachable"
        aggregate.append(dict(zip(["region","configuration","uq_method","regime","metric"], key),
                              display=display, n_achieved=n, n_seeds=5,
                              mean=g.value.mean() if n==5 else np.nan,
                              sd=g.value.std(ddof=1) if n==5 else np.nan))
    agg = pd.DataFrame(aggregate)
    agg.to_csv(output / "t6_aggregate_audit.csv", index=False)
    sections = []
    for key, g in agg.groupby(["region", "configuration", "uq_method"]):
        table = g.pivot(index="regime", columns="metric", values="display").reindex(
            ["pre-drift","drift","post-drift","all"])[METRICS].reset_index()
        name = "_".join(key)
        table.to_csv(output / ("t6_"+name+".csv"), index=False)
        sections += ["## " + " / ".join(key), "", table.to_markdown(index=False), ""]
    (output / "T6_operating_points.md").write_text(
        "# T6 — Escalation operating points\n\n"
        "Percentages are seed means ± sample SD in percentage points. Partial reachability is explicit. "
        "Stored thresholds only; no interpolation. Budgets are hard caps. Coverage is on retained "
        "forecasts, not human-corrected system coverage. All is an auxiliary whole-stream row.\n\n"
        + "\n".join(sections), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=Path("results/series/escalation_sweep.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/series/escalation_by_regime.csv"))
    p.add_argument("--manifest", type=Path, default=Path("results/series/escalation_manifest.json"))
    p.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/tables/uncertainty/t6"))
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    for path in [a.sweep, a.summary]:
        if digest(path) != manifest["outputs"][path.name]:
            raise ValueError(f"Upstream hash mismatch: {path}")
    sweep, summary, runs = [pd.read_csv(x) for x in [a.sweep, a.summary, a.runs]]
    checks = validate_sources(sweep, summary, runs)
    details = generate(sweep)
    write_outputs(details, a.output_dir)
    audit = {"task":"T6", "selection":"stored discrete thresholds; no interpolation",
             "target_rule":"minimum review achieving target among nonempty retained sets",
             "budget_rule":"maximum retained coverage at review <= budget; ties least review then smallest tau",
             "coverage_definition":"retained coverage; no assumed human correction",
             "upstream_log_comparisons":checks, "seed_points":len(details),
             "seeds":sorted(map(int,sweep.seed.unique())), "upstream_regimes":manifest["regimes"],
             "inputs_sha256":{str(x):digest(x) for x in [a.sweep,a.summary,a.manifest,a.runs]},
             "generator_sha256":digest(__file__),
             "runs_note":"Read-only; validates upstream legacy nearest-rate metrics. New 95% and hard-cap points traced to sweep rows, not claimed as existing log metrics."}
    (a.output_dir / "t6_manifest.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(f"T6 complete: {len(details)} seed operating points; {checks} verified log comparisons")


if __name__ == "__main__":
    main()
