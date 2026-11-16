"""PGS242-29: end-to-end run and leakage audit.

Runs the real pipeline twice from raw AEMO files, in two scratch copies of
the repository:

  baseline   the committed data, unchanged
  perturbed  every raw TOTALDEMAND observation after --cutoff is changed
             (x 1.25 + 300 MW); everything before it is byte-identical

Pipeline per copy (one forecast seed, one detector, one Arm D window):

  1. experiments/detection/run_aemo_drift_detection.py   (all detectors)
  2. experiments/adaptation/run_four_arm_adaptation.py   (A, B, C, D)
  3. experiments/uncertainty/run_aemo_agaci.py           (static, ACI, AgACI-BOA)
  4. lagged 30-minute escalation score, as in run_escalation_lag1_nonconformity.py

Leakage test: a causal pipeline cannot let a change to the future alter
anything decided earlier. So, comparing baseline with perturbed:

  * accepted alarms strictly before the first perturbed half-hour must match;
  * forecasts and intervals up to and including the first perturbed
    half-hour must match exactly (the forecast and interval for t are issued
    before y_t is seen);
  * the lagged review score must match one step further, because score(t)
    only uses y_(t-1);
  * the calibration radius must match (TEST data never enters it);
  * after the cutoff the outputs must differ, which shows the perturbation
    actually reached the pipeline.

Reproducibility test: baseline outputs are compared with the committed
results (change points, seed-level adaptation summary, PGS242-160 and
AgACI interval summaries).

It also checks split integrity, and perturbs a single y_t inside one stored
stream to test the interval and score functions directly.

Nothing under results/ in the repository is modified. All work happens in
--work-dir; the report is written to --report-dir.

Example:
    PYTHONPATH=src python experiments/audit/run_end_to_end_audit.py \
        --work-dir /tmp/pgs242_29 --report-dir results/audit/pgs242_29
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ("SA1", "NSW1")
COPY_ITEMS = ("src", "experiments", "data", "docs/decisions.md", "pyproject.toml")
STORED_CHANGEPOINTS = ROOT / "results/tables/detection/aemo_detection_changepoints.csv"
STORED_ADAPTATION = ROOT / "results/tables/adaptation/revised/pgs242_128_five_seed_final"
STORED_AGACI = ROOT / "results/tables/uncertainty/agaci_boa_fixed_calibration"
METHOD_COLUMNS = {
    "static_conformal": ("static_lower", "static_upper"),
    "adaptive_conformal": ("aci_lower", "aci_upper"),
    "agaci_boa": ("agaci_lower", "agaci_upper"),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.checks = []

    def add(self, group, name, passed, detail=""):
        self.checks.append(dict(group=group, check=name, passed=bool(passed), detail=str(detail)))
        print(f"[{'PASS' if passed else 'FAIL'}] {group}: {name}" + (f" — {detail}" if detail else ""))

    @property
    def n_failed(self):
        return sum(not c["passed"] for c in self.checks)


def package_versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "statsmodels", "river"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def make_copy(dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in COPY_ITEMS:
        src = ROOT / item
        target = dest / item
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, target)
    (dest / "results").mkdir()


def perturb_raw(copy_root: Path, cutoff: pd.Timestamp, scale: float, shift: float):
    """Change TOTALDEMAND for raw rows strictly after the cutoff. Returns rows changed."""
    changed = 0
    for region in REGIONS:
        for path in sorted((copy_root / "data/raw" / region).glob("*.csv")):
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            header, body = rows[0], rows[1:]
            i_time, i_demand = header.index("SETTLEMENTDATE"), header.index("TOTALDEMAND")
            touched = False
            for row in body:
                if pd.Timestamp(row[i_time]) > cutoff:
                    row[i_demand] = f"{float(row[i_demand]) * scale + shift:.2f}"
                    changed += 1
                    touched = True
            if touched:
                with path.open("w", newline="") as handle:
                    csv.writer(handle, lineterminator="\n").writerows([header, *body])
    return changed


def run(cmd, cwd: Path, log: Path):
    env = {"PYTHONPATH": str(cwd / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin",
           "HOME": str(Path.home()), "MPLBACKEND": "Agg"}
    start = time.time()
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed; see {log}")
    return round(time.time() - start, 1)


def run_pipeline(copy_root: Path, args, label: str):
    logs = copy_root / "logs"
    logs.mkdir(exist_ok=True)
    py = sys.executable
    timings = {}
    if args.reuse:
        timings = json.loads((logs / "timings.json").read_text())
        return load_outputs(copy_root, args, label, timings)
    timings["detection"] = run(
        [py, "experiments/detection/run_aemo_drift_detection.py"], copy_root, logs / "detection.log")
    timings["adaptation"] = run(
        [py, "experiments/adaptation/run_four_arm_adaptation.py",
         "--detections", "results/tables/detection/aemo_detection_changepoints.csv",
         "--decisions", "docs/decisions.md", "--detectors", args.detector,
         "--windows", str(args.window), "--seeds", str(args.seed),
         "--run-id", f"audit_{label}", "--output-root", "adaptation"],
        copy_root, logs / "adaptation.log")
    timings["uncertainty"] = run(
        [py, "experiments/uncertainty/run_aemo_agaci.py",
         "--run-dir", f"adaptation/audit_{label}", "--seeds", str(args.seed),
         "--output-dir", "agaci"],
        copy_root, logs / "uncertainty.log")
    (logs / "timings.json").write_text(json.dumps(timings))
    return load_outputs(copy_root, args, label, timings)


def load_outputs(copy_root: Path, args, label: str, timings):
    seed_dir = copy_root / "adaptation" / f"audit_{label}" / f"seed_{args.seed}"
    return dict(
        root=copy_root,
        timings=timings,
        changepoints=pd.read_csv(copy_root / "results/tables/detection/aemo_detection_changepoints.csv",
                                 parse_dates=["SETTLEMENTDATE"]),
        summary=pd.read_csv(seed_dir / "summary.csv"),
        retrains=pd.read_csv(seed_dir / "retraining_events.csv"),
        forecasts={r: pd.read_csv(seed_dir / f"forecasts_{r}.csv.gz", parse_dates=["SETTLEMENTDATE"], low_memory=False)
                   for r in REGIONS},
        intervals={r: pd.read_csv(copy_root / "agaci" / f"seed_{args.seed}" /
                                  f"agaci_comparison_intervals_{r}.csv.gz", parse_dates=["SETTLEMENTDATE"])
                   for r in REGIONS},
        uq_summary={r: pd.read_csv(copy_root / "agaci" / f"seed_{args.seed}" /
                                   f"agaci_comparison_summary_{r}.csv") for r in REGIONS},
    )


def lag1_score(y, f):
    return np.concatenate(([0.0], np.abs(np.asarray(y[:-1], float) - np.asarray(f[:-1], float))))


def frames_equal(a, b, cols):
    a = a.reset_index(drop=True)[cols]
    b = b.reset_index(drop=True)[cols]
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    diff = 0.0
    for c in cols:
        if pd.api.types.is_numeric_dtype(a[c]):
            diff = max(diff, float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))) if len(a) else 0.0))
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True):
                return False, f"{c} max|diff| = {diff:.3g}"
        elif not (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all():
            return False, f"{c} differs"
    return True, f"{len(a)} rows, max|diff| = {diff:.3g}"


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_splits(rep: Report):
    sys.path.insert(0, str(ROOT / "src"))
    from drift_lab.data_loader import load_and_split_aemo

    for region in REGIONS:
        full, train, cal, test = load_and_split_aemo(region, ROOT / "data/raw")
        g = f"split {region}"
        rep.add(g, "TRAIN < CALIBRATION < TEST, no overlap",
                train.SETTLEMENTDATE.max() < cal.SETTLEMENTDATE.min()
                and cal.SETTLEMENTDATE.max() < test.SETTLEMENTDATE.min(),
                f"TRAIN ends {train.SETTLEMENTDATE.max()}, CAL {cal.SETTLEMENTDATE.min()}–"
                f"{cal.SETTLEMENTDATE.max()}, TEST starts {test.SETTLEMENTDATE.min()}")
        rep.add(g, "row counts 35,039 / 2,880 / 67,249",
                (len(train), len(cal), len(test)) == (35039, 2880, 67249),
                f"{len(train)} / {len(cal)} / {len(test)}")
        steps = full.SETTLEMENTDATE.diff().dropna()
        rep.add(g, "continuous 30-minute grid, no duplicates",
                steps.eq(pd.Timedelta(minutes=30)).all() and full.SETTLEMENTDATE.is_unique,
                f"{len(full)} rows, {full.SETTLEMENTDATE.min()} to {full.SETTLEMENTDATE.max()}")
        processed = ROOT / "data/processed" / f"{region}.csv"
        if processed.is_file():
            p = pd.read_csv(processed, parse_dates=["SETTLEMENTDATE"])
            same = (len(p) == len(full)
                    and (p.SETTLEMENTDATE.to_numpy() == full.SETTLEMENTDATE.to_numpy()).all()
                    and np.allclose(p.TOTALDEMAND.to_numpy(float), full.TOTALDEMAND.to_numpy(float), atol=1e-6))
            rep.add(g, "data/processed matches the loader output", same, processed.relative_to(ROOT))


def check_reproducibility(rep: Report, base, args):
    g = "reproducibility"
    stored = pd.read_csv(STORED_CHANGEPOINTS, parse_dates=["SETTLEMENTDATE"])
    keys = ["region", "detector", "SETTLEMENTDATE", "detected_index"]
    ok, detail = frames_equal(stored.sort_values(keys), base["changepoints"].sort_values(keys),
                              keys + ["forecast_error", "detector_input_value"])
    rep.add(g, "re-run change points match committed aemo_detection_changepoints.csv", ok, detail)

    stored_summary = pd.read_csv(STORED_ADAPTATION / f"seed_{args.seed}" / "summary.csv")
    m = base["summary"].merge(stored_summary, on=["region", "configuration"], suffixes=("", "_stored"))
    cols = ["overall_mae", "n_retrains", "train_samples_total"]
    ok = len(m) == len(base["summary"]) and all(np.allclose(m[c], m[c + "_stored"], atol=1e-9) for c in cols)
    rep.add(g, f"re-run seed-{args.seed} adaptation summary matches committed (MAE, retrains, samples)",
            ok, "; ".join(f"{r.region}/{r.configuration} MAE {r.overall_mae:.3f}" for r in m.itertuples()))

    pgs160 = pd.read_csv(STORED_ADAPTATION / f"seed_{args.seed}" / "uncertainty" /
                         "aemo_conformal_comparison_summary.csv")
    for region in REGIONS:
        mine = base["uq_summary"][region]
        width = mine.loc[(mine.configuration == "A") & (mine.method == "static_conformal"),
                         "mean_interval_width"].iloc[0]
        w160 = pgs160.loc[(pgs160.region == region) & (pgs160.configuration == "A") &
                          (pgs160.method == "static_conformal"), "mean_interval_width"].iloc[0]
        rep.add(g, f"{region} static calibration width matches committed PGS242-160",
                abs(width - w160) < 1e-6, f"re-run {width:.4f} MW, PGS242-160 {w160:.4f} MW")
        agaci = pd.read_csv(STORED_AGACI / f"seed_{args.seed}" / f"agaci_comparison_summary_{region}.csv")
        wag = agaci.loc[(agaci.configuration == "A") & (agaci.method == "static_conformal"),
                        "mean_interval_width"].iloc[0]
        rep.add("reproducibility (committed AgACI)",
                f"{region} static calibration width matches committed AgACI comparison",
                abs(width - wag) < 1e-6,
                f"re-run {width:.4f} MW, committed AgACI {wag:.4f} MW ({100 * (wag / width - 1):+.2f}%)")


def check_future_perturbation(rep: Report, base, pert, first_changed):
    g = "future perturbation"
    for region in REGIONS:
        b = base["changepoints"][base["changepoints"].region == region]
        p = pert["changepoints"][pert["changepoints"].region == region]
        keys = ["detector", "SETTLEMENTDATE", "detected_index"]
        ok, detail = frames_equal(b[b.SETTLEMENTDATE < first_changed].sort_values(keys),
                                  p[p.SETTLEMENTDATE < first_changed].sort_values(keys), keys)
        rep.add(g, f"{region} alarms before {first_changed} unchanged", ok, detail)
        rep.add(g, f"{region} alarms after the cutoff do change (sanity)",
                not frames_equal(b[b.SETTLEMENTDATE >= first_changed].sort_values(keys),
                                 p[p.SETTLEMENTDATE >= first_changed].sort_values(keys), keys)[0],
                f"{(b.SETTLEMENTDATE >= first_changed).sum()} vs {(p.SETTLEMENTDATE >= first_changed).sum()} alarms")

        fb, fp = base["forecasts"][region], pert["forecasts"][region]
        upto = lambda f: f[f.SETTLEMENTDATE <= first_changed].sort_values(["configuration", "SETTLEMENTDATE"])
        ok, detail = frames_equal(upto(fb), upto(fp), ["configuration", "SETTLEMENTDATE", "y_pred"])
        rep.add(g, f"{region} forecasts up to and including {first_changed} unchanged (all arms)", ok, detail)
        later = lambda f: f[f.SETTLEMENTDATE > first_changed].sort_values(["configuration", "SETTLEMENTDATE"])
        rep.add(g, f"{region} forecasts after the cutoff do change (sanity)",
                not frames_equal(later(fb), later(fp), ["configuration", "y_pred"])[0])

        ib, ip = base["intervals"][region], pert["intervals"][region]
        cols = ["configuration", "SETTLEMENTDATE"] + [c for pair in METHOD_COLUMNS.values() for c in pair]
        upto_i = lambda f: f[f.SETTLEMENTDATE <= first_changed].sort_values(["configuration", "SETTLEMENTDATE"])
        ok, detail = frames_equal(upto_i(ib), upto_i(ip), cols)
        rep.add(g, f"{region} static/ACI/AgACI intervals up to and including {first_changed} unchanged", ok, detail)

        sb, sp = base["uq_summary"][region], pert["uq_summary"][region]
        wb = sb[sb.method == "static_conformal"].mean_interval_width.to_numpy()
        wp = sp[sp.method == "static_conformal"].mean_interval_width.to_numpy()
        rep.add(g, f"{region} calibration radius unchanged by TEST perturbation", np.allclose(wb, wp, atol=1e-9),
                f"static width {wb[0]:.4f} vs {wp[0]:.4f} MW")

        # Lagged review score: score(t) uses y_(t-1) only.
        for cfg, grp in ib.groupby("configuration"):
            gp = ip[ip.configuration == cfg]
            sb_ = lag1_score(grp.y_true, grp.point_forecast)
            sp_ = lag1_score(gp.y_true, gp.point_forecast)
            # k = index of the first step after first_changed; its score uses the
            # perturbed y at first_changed, so it is the first score allowed to move.
            k = int((grp.SETTLEMENTDATE <= first_changed).sum())
            if not np.array_equal(sb_[:k], sp_[:k]) or sb_[k] == sp_[k]:
                rep.add(g, f"{region} lag-1 review score unchanged up to and including {first_changed}, "
                        "first changes one step later", False, cfg)
                break
        else:
            rep.add(g, f"{region} lag-1 review score unchanged up to and including {first_changed}, "
                    "first changes one step later", True, f"{ib.configuration.nunique()} configurations")

        rb = base["retrains"][base["retrains"].region == region]
        end = pd.to_datetime(rb.training_window_end)
        at = pd.to_datetime(rb.retrain_timestamp)
        first = pd.to_datetime(rb.first_forecast_timestamp)
        bad = int(((end > at) | (first <= end)).sum())
        rep.add("adaptation", f"{region} each retrain trains on data up to its retrain time, "
                "and the refitted model is first used after it",
                bad == 0, f"{len(rb)} retrain events, {bad} violations")


def check_single_step(rep: Report, base):
    """Perturb one y_t inside a real stream; interval_t and score_t must not move."""
    sys.path.insert(0, str(ROOT / "src"))
    from drift_lab.uncertainty import adaptive_prediction_intervals
    from drift_lab.agaci import agaci_prediction_intervals

    region = "SA1"
    iv = base["intervals"][region]
    grp = iv[iv.configuration == "A"].reset_index(drop=True)
    f, y = grp.point_forecast.to_numpy(float), grp.y_true.to_numpy(float)
    width = float((grp.static_upper - grp.static_lower).iloc[0])
    # Rebuild residuals with the same radius is not needed: use the run's own residual quantile
    # via symmetric pseudo-residuals reproducing the calibration width is not exact, so we
    # perturb through the functions with a fixed synthetic calibration set instead.
    rng = np.random.default_rng(0)
    cal = rng.normal(0, width / 3.3, 2880)
    j = 5000
    y2 = y.copy()
    y2[j] += 5000.0
    g = "single-step y_t"
    a1 = adaptive_prediction_intervals(f, y, cal)[:2]
    a2 = adaptive_prediction_intervals(f, y2, cal)[:2]
    rep.add(g, "ACI interval at t unchanged when only y_t changes",
            np.array_equal(a1[0][:j + 1], a2[0][:j + 1]) and np.array_equal(a1[1][:j + 1], a2[1][:j + 1]),
            "and every earlier interval")
    rep.add(g, "ACI interval at t+1 reacts (sanity)", not np.array_equal(a1[0][j + 1:], a2[0][j + 1:]))
    g1 = agaci_prediction_intervals(f, y, cal)[:2]
    g2 = agaci_prediction_intervals(f, y2, cal)[:2]
    rep.add(g, "AgACI-BOA interval at t unchanged when only y_t changes",
            np.array_equal(g1[0][:j + 1], g2[0][:j + 1]) and np.array_equal(g1[1][:j + 1], g2[1][:j + 1]))
    s1, s2 = lag1_score(y, f), lag1_score(y2, f)
    rep.add(g, "lag-1 score at t unchanged, changes at t+1 only",
            np.array_equal(s1[:j + 1], s2[:j + 1]) and s1[j + 1] != s2[j + 1]
            and np.array_equal(s1[j + 2:], s2[j + 2:]))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def write_report(rep: Report, args, base, pert, changed_rows, first_changed, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = base["summary"][["region", "configuration", "overall_mae", "n_retrains"]]
    uq = pd.concat(base["uq_summary"].values())[["region", "configuration", "method",
                                                  "empirical_coverage", "mean_interval_width"]]
    payload = dict(
        task="PGS242-29", generated=pd.Timestamp.now("Australia/Adelaide").isoformat(timespec="seconds"),
        seed=args.seed, detector=args.detector, window_days=args.window,
        cutoff=str(args.cutoff), first_perturbed_half_hour=str(first_changed),
        perturbation=dict(scale=args.scale, shift_mw=args.shift, raw_rows_changed=changed_rows),
        environment=package_versions(),
        timings_s=dict(baseline=base["timings"], perturbed=pert["timings"]),
        n_checks=len(rep.checks), n_failed=rep.n_failed, checks=rep.checks,
    )
    (report_dir / "audit_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    summary.to_csv(report_dir / "end_to_end_adaptation_summary.csv", index=False)
    uq.to_csv(report_dir / "end_to_end_uncertainty_summary.csv", index=False)

    lines = [
        "# PGS242-29 — End-to-end run and leakage audit", "",
        "Generated by `experiments/audit/run_end_to_end_audit.py`; do not edit by hand.", "",
        f"**Result: {len(rep.checks) - rep.n_failed} of {len(rep.checks)} checks passed.**", "",
        "## Run", "",
        f"- Seed {args.seed}; detector `{args.detector}`; Arm D window {args.window} days; regions SA1 and NSW1.",
        f"- Perturbation: raw TOTALDEMAND × {args.scale} + {args.shift} MW for every raw row after "
        f"{args.cutoff} ({changed_rows:,} rows). First affected half-hour: {first_changed}.",
        "- Environment: " + ", ".join(f"{k} {v}" for k, v in package_versions().items()), "",
        "| Stage | Baseline (s) | Perturbed (s) |", "| --- | ---: | ---: |",
        *[f"| {k} | {base['timings'][k]} | {pert['timings'][k]} |" for k in base["timings"]], "",
        "## Checks", "",
        "| Group | Check | Result | Detail |", "| --- | --- | --- | --- |",
        *[f"| {c['group']} | {c['check']} | {'PASS' if c['passed'] else '**FAIL**'} | {c['detail'].replace('|', chr(92) + '|')} |"
          for c in rep.checks], "",
        "## End-to-end outputs (baseline)", "",
        summary.to_markdown(index=False, floatfmt=".3f"), "",
        uq.to_markdown(index=False, floatfmt=".4f"), "",
    ]
    (report_dir / "AUDIT_REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True, help="Scratch folder for the two pipeline copies.")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/audit/pgs242_29")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--detector", default="page_hinkley", choices=["adwin", "kswin", "page_hinkley"])
    parser.add_argument("--window", type=int, default=180, choices=[30, 60, 120, 180])
    parser.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp("2022-01-01 00:00"))
    parser.add_argument("--scale", type=float, default=1.25)
    parser.add_argument("--shift", type=float, default=300.0)
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse pipeline outputs already in --work-dir; only re-run the checks.")
    args = parser.parse_args()
    if args.cutoff <= pd.Timestamp("2020-03-01"):
        parser.error("--cutoff must fall inside TEST (after 2020-03-01).")

    rep = Report()
    check_splits(rep)

    base_root, pert_root = args.work_dir / "baseline", args.work_dir / "perturbed"
    if not args.reuse:
        make_copy(base_root)
        make_copy(pert_root)
        changed = perturb_raw(pert_root, args.cutoff, args.scale, args.shift)
        (args.work_dir / "perturbed_rows.txt").write_text(str(changed))
    else:
        changed = int((args.work_dir / "perturbed_rows.txt").read_text())
    # Raw files are 5-minute after Oct 2021 and are averaged into right-labelled
    # half-hours, so the first half-hour that contains a perturbed raw row is:
    first_changed = (args.cutoff + pd.Timedelta(minutes=1)).ceil("30min")

    print("Running baseline pipeline…")
    base = run_pipeline(base_root, args, "baseline")
    print("Running perturbed pipeline…")
    pert = run_pipeline(pert_root, args, "perturbed")

    check_reproducibility(rep, base, args)
    check_future_perturbation(rep, base, pert, first_changed)
    check_single_step(rep, base)
    write_report(rep, args, base, pert, changed, first_changed, args.report_dir)
    print(f"\n{len(rep.checks) - rep.n_failed}/{len(rep.checks)} checks passed. Report: {args.report_dir}")
    sys.exit(1 if rep.n_failed else 0)


if __name__ == "__main__":
    main()
