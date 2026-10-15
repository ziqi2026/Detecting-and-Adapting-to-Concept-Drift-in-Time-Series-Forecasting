"""Check stored T5 deliverables against independently recomputed seed metrics.

Run from this review copy with:
    python experiments/uncertainty/audit_t5_deliverables.py
This checks existing results and does not retrain or change reported metrics.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/tables/uncertainty/t5_aemo/tables"
REQUIRED_COLUMNS = [
    "UQ method", "regime", "coverage", "mean width",
    "normalised width", "worst 24h coverage",
]
METRIC_COLUMNS = {
    "coverage": "empirical_coverage",
    "mean width": "mean_interval_width",
    "normalised width": "normalised_interval_width",
    "worst 24h coverage": "worst_24h_coverage",
}


def validate_required_deliverables():
    paths = [
        "src/drift_lab/evaluation.py",
        "src/drift_lab/run_logger.py",
        "tests/uncertainty/test_t5_evaluation.py",
        "experiments/uncertainty/generate_t5_uncertainty_table.py",
        "results/runs.csv",
        "results/tables/uncertainty/t5_aemo/protocol.json",
    ]
    paths += [str((TABLES / name).relative_to(ROOT)) for name in (
        "t5_seed_metrics.csv", "t5_window_audit.csv",
        "t5_normalisation.json", "t5_manifest.json",
    )]
    for relative_path in paths:
        if not (ROOT / relative_path).is_file():
            raise ValueError(f"Missing required deliverable: {relative_path}")


def validate_seed_metrics_and_tables():
    runs_path = ROOT / "results/runs.csv"
    runs = pd.read_csv(runs_path)
    independent = pd.read_csv(ROOT / "results/verification/independent_seed_metrics.csv")
    snapshot = pd.read_csv(TABLES / "t5_seed_metrics.csv")
    audit = pd.read_csv(TABLES / "t5_window_audit.csv")
    manifest = json.loads((TABLES / "t5_manifest.json").read_text())
    assert hashlib.sha256(runs_path.read_bytes()).hexdigest() == manifest["runs_sha256"]
    assert manifest["runs_path"] == "results/runs.csv"
    assert len(runs) == len(snapshot) == 4080
    assert len(independent) == len(audit) == 1020
    identity = ["region", "method", "seed", "regime", "metric_name"]
    assert not runs.duplicated(identity).any()
    assert not snapshot.duplicated(identity).any()
    assert set(runs.seed) == {1, 2, 3, 4, 5}
    comparison_columns = [column for column in runs if column != "timestamp"]
    pd.testing.assert_frame_equal(
        runs.sort_values(identity)[comparison_columns].reset_index(drop=True),
        snapshot.sort_values(identity)[comparison_columns].reset_index(drop=True),
    )
    independently_checked = 0
    for row in independent.itertuples(index=False):
        method = f"{row.configuration}|{row.method}"
        selected = runs[
            runs.region.eq(row.region) & runs.method.eq(method)
            & runs.seed.eq(row.seed) & runs.regime.eq(row.regime)
        ]
        window_row = audit[
            audit.region.eq(row.region) & audit.method.eq(method)
            & audit.seed.eq(row.seed) & audit.regime.eq(row.regime)
        ]
        assert len(selected) == 4 and len(window_row) == 1
        assert window_row.iloc[0].n_eligible_24h_windows == row.windows
        for metric in METRIC_COLUMNS.values():
            recorded = selected.loc[selected.metric_name.eq(metric), "metric_value"]
            assert len(recorded) == 1
            np.testing.assert_allclose(recorded.iloc[0], getattr(row, metric), rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(window_row.iloc[0][metric], getattr(row, metric), rtol=1e-11, atol=1e-11)
            independently_checked += 1
    table_paths = sorted(TABLES.glob("T5_*.csv"))
    expected_names = {
        f"T5_{region}_{configuration}"
        for region in ("SA1", "NSW1")
        for configuration in manifest["protocol"]["configurations"]
    }
    assert {path.stem for path in table_paths} == expected_names
    checked_cells = 0
    for path in table_paths:
        region, configuration = path.stem[3:].split("_", 1)
        table = pd.read_csv(path)
        assert list(table.columns) == REQUIRED_COLUMNS and len(table) == 6
        assert not table.duplicated(["UQ method", "regime"]).any()
        assert set(table["UQ method"]) == {"static_conformal", "adaptive_conformal"}
        assert set(table.regime) == {"pre-drift", "drift", "post-drift"}
        assert path.with_suffix(".md").is_file()
        caption = path.with_suffix(".md").read_text()
        denominator = manifest["protocol"]["normalisation"][region]["value"]
        assert f"{denominator:.17g}" in caption
        for row in table.to_dict("records"):
            selected = independent[
                independent.region.eq(region) & independent.configuration.eq(configuration)
                & independent.method.eq(row["UQ method"]) & independent.regime.eq(row["regime"])
            ]
            assert len(selected) == 5 and set(selected.seed) == {1, 2, 3, 4, 5}
            for column, metric in METRIC_COLUMNS.items():
                values = selected[metric].to_numpy(dtype=float)
                expected = "N/A" if not np.isfinite(values).all() else (
                    f"{values.mean():.8f} ± {values.std(ddof=1):.8f}"
                )
                assert row[column] == expected, (path.name, column)
                checked_cells += 1
    return {"tables": len(table_paths), "metric_records": independently_checked,
            "window_audit_rows": len(audit), "summary_cells": checked_cells}


def inspect_names_and_duplicate_files():
    files_by_hash = defaultdict(list)
    function_names = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        files_by_hash[hashlib.sha256(path.read_bytes()).hexdigest()].append(str(path.relative_to(ROOT)))
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assert re.fullmatch(r"_?[a-z][a-z0-9_]*", node.name), (path.name, node.name)
            assert not re.search(r"(?:^|_)(?:pr|pgs)\d", node.name), (path.name, node.name)
            function_names.append(node.name)
    duplicates = [paths for paths in files_by_hash.values() if len(paths) > 1]
    assert not duplicates, duplicates
    return {"function_definitions_checked": len(function_names), "byte_identical_file_groups": duplicates}


def main():
    validate_required_deliverables()
    result = {"status": "passed", **validate_seed_metrics_and_tables(),
              **inspect_names_and_duplicate_files()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
