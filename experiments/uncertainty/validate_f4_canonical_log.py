"""Validate stored F4 results against the canonical T5 log exported by git archive.

No model fitting, interval regeneration, or shared-log changes. Run from repo root.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd

REFERENCE_COMMIT = 'b3beabc72b01e1a497e2764ec967a34080c45425'
KEYS = ['group', 'method', 'dataset', 'region', 'seed', 'config_hash',
        'split_id', 'regime', 'metric_name']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(f4_dir, t5_dir, canonical_zip):
    report_path = f4_dir / 'F4_validation.json'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    if report['status'] != 'passed_against_t5_review_snapshot':
        raise ValueError('Complete the five-seed F4/T5 validation before checking the log.')
    if not report['t5_reference_commit'].startswith(REFERENCE_COMMIT):
        raise ValueError('F4 reference commit differs from this canonical-log validation.')
    base = t5_dir / 'results/tables/uncertainty/t5_aemo'
    manifest_path = base / 'tables/t5_manifest.json'
    seed_path = base / 'tables/t5_seed_metrics.csv'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    with zipfile.ZipFile(canonical_zip) as archive:
        if archive.comment.decode('ascii').strip() != REFERENCE_COMMIT:
            raise ValueError('Canonical log archive is not from the inspected PR38 commit.')
        raw = archive.read('results/runs.csv')
    # Git checkout/archive attributes can convert LF to CRLF on Windows.
    lf_hash = digest(raw.replace(b'\r\n', b'\n'))
    if manifest['runs_sha256'] not in (digest(raw), lf_hash):
        raise ValueError('Canonical log hash differs from T5 even after newline normalisation.')
    canonical = pd.read_csv(io.BytesIO(raw))
    selected = canonical[canonical.group.eq('t5_uncertainty')]
    seed = pd.read_csv(seed_path)
    for frame in (seed, selected):
        if frame.duplicated(KEYS).any():
            raise ValueError('Duplicate canonical T5 metric keys.')
    pd.testing.assert_frame_equal(
        seed.sort_values(KEYS).reset_index(drop=True),
        selected.sort_values(KEYS).reset_index(drop=True))
    reference = selected.set_index(KEYS)
    checks_path = f4_dir / 'F4_t5_seed_validation.csv'
    checks = pd.read_csv(checks_path)
    if len(checks) != report['t5_validation']['seed_metric_checks'] or not checks.passed.eq(True).all():
        raise ValueError('Incomplete or unsuccessful stored F4 checks.')
    recomputed_path = f4_dir / 'F4_t5_recomputed_metrics.csv'
    recomputed = pd.read_csv(recomputed_path)
    if recomputed.duplicated(['region', 'seed', 'configuration', 'method', 'regime']).any():
        raise ValueError('Duplicate recomputed F4 regime rows.')
    rows = []
    for row in recomputed.itertuples():
        for metric in ('empirical_coverage', 'mean_interval_width',
                       'normalised_interval_width', 'worst_24h_coverage'):
            matching = selected[(selected.region == row.region) & (selected.seed == row.seed)
                & (selected.method == f'{row.configuration}|{row.method}')
                & (selected.regime == row.regime) & (selected.metric_name == metric)]
            if len(matching) != 1:
                raise ValueError('Missing or ambiguous canonical log record.')
            ref = matching.iloc[0]
            actual = float(getattr(row, metric))
            np.testing.assert_allclose(actual, float(ref.metric_value), rtol=0,
                                       atol=1e-8, equal_nan=True)
            recorded = checks[(checks.region == row.region) & (checks.seed == row.seed)
                & (checks.configuration == row.configuration) & (checks.method == row.method)
                & (checks.regime == row.regime) & (checks.metric == metric)]
            if len(recorded) != 1:
                raise ValueError('Stored F4 check is missing or duplicated.')
            np.testing.assert_allclose([float(recorded.iloc[0].stored),
                float(recorded.iloc[0].recomputed)], [float(ref.metric_value), actual],
                rtol=0, atol=1e-8, equal_nan=True)
            rows.append({**{key: ref[key] for key in KEYS},
                         'canonical_metric_value': float(ref.metric_value),
                         'f4_recomputed_value': actual, 'passed': True})
    if len(rows) != len(checks):
        raise ValueError('F4 regime metric/check counts differ.')
    evidence = dict(status='passed', reference_commit=REFERENCE_COMMIT,
        canonical_archive=str(canonical_zip), canonical_entry='results/runs.csv',
        canonical_archive_sha256=digest(canonical_zip.read_bytes()),
        canonical_raw_sha256=digest(raw), canonical_lf_sha256=lf_hash,
        t5_manifest_expected_sha256=manifest['runs_sha256'],
        newline_note='CRLF/LF normalisation only; parsed T5 records identical.',
        t5_records_matched=len(seed), f4_metrics_matched=len(rows),
        t5_review_status='provisional; no claim of final regime approval',
        sources={str(path): digest(path.read_bytes()) for path in
            (Path(__file__), manifest_path, seed_path, checks_path, recomputed_path)})
    csv_path = f4_dir / 'F4_canonical_log_validation.csv'
    json_path = f4_dir / 'F4_canonical_log_validation.json'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    evidence['validation_csv_sha256'] = digest(csv_path.read_bytes())
    json_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    report['runs_csv_validation'] = dict(status='passed', canonical_entry='results/runs.csv',
        reference_commit=REFERENCE_COMMIT, t5_records_matched=len(seed),
        f4_metrics_matched=len(rows), evidence_file=json_path.name,
        evidence_sha256=digest(json_path.read_bytes()))
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f4-dir', type=Path, default=Path(
        'results/tables/adaptation/revised/pgs242_128_five_seed_final/f4_t5_review'))
    parser.add_argument('--t5-review-dir', type=Path, default=Path('T5_PR38_Review'))
    parser.add_argument('--canonical-zip', type=Path, default=Path('T5_Canonical_Log.zip'))
    args = parser.parse_args()
    result = validate(args.f4_dir, args.t5_review_dir, args.canonical_zip)
    print(f"PASS: {result['t5_records_matched']} T5 records match canonical results/runs.csv.")
    print(f"PASS: {result['f4_metrics_matched']} F4 metrics trace to the canonical log.")
    print('Saved canonical-log evidence and updated F4_validation.json.')
    print('Regime definition remains provisional pending team/supervisor confirmation.')


if __name__ == '__main__':
    main()
