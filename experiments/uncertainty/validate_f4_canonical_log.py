#!/usr/bin/env python3
"""Validate F4 detector-panel outputs against current T5 and input evidence.

Checks the generator's five-seed stream audit, figure coverage, per-configuration
rolling CSVs, aggregated seed means/SD, source hashes and comparable T5 metrics.
Does not validate or alter the canonical results/runs.csv log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ('static', 'aci', 'agaci')
METRICS = ('empirical_coverage', 'mean_interval_width',
           'normalised_interval_width', 'worst_24h_coverage')
REGIMES = ('pre-drift', 'drift', 'post-drift')
IDENTITY = ['region', 'seed', 'configuration', 'method', 'regime']
METHOD_ALIASES = {
    'static': ('static', 'static_conformal', 'split_conformal', 'static_split_conformal'),
    'aci': ('aci', 'adaptive', 'adaptive_conformal', 'aci_conformal'),
    'agaci': ('agaci', 'agaci_boa', 'agaci_boa_conformal', 'agaci_boa_90'),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_method(raw_method, config):
    raw = str(raw_method).strip().lower().replace('-', '_').replace(' ', '_')
    if '|' in raw:
        prefix, raw = raw.split('|', 1)
        if prefix != str(config).lower():
            return None
    for method, aliases in METHOD_ALIASES.items():
        if raw in aliases:
            return method
    return None


def t5_to_long(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    required = {'region', 'seed', 'method', 'regime'}
    if not required <= set(data):
        raise ValueError(f'T5 missing columns: {sorted(required-set(data))}')
    if 'configuration' not in data:
        if not data.method.astype(str).str.contains('|', regex=False).all():
            raise ValueError('T5 needs a configuration column or configuration|method keys.')
        data['configuration'] = data.method.astype(str).str.split('|', regex=False).str[0]
    data['method'] = [canonical_method(m, c) for m,c in zip(data.method, data.configuration)]
    data = data.loc[data.method.notna()].copy()
    if data.empty:
        raise ValueError('No Static, ACI or AgACI method names recognised in T5 metrics.')
    if {'metric_name','metric_value'} <= set(data):
        result = data[IDENTITY + ['metric_name','metric_value']].rename(
            columns={'metric_name':'metric', 'metric_value':'expected'})
    else:
        available = [metric for metric in METRICS if metric in data]
        if not available:
            raise ValueError('T5 contains no comparable named metrics.')
        result = data.melt(id_vars=IDENTITY,value_vars=available,
                           var_name='metric',value_name='expected')
    result = result.loc[result.metric.isin(METRICS)].copy()
    if result.duplicated(IDENTITY + ['metric']).any():
        raise ValueError('Duplicate T5 identity/metric rows.')
    return result


def check_t5_metrics(f4_dir, t5_dir, report):
    f4_file = f4_dir / 'F4_t5_recomputed_metrics.csv'
    t5_file = t5_dir / 't5_seed_metrics.csv'
    if not t5_file.exists():
        raise FileNotFoundError(f'Current T5 seed metrics not found: {t5_file}')
    f4 = pd.read_csv(f4_file)
    if not set(IDENTITY + list(METRICS)) <= set(f4):
        raise ValueError('F4 missing columns necessary for current T5 cross-check.')
    if f4.duplicated(IDENTITY).any():
        raise ValueError('Duplicate F4 seed/configuration/method/regime rows.')
    actual = f4.melt(id_vars=IDENTITY,value_vars=list(METRICS),
                     var_name='metric',value_name='actual')
    expected = t5_to_long(pd.read_csv(t5_file))
    matched = actual.merge(expected,on=IDENTITY+['metric'],how='inner',validate='one_to_one')
    if matched.empty:
        raise ValueError('No matching F4/T5 metric rows. Check T5 method naming and config columns.')
    groups = f4[IDENTITY].drop_duplicates()
    covered_groups = matched[IDENTITY].drop_duplicates()
    missing = groups.merge(covered_groups,on=IDENTITY,how='left',indicator=True)
    if missing['_merge'].eq('left_only').any():
        sample = missing.loc[missing['_merge'].eq('left_only'), IDENTITY].head(5)
        raise ValueError('T5 missing a metric for some F4 groups:\n'+sample.to_string(index=False))
    # T5 methods may legitimately omit a metric, for example normalisation if the
    # region normalisation constants were not exported. Compare only common fields.
    np.testing.assert_allclose(matched.actual.to_numpy(dtype=float),
                               pd.to_numeric(matched.expected,errors='coerce').to_numpy(dtype=float),
                               rtol=0,atol=1e-8,equal_nan=True,
                               err_msg='F4 and T5 metrics disagree; check regime rules and input versions.')
    matched['passed'] = True
    outfile = f4_dir / 'F4_current_t5_validation.csv'
    matched.to_csv(outfile,index=False)
    return {'matched_metrics':len(matched),'matched_groups':len(covered_groups),
            't5_source':str(t5_file),'t5_source_sha256':digest(t5_file),
            'comparison_csv':outfile.name,'comparison_csv_sha256':digest(outfile)}


def check_figures_and_streams(f4_dir, report):
    seeds = report['seeds']; regions = report['regions']
    groups = report.get('detector_panels',{})
    if not groups:
        raise ValueError('Generator manifest has no detector_panels; run current F4 generator.')
    configs = set(report['configurations'])
    for detector, panel in groups.items():
        if len(panel)!=4 or panel[0:2]!=['A','B'] or not set(panel)<=configs:
            raise ValueError(f'Invalid A/B/C/D panel mapping: {detector}: {panel}')
        if '180' not in panel[3].lower():
            raise ValueError(f'Arm D must use the 180-day window: {detector}: {panel[3]}')
        for region in regions:
            figure = f4_dir/f'F4_{region}_{detector}_four_arms.png'
            if not figure.is_file() or figure.stat().st_size < 5000:
                raise FileNotFoundError(f'Missing or empty detector figure: {figure}')
    expected = {(r,int(s),c) for r in regions for s in seeds for c in configs}
    records=report.get('streams',[])
    actual={(r.get('region'),int(r.get('seed')),r.get('configuration')) for r in records}
    if actual != expected or len(records)!=len(expected):
        raise ValueError(f'F4 stream audit incomplete: {len(records)} records for {len(expected)} expected runs.')
    flags=('chronological','interval_bounds','coverage_flags',
           'width_validation','trailing_window','prefix_invariance')
    for entry in records:
        if not all(entry.get(flag) is True for flag in flags):
            raise ValueError(f'F4 failed/missing stream check: {entry}')
    for region in regions:
        for config in configs:
            agg_path=f4_dir/f'F4_{region}_{config}_mean_sd.csv.gz'
            if not agg_path.exists():
                raise FileNotFoundError(agg_path)
            agg=pd.read_csv(agg_path)
            if agg.empty or 'SETTLEMENTDATE' not in agg:
                raise ValueError(f'Empty/malformed aggregate: {agg_path}')
            for method in METHODS:
                required={f'{method}_mean',f'{method}_sd',f'{method}_n_seeds'}
                if not required <= set(agg):
                    raise ValueError(f'Missing {method} seed aggregation: {agg_path}')
                counts=agg[f'{method}_n_seeds'].dropna()
                if not counts.empty and not counts.isin([0,len(seeds)]).all():
                    raise ValueError(f'Incomplete seed aggregation for {method}: {agg_path}')
                vals=agg[f'{method}_mean'].dropna()
                if not vals.between(0,1).all():
                    raise ValueError(f'Invalid coverage outside [0,1]: {agg_path}')
                for seed in seeds:
                    per_seed=f4_dir/f'F4_{region}_seed{seed}_{config}_rolling.csv.gz'
                    if not per_seed.exists():
                        raise FileNotFoundError(per_seed)
    return {'figures':len(regions)*len(groups),'audited_streams':len(records),
            'detectors':list(groups),'configurations':len(configs)}


def validate(f4_dir,t5_dir):
    report_file=f4_dir/'F4_validation.json'
    report=json.loads(report_file.read_text(encoding='utf-8-sig'))
    if report.get('status') != 'passed_stream_checks':
        raise ValueError('F4 generator has not completed successful stream checks.')
    if sorted(report.get('methods',[])) != sorted(METHODS):
        raise ValueError('F4 generator method set is not Static/ACI/AgACI.')
    if sorted(report.get('seeds',[])) != [1,2,3,4,5]:
        raise ValueError('Final F4 figures must use seeds 1–5.')
    for name, previous_hash in report.get('sources',{}).items():
        path=Path(name)
        if not path.exists() or digest(path)!=previous_hash:
            raise ValueError(f'Source missing or changed since F4 generation: {path}')
    figure_checks=check_figures_and_streams(f4_dir,report)
    t5_checks=check_t5_metrics(f4_dir,t5_dir,report)
    evidence={'status':'passed_against_current_t5','figure_checks':figure_checks,
              't5_checks':t5_checks,'t5_dir':str(t5_dir),
              'canonical_runs_log':'not_checked; current T5 cross-check only',
              'generator_report':str(report_file)}
    output=f4_dir/'F4_current_t5_validation.json'
    output.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    report['t5_validation']={'status':'passed','matched_seed_metrics':t5_checks['matched_metrics'],
                             'evidence_file':output.name}
    report_file.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f4-dir',type=Path,default=Path('results/figures/uncertainty/f4_by_detector'))
    parser.add_argument('--t5-dir',type=Path,default=Path('results/tables/uncertainty/t5_agaci_full_post7'))
    args=parser.parse_args()
    result=validate(args.f4_dir,args.t5_dir)
    print(f"PASS: {result['figure_checks']['figures']} detector figures, "
          f"{result['figure_checks']['audited_streams']} five-seed streams checked.")
    print(f"PASS: {result['t5_checks']['matched_metrics']} comparable F4/T5 metrics checked.")
    print('Saved F4_current_t5_validation.csv and F4_current_t5_validation.json.')
    print('Canonical results/runs.csv was not validated.')


if __name__=='__main__':
    main()
