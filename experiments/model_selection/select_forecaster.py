#!/usr/bin/env python3
"""Historical-only AEMO forecaster selection, using existing drift_lab models.

From repository root:
  cp <downloaded_file> experiments/model_selection/select_forecaster.py
  python3 experiments/model_selection/select_forecaster.py --region both
  python3 experiments/model_selection/select_forecaster.py --report-only

Selection protocol: fit Jan 2018–Sep 2019; validate Oct–Dec 2019.
Calibration Jan–Feb 2020 and test Mar 2020+ are NEVER scored or used for
selection. Five seeds (1..5) for stochastic GB/PatchTST; one for deterministic
Seasonal Naive/SARIMAX. Selection: lowest mean validation MAE among models
satisfying a predeclared runtime budget, tie-break mean total runtime.
Seven-day rolling absolute-error MAE is separately reported for F1 context.
This script DOES NOT alter existing forecaster implementations or adaptation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from drift_lab.data_loader import load_and_split_aemo
from drift_lab.forecasters import MODEL_REGISTRY
from drift_lab.evaluation import calculate_mae, calculate_rolling_mae
from drift_lab.run_logger import ensure_runs_file, log_run

FIT_START = pd.Timestamp('2018-01-01')
VALID_START = pd.Timestamp('2019-10-01')
TRAIN_END = pd.Timestamp('2020-01-01')
WINDOW = 336  # 7 days * 48 half-hour observations; F1 specification
BASELINES = ('seasonal_naive', 'sarimax', 'gradient_boosted')
STOCHASTIC = {'gradient_boosted', 'patchtst'}
NAME = {'seasonal_naive':'Seasonal Naive', 'sarimax':'SARIMAX',
        'gradient_boosted':'Gradient Boosting', 'patchtst':'PatchTST'}


def validate_frame(frame, label):
    if len(frame) == 0:
        raise ValueError(f'{label} is empty')
    ts = frame.SETTLEMENTDATE
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError(f'{label}: timestamps invalid or not unique and sorted')
    if not ts.diff().dropna().eq(pd.Timedelta(minutes=30)).all():
        raise ValueError(f'{label}: 30-minute continuity failed')
    if not np.isfinite(frame.TOTALDEMAND.astype(float).to_numpy()).all():
        raise ValueError(f'{label}: nonfinite target data')


def fmt(mean, sd, n, digits=2):
    if pd.isna(mean):
        return '—'
    return f'{mean:.{digits}f} ± {sd:.{digits}f}' if n > 1 else f'{mean:.{digits}f}'


def md_table(rows, columns):
    s = ['| ' + ' | '.join(columns) + ' |', '|'+ '|'.join(['---'] * len(columns)) +'|']
    for r in rows:
        s.append('| '+ ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(s)


def aggregate(trials, candidates):
    t = pd.DataFrame(trials)
    rows = []
    for model in candidates:
        d = t.loc[(t.model == model) & (t.status == 'ok')]
        expected = 5 if model in STOCHASTIC else 1
        row = dict(model=model, status='ok' if len(d)==expected else 'incomplete',
                   seeds=int(len(d)), expected_seeds=expected)
        for metric in ('mae_mw', 'rolling_7d_mae_mw', 'fit_seconds', 'predict_seconds', 'total_seconds'):
            row[metric + '_mean'] = float(d[metric].mean()) if len(d) else None
            row[metric + '_sd'] = float(d[metric].std(ddof=1)) if len(d)>1 else 0.0 if len(d) else None
        rows.append(row)
    return rows


def report(outdir, max_seconds):
    files = {r:outdir/r/'selection.json' for r in ('SA1','NSW1')}
    present = {r:json.loads(p.read_text()) for r,p in files.items() if p.exists()}
    if not present:
        raise FileNotFoundError('No results found: run --region SA1 / NSW1 / both first')
    for r, s in present.items():
        lines = [f'# Forecaster selection — {r}', '',
            '**Historical validation (October–December 2019), not the calibration window.**', '',
            'Fit January 2018–September 2019; project calibration Jan–Feb 2020 and test from March 2020 are excluded. '
            'All forecasters use the existing `drift_lab` loader and registered architectures.', '',
            '**Predeclared selection rule:** lowest mean validation MAE (MW) among candidates meeting '
            'the optional mean fit+predict runtime budget; runtime breaks ties. '
            '7-day rolling MAE is a supplementary F1-aligned metric, not the selection metric.', '',
            f'Runtime budget: {s["max_seconds"] if s["max_seconds"] is not None else "not specified"} seconds.', '',
            md_table([[NAME.get(a['model'],a['model']), fmt(a['mae_mw_mean'],a['mae_mw_sd'],a['seeds']),
                       fmt(a['rolling_7d_mae_mw_mean'],a['rolling_7d_mae_mw_sd'],a['seeds']),
                       fmt(a['fit_seconds_mean'],a['fit_seconds_sd'],a['seeds']),
                       fmt(a['predict_seconds_mean'],a['predict_seconds_sd'],a['seeds']),
                       fmt(a['total_seconds_mean'],a['total_seconds_sd'],a['seeds']),
                       str(a['seeds'])+'/'+str(a['expected_seeds']),
                       '✓' if a['model']==s['selected_model'] else ('incomplete' if a['status']!='ok' else '')]
                       for a in s['aggregated']],
                     ['Model','Validation MAE (MW)','7-day rolling MAE (MW)', 'Fit (s)','Predict (s)',
                      'Total (s)','Seeds','Pick']), '',
            f'**Selected: {NAME.get(s["selected_model"],s["selected_model"]) if s["selected_model"] else "None — incomplete comparison"}.**', '',
            'Numbers are mean ± sample SD for five-seed models; deterministic models show one run. '
            '7-day rolling MAE averages complete 336-observation rolling windows, and therefore '
            'does not equal overall MAE. Stochastic seeds are 1–5, deterministic seed 0.', '',
            'Model choice does not by itself demonstrate degradation or guarantee that downstream '
            'adaptation uses that model. Region-specific choices must be agreed across teams for T7.', '',
            'Raw data: `validation_runs.csv`; aggregate: `validation_summary.csv`; protocol: `selection.json`.', '',
            'Potential caveat: results are causal one-step-ahead forecasts conditional on observed '
            'past values; they are not 48-step blind forecasts. Compare with Team 41 only if horizons match.', '']
        (outdir/r/'model_selection_report.md').write_text('\n'.join(lines), encoding='utf-8')
    if len(present)==2:
        combined = ['# Forecaster selection — SA1 and NSW1', '',
             '**Validation:** October–December 2019, using only the 2018–2019 training partition. '
             'No calibration or test leakage.', '',
             'Selection by lowest mean validation MAE among runtime-eligible models. '
             'Rolling MAE (336 half-hours), per-seed MAE and separate runtimes are also reported.', '']
        entries=[]
        for region in ('SA1','NSW1'):
            s=present[region]
            for a in s['aggregated']:
                entries.append([region, NAME.get(a['model'],a['model']),
                    fmt(a['rolling_7d_mae_mw_mean'],a['rolling_7d_mae_mw_sd'],a['seeds']),
                    fmt(a['mae_mw_mean'],a['mae_mw_sd'],a['seeds']),
                    fmt(a['fit_seconds_mean'],a['fit_seconds_sd'],a['seeds']),
                    fmt(a['predict_seconds_mean'],a['predict_seconds_sd'],a['seeds']),
                    fmt(a['total_seconds_mean'],a['total_seconds_sd'],a['seeds']),
                    f'{a["seeds"]}/{a["expected_seeds"]}',
                    '✓' if a['model']==s['selected_model'] else a['status'] if a['status']!='ok' else ''])
        combined += [md_table(entries,['Region','Model','7-day rolling MAE (MW)', 'MAE (MW)',
                            'Fit (s)', 'Predict (s)', 'Total (s)', 'Seeds','Pick']), '',
           '**Interpretation:** F1 still evaluates all frozen baselines on the test stream; '
           'this selection uses no F1 degradation results. One forecaster/family should be aligned '
           'between teams before T7 trigger comparisons.', '']
        (outdir/'model_selection_summary.md').write_text('\n'.join(combined),encoding='utf-8')
    print(f'Markdown reports updated in {outdir}')


def run_region(region, args, outdir):
    # Loader returns full series but only train portion is passed to fit/validate.
    _, train_full, _, _ = load_and_split_aemo(region, data_path=args.data_path)
    train_full = train_full[['SETTLEMENTDATE','TOTALDEMAND']].copy().reset_index(drop=True)
    train_full.SETTLEMENTDATE = pd.to_datetime(train_full.SETTLEMENTDATE)
    train_full = train_full.sort_values('SETTLEMENTDATE').reset_index(drop=True)
    validate_frame(train_full, 'train_full')
    if train_full.SETTLEMENTDATE.min() < FIT_START or train_full.SETTLEMENTDATE.max()>=TRAIN_END:
        raise ValueError('Project training split unexpectedly includes out-of-period data')
    fit = train_full[train_full.SETTLEMENTDATE < VALID_START].copy().reset_index(drop=True)
    valid = train_full[train_full.SETTLEMENTDATE >= VALID_START].copy().reset_index(drop=True)
    validate_frame(fit,'fit'); validate_frame(valid,'validation')
    if valid.SETTLEMENTDATE.iloc[0]-fit.SETTLEMENTDATE.iloc[-1]!=pd.Timedelta(minutes=30):
        raise ValueError('Fit/validation boundary not continuous')
    if len(valid) < WINDOW:
        raise ValueError('Not enough validation rows for seven-day rolling MAE')
    rd=outdir/region; rd.mkdir(parents=True, exist_ok=True)
    candidates = [*BASELINES,'patchtst']
    trials=[]
    shared_log = ROOT / 'results' / 'tables' / 'runs.csv'
    ensure_runs_file(results_path=shared_log)
    print(f'{region}: fit {len(fit)}, validate {len(valid)}, window {WINDOW}',flush=True)
    for name in candidates:
        seeds=range(1,6) if name in STOCHASTIC else (0,)
        for seed in seeds:
            base={'region':region,'model':name,'seed':seed,'status':'failed',
                  'mae_mw':None,'rolling_7d_mae_mw':None,'fit_seconds':None,
                  'predict_seconds':None,'total_seconds':None,'n_predictions':None,'error':''}
            try:
                if name not in MODEL_REGISTRY:
                    raise RuntimeError('Missing from MODEL_REGISTRY (optional dependency unavailable?)')
                spec=MODEL_REGISTRY[name]
                config=copy.deepcopy(spec['config'])
                if name in STOCHASTIC:
                    config['seed']=seed
                model=spec['class'](**config)
                t0=time.perf_counter(); model.fit(fit); t1=time.perf_counter()
                # SARIMAX fitted state already includes fit dates; do not extend them again.
                history=None if name=='sarimax' else fit
                pred=np.asarray(model.predict(valid, history_df=history),dtype=float)
                t2=time.perf_counter()
                if pred.shape!=(len(valid),) or not np.isfinite(pred).all():
                    raise ValueError('Nonfinite or incorrectly sized predictions')
                y=valid.TOTALDEMAND.to_numpy(dtype=float)
                # Shared project evaluation functions for alignment with F1.
                mae=float(calculate_mae(y,pred))
                rolling=calculate_rolling_mae(y,pred,window=WINDOW)
                rolling_value=float(np.nanmean(np.asarray(rolling,dtype=float)))
                base.update(status='ok',mae_mw=mae,rolling_7d_mae_mw=rolling_value,
                            fit_seconds=t1-t0,predict_seconds=t2-t1,total_seconds=t2-t0,
                            n_predictions=len(valid))
                # Shared long-format evaluation log; validation is tagged pre-drift.
                hash_id = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
                for metric_name, metric_value in (
                    ('validation_mae_mw', mae),
                    ('validation_rolling_7d_mae_mw', rolling_value),
                    ('validation_fit_seconds', t1-t0),
                    ('validation_predict_seconds', t2-t1),
                    ('validation_total_seconds', t2-t0),
                ):
                    log_run(method=name, dataset='AEMO_TOTALDEMAND', region=region,
                            metric_name=metric_name, metric_value=float(metric_value),
                            regime='pre-drift', train_samples=len(fit), group='model_selection',
                            seed=seed, config_hash=hash_id,
                            split_id='aemo_fit_2018_2019Q3_validation_2019Q4',
                            n_retrains=0, wall_clock_s=float(t2-t0), results_path=shared_log)
                print(f'  {name} seed {seed}: MAE={mae:.3f}, 7d={rolling_value:.3f}, '
                      f'fit={t1-t0:.2f}s predict={t2-t1:.2f}s',flush=True)
            except Exception as ex:
                base['error']=f'{type(ex).__name__}: {ex}'
                print(f'  {name} seed {seed} FAILED: {base["error"]}',flush=True)
            trials.append(base)
            pd.DataFrame(trials).to_csv(rd/'validation_runs.csv',index=False)
    agg=aggregate(trials,candidates)
    pd.DataFrame(agg).to_csv(rd/'validation_summary.csv',index=False)
    all_ok=all(a['status']=='ok' for a in agg)
    eligible=[a for a in agg if a['status']=='ok' and (
        args.max_seconds is None or a['total_seconds_mean'] <= args.max_seconds)]
    winner=min(eligible,key=lambda a:(a['mae_mw_mean'],a['total_seconds_mean']))['model'] if all_ok and eligible else None
    summary={'region':region,'selection_protocol':'2018-2019 historical validation only',
        'fit_range':[str(fit.SETTLEMENTDATE.iloc[0]),str(fit.SETTLEMENTDATE.iloc[-1])],
        'validation_range':[str(valid.SETTLEMENTDATE.iloc[0]),str(valid.SETTLEMENTDATE.iloc[-1])],
        'calibration_test_used':False,'window_halfhours':WINDOW,'seeds_stochastic':[1,2,3,4,5],
        'seeds_deterministic':[0], 'primary_metric':'mean validation MAE (MW)',
        'secondary_metric':'mean 7-day rolling MAE (MW)',
        'selection_rule':'minimum mean validation MAE with optional runtime eligibility; runtime tiebreak',
        'max_seconds':args.max_seconds,'selected_model':winner,'comparison_complete':all_ok,
        'aggregated':agg,'candidate_configs':{n:MODEL_REGISTRY[n]['config'] if n in MODEL_REGISTRY else None for n in candidates}}
    (rd/'selection.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(f'{region}: '+(f'SELECTED {winner}' if winner else 'INCOMPLETE — no final winner'),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--region',choices=('SA1','NSW1','both'),default='both')
    ap.add_argument('--data-path',default=str(ROOT/'data'/'raw'))
    ap.add_argument('--output-dir',default=str(ROOT/'results'/'model_selection'))
    ap.add_argument('--max-seconds',type=float,default=None,
                    help='Predeclare max mean fit+predict runtime in seconds (optional)')
    ap.add_argument('--report-only',action='store_true',
                    help='Regenerate Markdown from this version\'s selection.json, without rerunning')
    args=ap.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        ap.error('--max-seconds must be positive')
    outdir=Path(args.output_dir)
    if not args.report_only:
        for region in ('SA1','NSW1') if args.region=='both' else (args.region,):
            run_region(region,args,outdir)
    report(outdir,args.max_seconds)

if __name__=='__main__':
    main()
