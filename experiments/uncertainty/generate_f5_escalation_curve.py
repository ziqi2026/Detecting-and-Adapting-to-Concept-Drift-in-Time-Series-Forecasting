"""F5: retained coverage versus human-review rate, separated by drift regime.

Reads an existing escalation_sweep.csv only. No model fitting, threshold
optimisation, synthetic observations, or modification of original results.
Each region/configuration figure contains three regime panels and compares
Static, ACI, and AgACI when available. Five-seed means and sample SD are
computed at identical threshold-grid labels (tau_quantile), not absolute tau.

Example:
  PYTHONPATH=src python3 experiments/uncertainty/generate_f5_escalation_curve.py \
    --sweep results/series/escalation_agaci_full_post7/escalation_sweep.csv \
    --output-dir results/figures/f5_interval_width \
    --score-label 'Interval width'
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REGIMES = ('pre-drift', 'drift', 'post-drift')
METHODS = ('static_conformal', 'adaptive_conformal', 'agaci_boa')
LABELS = {'static_conformal':'Static conformal',
          'adaptive_conformal':'Adaptive conformal (ACI)',
          'agaci_boa':'AgACI-BOA'}
STYLES = {
    'static_conformal': dict(color='#596779', linestyle='--', marker='s', markersize=8.5, zorder=9),
    'adaptive_conformal': dict(color='#167f9f', linestyle='-', marker='o', markersize=3.8, zorder=5),
    'agaci_boa': dict(color='#bb691f', linestyle='-.', marker='o', markersize=5.0, zorder=4),
}
TARGET = 0.90
REQUIRED = {'region','configuration','uq_method','seed','regime',
            'tau_quantile','n_total','n_escalated','n_retained',
            'review_rate','retained_coverage'}


def load_sweep(path):
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f'Missing sweep columns: {sorted(missing)}')
    frame = frame[frame.regime.isin(REGIMES)].copy()
    if frame.empty:
        raise ValueError('No pre-drift, drift, or post-drift rows')
    keys = ['region','configuration','uq_method','seed','regime','tau_quantile']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate sweep rows for a threshold identity')
    if not np.allclose(frame.n_total, frame.n_escalated+frame.n_retained):
        raise ValueError('Escalated/retained counts do not sum to n_total')
    if not np.allclose(frame.review_rate,frame.n_escalated/frame.n_total,atol=1e-10):
        raise ValueError('Review rate disagrees with counts')
    if frame.loc[frame.n_retained.eq(0), 'retained_coverage'].notna().any():
        raise ValueError('Coverage on empty retained set must be undefined')
    groups = frame.groupby(['region','configuration','uq_method','regime'])
    for key, g in groups:
        if set(g.seed.astype(int)) != {1,2,3,4,5}:
            raise ValueError(f'Five seeds (1–5) required for {key}')
        by_seed = g.groupby('seed').tau_quantile.apply(lambda x: set(x.astype(str)))
        if any(s != by_seed.iloc[0] for s in by_seed):
            raise ValueError(f'Inconsistent threshold grids across seeds: {key}')
    return frame


def aggregate(df):
    keys = ['region','configuration','uq_method','regime','tau_quantile']
    rows=[]
    for key,g in df.groupby(keys,sort=False):
        if len(g)!=5 or g.seed.nunique()!=5:
            raise ValueError(f'Expected five distinct seeds at threshold {key}')
        valid=g[g.n_retained.gt(0) & g.retained_coverage.notna()]
        row=dict(zip(keys,key))
        row.update(n_valid=len(valid), review_mean=g.review_rate.mean(),
                   review_sd=g.review_rate.std(ddof=1),
                   coverage_mean=valid.retained_coverage.mean() if len(valid)==5 else np.nan,
                   coverage_sd=valid.retained_coverage.std(ddof=1) if len(valid)==5 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _curve_points(g, max_review):
    # Thresholds are the original measured operating points. No interpolation.
    visible=g[(g.review_mean<=max_review+1e-12) & g.coverage_mean.notna()].copy()
    visible=visible.sort_values(['review_mean','coverage_mean','tau_quantile'])
    # Equal review rates need not have the same coverage across seed averages;
    # retain all distinct
    return visible


def make_figure(summary, region, config, output, score_label, max_review):
    subset=summary[(summary.region==region)&(summary.configuration==config)]
    plt.rcParams.update({'font.family':'serif','font.size':10.5,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16.0,5.6),sharex=True,sharey=True)
    fig.suptitle('F5 Escalation Curve',fontsize=18,fontweight='normal',y=.975)
    fig.text(.5,.902,
             f'Region: {region} · Configuration: {config} · Five-seed aggregate',
             ha='center',va='center',fontsize=12,fontweight='normal',family='serif')
    all_visible=subset[(subset.review_mean<=max_review) & subset.coverage_mean.notna()]
    if all_visible.empty:
        raise ValueError(f'No valid operating points for {region}, {config}')
    # Shared y-axis: zoom to the actual data but always include the 90% target.
    lows=(all_visible.coverage_mean-all_visible.coverage_sd.fillna(0)).to_numpy()
    highs=(all_visible.coverage_mean+all_visible.coverage_sd.fillna(0)).to_numpy()
    ymin=max(0, min(np.nanmin(lows),TARGET)-.025)
    ymax=min(1, max(np.nanmax(highs),TARGET)+.025)
    if ymax-ymin<.06:
        center=(ymin+ymax)/2; ymin=max(0,center-.035); ymax=min(1,center+.035)
    audit=[]
    for ax,regime in zip(axes,REGIMES):
        ax.set_title(regime.capitalize(), fontsize=12.5,fontweight='normal',pad=12)
        ax.axhline(TARGET*100,linestyle='--',color='#3c4954',lw=1.2,alpha=.8,zorder=1)
        ax.axvline(5,color='#b8c1c7',lw=.8,ls=':',zorder=0)
        ax.axvline(10,color='#b8c1c7',lw=.8,ls=':',zorder=0)
        ax.axvline(20,color='#b8c1c7',lw=.8,ls=':',zorder=0)
        any_points=False
        for method in METHODS:
            group=subset[(subset.regime==regime)&(subset.uq_method==method)]
            if group.empty:
                continue
            visible=_curve_points(group,max_review)
            if visible.empty:
                continue
            x=visible.review_mean.to_numpy()*100
            y=visible.coverage_mean.to_numpy()*100
            sd=visible.coverage_sd.fillna(0).to_numpy()*100
            style=STYLES[method]
            color=style['color']
            # Fixed-width conformal intervals usually give only 0% or 100%
            # review. Plot the 0% outcome as a prominent square
            if method == 'static_conformal':
                unique = np.unique(np.round(x, 8), return_index=True)[1]
                x, y, sd = x[unique], y[unique], sd[unique]
            if len(np.unique(np.round(x, 7))) > 1:
                ax.plot(x, y, color=color, linestyle=style['linestyle'],
                        linewidth=1.9, alpha=.95, zorder=style['zorder'])
            ax.plot(x, y, linestyle='none', marker=style['marker'],
                    markersize=style['markersize'], markerfacecolor=color,
                    markeredgecolor='white' if method == 'static_conformal' else color,
                    markeredgewidth=1.0 if method == 'static_conformal' else 0,
                    alpha=.95, zorder=style['zorder'], clip_on=False)
            chosen=np.unique(np.linspace(0,len(x)-1,min(9,len(x))).astype(int))
            ax.errorbar(x[chosen],y[chosen],yerr=sd[chosen],fmt='none',
                        ecolor=color,elinewidth=.95,capsize=2,alpha=.6,
                        zorder=style['zorder']-1)
            any_points=True
            for _,p in group.iterrows():
                audit.append({'region':region,'configuration':config,'regime':regime,
                              'uq_method':method,'threshold':p.tau_quantile,
                              'review_mean':p.review_mean,'review_sd':p.review_sd,
                              'coverage_mean':p.coverage_mean,'coverage_sd':p.coverage_sd,
                              'n_valid_seeds':p.n_valid})
        ax.set_xlim(-1.0,max_review*100)
        ax.set_ylim(ymin*100,ymax*100)
        ax.set_xlabel('Human-review rate (%)',fontsize=11)
        ax.grid(axis='y',color='#d9e0e5',lw=.8,alpha=.7)
        if not any_points:
            ax.text(.5,.5,'No feasible points',transform=ax.transAxes,
                    ha='center',va='center',color='#687580')
    axes[0].set_ylabel('Retained coverage (%)',fontsize=11)
    handles=[Line2D([0],[0],color=STYLES[m]['color'],lw=1.9,
                    ls=STYLES[m]['linestyle'],marker=STYLES[m]['marker'],
                    markersize=5.5,label=LABELS[m])
             for m in METHODS if m in set(subset.uq_method)]
    handles.append(Line2D([0],[0],color='#3c4954',ls='--',lw=1.3,label='90% target'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),
               ncol=len(handles),frameon=False,fontsize=10.5)
    fig.text(.5,.016,
             f'{score_label} · Five-seed mean ± SD · Retained coverage',
             ha='center',fontsize=9,color='#596775')
    fig.subplots_adjust(top=.79,bottom=.205,left=.075,right=.985,wspace=.10)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=220,facecolor='white')
    plt.close(fig)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--score-label',default='Escalation score')
    parser.add_argument('--max-review',type=float,default=.25,
                        help='Maximum review fraction shown on x-axis (default 0.25)')
    args=parser.parse_args()
    if not 0<args.max_review<=1:
        parser.error('--max-review must be in (0, 1]')
    frame=load_sweep(args.sweep)
    summary=aggregate(frame)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f'Output directory is not empty: {args.output_dir}; choose a new folder')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output_dir/'F5_curve_points_five_seed.csv',index=False)
    groups=summary[['region','configuration']].drop_duplicates().sort_values(['region','configuration'])
    for row in groups.itertuples(index=False):
        stem=args.output_dir/f'F5_{row.region}_{row.configuration}'
        make_figure(summary,row.region,row.configuration,stem,args.score_label,args.max_review)
        print(f'Generated {stem.name}.png',flush=True)
    manifest={'source':str(args.sweep),
              'source_sha256':hashlib.sha256(args.sweep.read_bytes()).hexdigest(),
              'score_label':args.score_label,'max_review':args.max_review,
              'regimes':REGIMES,'seeds':[1,2,3,4,5],
              'figure_count':len(groups),
              'aggregation':'mean and sample SD at matching tau_quantile over 5 seeds',
              'selection':'no threshold selection for plots; test-sweep operating points',
              'warning':'Descriptive offline curve; no prospective threshold guarantee'}
    (args.output_dir/'F5_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Complete: {len(groups)} figures in {args.output_dir}')

if __name__=='__main__':
    main()
