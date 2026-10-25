from pathlib import Path
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Concept Drift Lab | Team 42', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
ROOT = next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / 'results').is_dir()), Path(__file__).resolve().parents[1])
RESULTS = ROOT / 'results'
FINAL = RESULTS / 'tables/adaptation/pgs242_128_five_seed_final'

@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)

def load(path):
    path = Path(path)
    if not path.is_file():
        st.warning(f'File unavailable: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}')
        return pd.DataFrame()
    try:
        return read_csv(str(path))
    except Exception as exc:
        st.error(f'Cannot read {path.name}: {exc}')
        return pd.DataFrame()

def region_control(key):
    return st.selectbox('AEMO region', ['SA1', 'NSW1'], key=key)

def table(df, name):
    if df.empty:
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button('Download table (CSV)', df.to_csv(index=False).encode(), file_name=name+'.csv', mime='text/csv', key='download_'+name)

def show_image(path, caption=None):
    if path.is_file():
        st.image(str(path), caption=caption or path.name, width="stretch")
    else:
        st.info(f'Figure not available: {path.name}')

def filter_region(df, region):
    return df[df['region'].astype(str).str.upper() == region].copy() if 'region' in df else df.copy()

def choice(df, col, title, key):
    vals = sorted(df[col].dropna().astype(str).unique().tolist()) if col in df else []
    return st.selectbox(title, vals, key=key) if vals else None

def number_chart(df, x, y, title, color=None):
    if df.empty or x not in df or y not in df:
        return
    fig = px.bar(df, x=x, y=y, color=color if color in df else None, title=title)
    fig.update_layout(height=440, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""<style>
:root { --navy: #244766; --blue: #668fb4; --teal: #6eafad; }
html, body, [data-testid="stApp"], .stApp,
[data-testid="stAppViewContainer"] { background: #f4f7fa !important; }
[data-testid="stMain"], section.main {
  background: #ffffff !important;
  box-sizing: border-box !important;
  border: 5px solid #a8c9df !important;
  border-radius: 24px !important;
  margin: 12px 16px 12px 8px !important;
  box-shadow: 0 5px 22px rgba(32,67,98,.09) !important;
  min-width: 0 !important;
}
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] .block-container,
section.main .block-container,
.main .block-container {
  background: #ffffff !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  width: 100% !important;
  margin: 0 auto !important;
  max-width: 1320px !important;
  min-height: 0 !important;
  padding: 44px 42px 68px !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] { background: #eef3f7 !important; }
[data-testid="stSidebar"] [role="radiogroup"] {
  display: flex !important;
  flex-direction: column !important;
  gap: 0 !important;
  background: #e5edf3 !important;
  border: 1px solid #d6e2eb !important;
  border-radius: 13px !important;
  overflow: hidden !important;
  padding: 6px 0 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  margin: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  padding: 9px 12px !important;
  background: transparent !important;
  box-shadow: none !important;
  min-height: 40px !important;
  transition: background .12s ease !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #d8e6f0 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: #cee0ed !important;
  box-shadow: inset 4px 0 0 #356c98 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:has(input[type="radio"]) {
  display: none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {
  font-size: .93rem !important;
  font-weight: 570 !important;
  color: #263f54 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
  color: #153b5b !important; font-weight: 720 !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #347e7b !important;
    border-radius: 8px !important;
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}
@media (max-width: 800px) {
  [data-testid="stMain"], section.main { margin: 7px !important; border-width: 6px !important; border-radius: 17px !important; }
  [data-testid="stMainBlockContainer"], [data-testid="stMain"] .block-container,
  section.main .block-container, .main .block-container {
    padding: 24px 15px 38px !important;
  }
}
</style>""", unsafe_allow_html=True)


def render_0():
    st.header('F1 · Forecast degradation')
    region = region_control('f1_region')
    kind = st.selectbox('Figure', ['forecasting_degradation', 'baseline_degradation'], key='f1_kind')
    show_image(RESULTS / f'figures/degradation/F1_{region}_{kind}.png')
    for filename in ['F1_rolling_mae.csv', 'F1_rolling_mae_baseline.csv', 'patchtst_synthetic_degradation_results.csv']:
        with st.expander(filename):
            df = load(RESULTS / 'tables/degradation' / filename)
            table(filter_region(df, region), 'f1_'+region+'_'+filename.replace('.csv',''))

def render_1():
    st.header('T1 · Synthetic drift detection')
    st.caption('Synthetic experiments are not region-specific; SA1/NSW1 selection does not apply.')
    df = load(RESULTS / 'tables/detection/T1_detection_on_synthetic.csv')
    if not df.empty:
        detector = choice(df, 'detector', 'Detector', 't1_detector')
        if detector:
            df = df[df['detector'].astype(str) == detector]
        table(df, 't1_synthetic')
    st.subheader('Synthetic detection figures')
    for detector in ['adwin', 'kswin', 'page_hinkley']:
        with st.expander(detector.upper()):
            show_image(RESULTS / f'figures/detection/{detector}_synthetic_evaluation_seed1.png')

def render_2():
    st.header('T2 / F2 · AEMO drift detection')
    region = region_control('t2_region')
    df = filter_region(load(RESULTS / 'tables/detection/T2_aemo_detection.csv'), region)
    if not df.empty:
        detector = choice(df, 'detector', 'Detector', 't2_detector')
        selected = df[df['detector'].astype(str) == detector] if detector else df
        if len(selected):
            r = selected.iloc[0]
            a,b,c,d = st.columns(4)
            a.metric('Detections', int(r['detections']))
            b.metric('Tier-1 matches', int(r['tier1_matched']))
            c.metric('Tier-1 precision', f"{float(r['tier1_precision']):.1%}")
            d.metric('Unmatched', int(r['unmatched']))
        number_chart(df, 'detector', 'detections', 'Detections by detector')
        table(df, 't2_'+region)
        st.subheader('F2 · Individual detector panel')
        if detector:
            show_image(RESULTS / f'figures/detection/F2_individual/F2_{region}_{detector}_.png')
    st.subheader('F2 · Combined detection panel')
    show_image(RESULTS / f'figures/detection/F2_detection_panel_{region}.png')
    for name in ['T2_detection_summary.csv','T2_detection_matches.csv','T2_tier1_event_details.csv']:
        with st.expander(name):
            table(filter_region(load(RESULTS / 'tables/detection' / name), region), 't2_'+region+'_'+name[:-4])

def render_3():
    st.header('T3 · Forecast accuracy by regime')
    region = region_control('t3_region')
    regime = st.radio('Regime evaluation', ['Full', 'Local'], horizontal=True, key='t3_regime')
    base = RESULTS / 'tables/adaptation/t3'
    tag = regime.lower()
    # The revised T3 outputs were moved to results/tables/adaptation/t3/.
    # Load the current seed-level evaluation as the source of truth.
    seed_candidates = [base / f't3_seed_metrics_{tag}.csv',
                       base / 't3_full_and_local' / f't3_seed_metrics_{tag}.csv']
    seed_path = next((p for p in seed_candidates if p.is_file()), None)
    if seed_path is None:
        st.warning(f'Updated T3 {regime.lower()} evaluation not found under results/tables/adaptation/t3/.')
        return
    raw = filter_region(load(seed_path), region)
    if raw.empty:
        st.info('No rows for the selected region.')
        return
    if 'configuration' in raw.columns:
        opts = sorted(raw['configuration'].dropna().astype(str).unique())
        selected = st.multiselect('Adaptation configurations', opts, default=opts, key='t3_configs')
        raw = raw[raw['configuration'].astype(str).isin(selected)]
    mae_cols = [f'mae_{r}' for r in ('pre-drift','drift','post-drift')]
    available = [c for c in mae_cols if c in raw.columns]
    if available and {'configuration','seed'}.issubset(raw.columns):
        rows = []
        for cfg, group in raw.groupby('configuration', sort=True):
            item = {'configuration': cfg, 'seeds': int(group['seed'].nunique())}
            for metric in available + (['pinball_loss'] if 'pinball_loss' in group else []):
                nums = pd.to_numeric(group[metric], errors='coerce').dropna()
                item[metric] = (f'{nums.mean():.2f} ± {nums.std(ddof=1):.2f}'
                                if len(nums)>1 else (f'{nums.iloc[0]:.2f}' if len(nums) else '—'))
            rows.append(item)
        summary = pd.DataFrame(rows)
        st.subheader('Five-seed forecast accuracy')
        table(summary, f't3_{tag}_{region}_five_seed_summary')
        plot = raw.melt(id_vars=['configuration','seed'], value_vars=available,
                        var_name='regime', value_name='MAE (MW)')
        plot['regime'] = plot['regime'].str.removeprefix('mae_')
        plot['MAE (MW)'] = pd.to_numeric(plot['MAE (MW)'], errors='coerce')
        plot = plot.groupby(['configuration', 'regime'], as_index=False)['MAE (MW)'].mean()
        fig = px.bar(plot, x='configuration', y='MAE (MW)', color='regime',
                     barmode='group', title='Forecast error by drift regime',
                     category_orders={'regime':['pre-drift','drift','post-drift']})
        fig.update_layout(height=460, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Unexpected T3 seed-level schema; showing the source without aggregation.')
    with st.expander('T3 · Source data by seed'):
        table(raw, f't3_{tag}_{region}_seed_metrics')
    count_path = base / 'regime_observation_counts.csv'
    if count_path.is_file():
        with st.expander('Regime observation counts'):
            table(filter_region(load(count_path),region), f't3_{region}_regime_counts')

def render_4():
    st.header('T4 · Adaptation cost and forecasting performance')
    region = region_control('t4_region')
    detail = filter_region(load(RESULTS / 'tables/adaptation/t4/t4_seed_level_metrics.csv'), region)
    if detail.empty:
        return

    def column(*names):
        return next((name for name in names if name in detail.columns), None)

    cfg_col = column('configuration', 'config')
    seed_col = column('seed')
    if cfg_col is None or seed_col is None:
        st.warning('T4 seed-level file does not contain configuration and seed columns.')
        table(detail, 't4_source_' + region)
        return

    # A and B are controls; C and D use the selected drift detector.
    detector = st.selectbox('Drift detector', ['adwin', 'kswin', 'page_hinkley'], key='t4_detector')
    config_values = detail[cfg_col].astype(str)
    controls = config_values.isin(['A', 'B', 'never', 'every_30_days'])
    detector_rows = config_values.str.lower().str.contains(detector, regex=False)
    detail = detail[controls | detector_rows].copy()

    def arm(config):
        s = str(config).lower()
        if s == 'a' or s == 'never' or s.startswith('a_'):
            return 'A · Never retrain'
        if s == 'b' or s == 'every_30_days' or s.startswith('b_'):
            return 'B · Every 30 days'
        if s.startswith('c_') or s.startswith('drift_full_history'):
            return 'C · Drift / full history'
        if s.startswith('d_') or s.startswith('drift_recent_window'):
            return 'D · Drift / recent window'
        return str(config)

    detail['Arm'] = detail[cfg_col].map(arm)
    arm_order = ['A · Never retrain', 'B · Every 30 days',
                 'C · Drift / full history', 'D · Drift / recent window']
    colors = {'A · Never retrain': '#6b7280', 'B · Every 30 days': '#a0aab5',
              'C · Drift / full history': '#277d99', 'D · Drift / recent window': '#c47f43'}


    d_configs = sorted(detail.loc[detail['Arm'] == 'D · Drift / recent window', cfg_col].astype(str).unique())
    if len(d_configs) > 1:
        preferred = next((x for x in d_configs if '180d' in x), d_configs[0])
        chosen_d = st.selectbox('Arm D window', d_configs,
                                index=d_configs.index(preferred), key='t4_d_window')
        detail = detail[(detail['Arm'] != 'D · Drift / recent window') |
                        (detail[cfg_col].astype(str) == chosen_d)].copy()

    present = [x for x in arm_order if x in set(detail['Arm'])]
    selected_arms = st.multiselect('Compare adaptation arms', present,
                                   default=present, key='t4_arms')
    detail = detail[detail['Arm'].isin(selected_arms)].copy()
    if detail.empty:
        st.info('Select at least one adaptation arm.')
        return

    metric_candidates = {
        'Overall MAE (MW)': ('overall_mae', 'mae', 'test_mae'),
        'Drift MAE (MW)': ('mae_drift', 'drift_mae'),
        'MAE gain vs Arm A (MW)': ('mae_gain_vs_arm_a', 'mae_gain_vs_a', 'mae_gain_vs_A'),
        'Number of retrains': ('n_retrains', 'retrains'),
        'Cumulative training samples': ('train_samples_total', 'cumulative_train_samples', 'train_samples'),
        'Retraining runtime (s)': ('retraining_wall_clock_s', 'wall_clock_s', 'runtime_s'),
    }
    metrics = {label: column(*candidates) for label, candidates in metric_candidates.items()}
    metrics = {label: key for label, key in metrics.items() if key is not None}
    perf = [x for x in ['Overall MAE (MW)', 'Drift MAE (MW)', 'MAE gain vs Arm A (MW)'] if x in metrics]
    cost = [x for x in ['Number of retrains', 'Cumulative training samples', 'Retraining runtime (s)'] if x in metrics]
    if not perf or not cost:
        st.warning('The updated T4 file is missing the MAE or retraining-cost metrics needed for these charts.')
        table(detail, 't4_seed_metrics_' + region)
        return

    st.subheader('Five-seed accuracy and computational cost')
    left, right = st.columns(2)
    with left:
        performance = st.selectbox('Forecasting performance', perf,
                                   index=perf.index('Overall MAE (MW)') if 'Overall MAE (MW)' in perf else 0,
                                   key='t4_performance')
    with right:
        expense = st.selectbox('Retraining cost', cost, key='t4_expense')

    def plot_metric(label):
        metric = metrics[label]
        values = detail[['Arm', seed_col, metric]].copy()
        values[metric] = pd.to_numeric(values[metric], errors='coerce')
        values = values.dropna(subset=[metric])
        # Aggregate one value per arm/seed before computing across-seed SD.
        by_seed = values.groupby(['Arm', seed_col], as_index=False)[metric].mean()
        summary = by_seed.groupby('Arm', as_index=False)[metric].agg(['mean', 'std', 'count']).reset_index()
        summary = summary.set_index('Arm').reindex([x for x in arm_order if x in set(summary['Arm'])]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=summary['Arm'], y=summary['mean'],
            marker_color=[colors[x] for x in summary['Arm']],
            error_y=dict(type='data', array=summary['std'].fillna(0), visible=True),
            text=summary['mean'].map(lambda x: f'{x:,.1f}'), textposition='outside',
            customdata=summary[['count']].to_numpy(),
            hovertemplate='%{x}<br>Mean: %{y:,.2f}<br>Seeds: %{customdata[0]}<extra></extra>',
        ))
        if 'gain' in label.lower():
            fig.add_hline(y=0, line_dash='dash', line_color='#64748b')
        fig.update_layout(title=label, height=430, plot_bgcolor='white',
                          paper_bgcolor='white', font_color='#244766',
                          xaxis_title=None, yaxis_title=label,
                          margin=dict(t=65, b=90), showlegend=False)
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        return summary

    with left:
        performance_summary = plot_metric(performance)
    with right:
        expense_summary = plot_metric(expense)

    st.caption('Bars show five-seed means ± standard deviation. Lower MAE and lower retraining cost are preferable. Arms A and B are controls; C and D are drift-triggered strategies.')

    # Side-by-side numbers make the comparisons legible during the demo.
    st.subheader('Comparison across the four arms')
    numeric = detail[['Arm', seed_col] + list(dict.fromkeys(metrics.values()))].copy()
    for name in metrics.values():
        numeric[name] = pd.to_numeric(numeric[name], errors='coerce')
    grouped = numeric.groupby('Arm')
    summary_rows = []
    for name in arm_order:
        if name not in grouped.groups:
            continue
        group = grouped.get_group(name)
        row = {'Arm': name, 'Seeds': group[seed_col].nunique()}
        for label, metric in metrics.items():
            per_seed = group.groupby(seed_col)[metric].mean().dropna()
            if len(per_seed):
                sd = per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0
                row[label] = f'{per_seed.mean():,.2f} ± {sd:,.2f}'
            else:
                row[label] = '—'
        summary_rows.append(row)
    table(pd.DataFrame(summary_rows), f't4_arm_comparison_{region}_{detector}')

    with st.expander('Original T4 five-seed summary'):
        official_path = RESULTS / 'tables/adaptation/t4/t4_adaptation_cost.csv'
        if official_path.is_file():
            table(filter_region(load(official_path), region), 't4_official_' + region)
    with st.expander('Individual seed-level results'):
        seeds = sorted(detail[seed_col].dropna().unique().tolist())
        seed = st.selectbox('Seed', seeds, key='t4_seed')
        table(detail[detail[seed_col] == seed], 't4_seed_' + region + '_' + str(seed))

def render_5():
    st.header('F3 · Four-arm comparison')
    region = region_control('f3_region')
    detector = st.selectbox('Detector', ['adwin','kswin','page_hinkley'], key='f3_detector')
    show_image(RESULTS / f'figures/adaptation/F3_four_arms/F3_{region}_four_arm_comparison_{detector}_180d.png')
    st.caption('F3 · Saved four-arm comparison figure from the experiment repository.')
    df = filter_region(load(RESULTS / 'tables/adaptation/revised/pgs242_128_five_seed_final/summary_all_seeds.csv'), region)
    if not df.empty:
        df = df[(df['arm'].isin(['A','B'])) | (df['detector'].astype(str)==detector)]
        selected_configs = ['A', 'B', f'C_{detector}', f'D_180d_{detector}']
        comparison = df[df['configuration'].isin(selected_configs)].copy()

        summary = comparison.groupby('configuration').agg(
            seeds=('seed', 'nunique'),
            mae_mean=('overall_mae', 'mean'),
            mae_sd=('overall_mae', 'std'),
            retrains_mean=('n_retrains', 'mean'),
        ).reindex(selected_configs)

        summary.index.name = 'Configuration'
        summary = summary.rename(columns={
            'seeds': 'Seeds available',
            'mae_mean': 'Mean overall MAE (MW)',
            'mae_sd': 'MAE SD (MW)',
            'retrains_mean': 'Mean retrains',
        })

        st.subheader('Four-arm summary · matching the figure')
        st.dataframe(summary.round(2), use_container_width=True)
        st.caption(
            'Overall test-period MAE: mean and sample SD across available seeds. '
            'Lower MAE means smaller prediction errors. '
            'The figure shows 7-day rolling MAE. Arm D uses 180 days here. '
            'Other recent-window configurations remain in the detailed table below.'
        )

        st.subheader('Five-seed experiment rows for selected detector')
        table(df, 'f3_five_seed_'+region+'_'+detector)

def render_6():
    st.header('Interactive actual vs forecast')
    region = region_control('forecast_region')
    seed = st.selectbox('Seed', [1, 2, 3, 4, 5], key='forecast_seed')
    # Forecasts can live under different adaptation result revisions.
    candidates = [FINAL / f'seed_{seed}' / f'forecasts_{region}.csv.gz']
    candidates += sorted((RESULTS / 'tables/adaptation').rglob(f'seed_{seed}/forecasts_{region}.csv.gz'))
    candidates += sorted((RESULTS / 'tables/adaptation').rglob(f'seed_{seed}/forecasts_{region}.csv'))
    # The existing AgACI interval data also contain the same point forecasts.
    fallback = (RESULTS / 'tables/uncertainty/agaci_boa_fixed_calibration'
                / f'seed_{seed}' / f'agaci_comparison_intervals_{region}.csv.gz')
    candidates.append(fallback)
    path = next((x for x in candidates if x.is_file() and 'archive' not in x.parts), None)
    if path is None:
        st.warning('Forecast data unavailable: no seed-level forecast or AgACI comparison file found.')
        return

    @st.cache_data(show_spinner='Loading forecast data…')
    def load_forecasts(p):
        data = pd.read_csv(p)
        if 'y_pred' not in data and 'point_forecast' in data:
            data = data.rename(columns={'point_forecast': 'y_pred'})
        data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], errors='coerce')
        return data.dropna(subset=['SETTLEMENTDATE'])

    df = load_forecasts(str(path))
    required = {'configuration', 'SETTLEMENTDATE', 'y_true', 'y_pred'}
    if not required.issubset(df.columns):
        st.error(f'Forecast file missing columns: {sorted(required - set(df.columns))}')
        return
    if path == fallback:
        st.caption('Point forecasts sourced from the saved five-seed uncertainty interval outputs.')
    config = choice(df, 'configuration', 'Adaptation configuration', 'forecast_config')
    if not config:
        return
    df = df[df['configuration'].astype(str) == config].sort_values('SETTLEMENTDATE').copy()
    if df.empty:
        return
    low, high = df['SETTLEMENTDATE'].min().date(), df['SETTLEMENTDATE'].max().date()
    dates = st.date_input('Date range', value=(low, min(low + pd.Timedelta(days=14), high)),
                          min_value=low, max_value=high, key='forecast_dates')
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info('Choose a start and end date.')
        return
    start, end = dates
    view = df[(df['SETTLEMENTDATE'].dt.date >= start) &
              (df['SETTLEMENTDATE'].dt.date <= end)].copy()
    if view.empty:
        st.info('No forecasts in selected dates.')
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'], y=view['y_true'], name='Actual demand', mode='lines'))
    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'], y=view['y_pred'], name='Forecast', mode='lines'))
    if 'retrain_indicator' in view:
        mask = view['retrain_indicator'].astype(str).str.lower().isin(['true', '1'])
        retrains = view[mask]
        if len(retrains):
            fig.add_trace(go.Scatter(x=retrains['SETTLEMENTDATE'], y=retrains['y_pred'],
                                     name='Retraining', mode='markers', marker={'symbol':'diamond','size':10}))
    fig.update_layout(height=550, xaxis_title='Settlement date', yaxis_title='Electricity demand (MW)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    st.metric('Selected period MAE', f"{(view['y_true'] - view['y_pred']).abs().mean():.2f} MW")
    st.caption('Period MAE is calculated from selected rows, not the official full-test T3 value.')
    table(view, f'forecast_{region}_seed{seed}_{config}')

def _t5_directories():
    """Main selector: real AEMO evaluation with all three UQ methods only.

    Exclude legacy t5_aemo and simulation folders; simulations have their
    own browser below the main T5 evaluation.
    """
    base = RESULTS / 'tables/uncertainty'
    if not base.exists():
        return []
    directories = set()
    for path in base.rglob('T5_SA1_*.csv'):
        rel = str(path.parent.relative_to(base)).lower()
        if any(x in rel for x in ('archive', 'simulation', 't5_aemo')):
            continue
        try:
            head = pd.read_csv(path, usecols=['UQ method'])
            methods = head['UQ method'].astype(str).str.lower()
            if (methods.str.contains('agaci').any() and
                methods.str.contains('static').any() and
                methods.str.contains('adaptive|aci', regex=True).any()):
                directories.add(path.parent)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(directories, key=lambda p: ('post7' not in str(p).lower(), str(p)))


def render_t5_simulation_browser():
    """Browse simulation inputs/results without mixing them with real AEMO T5."""
    base = RESULTS / 'tables/uncertainty/t5_simulation'
    if not base.is_dir():
        st.info('No T5 simulation results found under results/tables/uncertainty/t5_simulation/.')
        return
    files = sorted(p for p in base.rglob('*') if p.is_file() and
                   p.suffix.lower() in ('.csv', '.png', '.json', '.md') and 'archive' not in p.parts)
    if not files:
        st.info('No simulation result files found.')
        return
    selected = st.selectbox('Simulation output', files,
                            format_func=lambda p: str(p.relative_to(base)),
                            key='t5_simulation_file')
    if selected.suffix == '.csv':
        data = load(selected)
        region = st.selectbox('Region filter', ['All', 'SA1', 'NSW1'], key='t5_simulation_region')
        if region != 'All':
            data = filter_region(data, region)
        table(data, 't5_simulation_' + selected.stem)
    elif selected.suffix == '.png':
        show_image(selected)
    else:
        st.code(selected.read_text(encoding='utf-8', errors='replace')[:50000],
                language='json' if selected.suffix == '.json' else 'markdown')

def _parse_mean_sd(values):
    parts = values.astype(str).str.split('±', n=1, expand=True)
    avg = pd.to_numeric(parts[0].str.strip().str.replace('%','',regex=False), errors='coerce')
    dev = (pd.to_numeric(parts[1].str.strip().str.replace('pp','',regex=False), errors='coerce')
           if parts.shape[1]>1 else pd.Series(0., index=values.index))
    return avg, dev.fillna(0.)


def _render_t5_main():
    st.header('T5 · Uncertainty coverage and interval width')
    directories = _t5_directories()
    if not directories:
        st.warning('No real-data T5 tables containing Static, ACI and AgACI were found. Browse the simulation in the adjacent tab.')
        return
    a,b = st.columns([1,2])
    with a: region = region_control('t5_region_updated')
    with b:
        folder = st.selectbox('T5 evaluation dataset', directories,
            format_func=lambda x: str(x.relative_to(RESULTS)), key='t5_dataset_updated',
            help='Prefer the newest Static / ACI / AgACI-BOA five-seed evaluation.')
    files = sorted(folder.glob(f'T5_{region}_*.csv'))
    if not files:
        st.warning(f'No T5 tables for {region} in the selected experiment.')
        return
    configurations = {x.stem[len(f'T5_{region}_'):]:x for x in files}
    preferred = next((x for x in configurations if x=='A'), next(iter(configurations)))
    config = st.selectbox('Adaptation configuration', list(configurations),
        index=list(configurations).index(preferred), key='t5_config_updated')
    original = load(configurations[config])
    necessary=['UQ method','regime','coverage','mean width','normalised width','worst 24h coverage']
    if original.empty or not set(necessary).issubset(original):
        st.error('Selected T5 table is missing expected columns.'); return
    methods=original['UQ method'].dropna().unique().tolist()
    if not any('agaci' in str(m).lower() for m in methods):
        st.warning('This is an older T5 dataset: AgACI-BOA is NOT included. Select the three-method AgACI evaluation above.')
    selected=st.multiselect('Uncertainty methods',methods,default=methods,key='t5_methods_updated')
    data=original[original['UQ method'].isin(selected)].copy()
    if data.empty: st.info('Choose at least one method.');return
    for metric in necessary[2:]:
        data[metric],data[metric+'_sd']=_parse_mean_sd(data[metric])
    regimes=['pre-drift','drift','post-drift']
    data['regime']=pd.Categorical(data['regime'],categories=regimes,ordered=True)
    data=data.sort_values(['regime','UQ method'])
    drift=data[data['regime']=='drift']
    if not drift.empty:
        cols=st.columns(3)
        cols[0].metric('Drift coverage · method average', f"{100*drift['coverage'].mean():.2f}%")
        cols[1].metric('Drift mean width · method average', f"{drift['mean width'].mean():.1f} MW")
        worst=drift['worst 24h coverage'].dropna()
        cols[2].metric('Worst 24h coverage · minimum method', f"{100*worst.min():.2f}%" if len(worst) else 'N/A')
    colors={'static':'#4a647f','adaptive':'#c47f43','agaci':'#248174'}  # Match F4: Static, ACI, AgACI-BOA
    def colour(method):
        name=str(method).lower()
        return next((value for keyword,value in colors.items() if keyword in name),'#447d88')
    def chart(metric,title,percentage=False,target=False):
        fig=go.Figure()
        for method in selected:
            rows=data[data['UQ method']==method].sort_values('regime')
            factor=100 if percentage else 1
            fig.add_trace(go.Bar(x=rows['regime'].astype(str),y=rows[metric]*factor,
                name=str(method),marker_color=colour(method),
                error_y=dict(type='data',array=rows[metric+'_sd']*factor,visible=True),
                hovertemplate='%{x}<br>%{y:.2f}'+('%' if percentage else '')+'<extra>%{fullData.name}</extra>'))
        if target: fig.add_hline(y=90,line_dash='dash',line_color='#a45555',annotation_text='90% nominal target')
        fig.update_layout(title=title,barmode='group',height=380,legend=dict(orientation='h',y=-.23),
            plot_bgcolor='white',margin=dict(t=60,b=85),yaxis_title='Coverage (%)' if percentage else title)
        fig.update_xaxes(categoryorder='array',categoryarray=regimes)
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Coverage and reliability · five-seed mean ± SD')
    chart('coverage','Empirical coverage',percentage=True,target=True)
    chart('worst 24h coverage','Worst 24-hour coverage',percentage=True)
    x,y=st.columns(2)
    with x: chart('mean width','Mean prediction-interval width (MW)')
    with y: chart('normalised width','Normalised interval width')
    st.caption('Values come directly from the selected T5 five-seed tables. Compare coverage together with interval width; the 90% line is a nominal target, not a guarantee.')
    st.subheader('T5 · Original results')
    table(original, f't5_{region}_{config}_updated')

def render_7():
    main, simulation = st.tabs(['T5 · AEMO three-method evaluation', 'Browse T5 simulation'])
    with main:
        _render_t5_main()
    with simulation:
        render_t5_simulation_browser()


def render_8():
    region = region_control('gallery_region')
    figures_tab, tables_tab = st.tabs(['Figures', 'Tables'])

    with figures_tab:
        paths = sorted((RESULTS / 'figures').rglob('*.png'))
        paths = [p for p in paths if 'archive' not in p.parts and
                 (region in p.name or not any(r in p.name for r in ['SA1', 'NSW1']))]
        category = st.selectbox('Figure category',
            ['All', 'degradation', 'detection', 'adaptation', 'uncertainty', 'diagnosis'],
            key='gallery_category')
        if category != 'All':
            paths = [p for p in paths if category in p.parts]
        if paths:
            selected = st.selectbox('Saved figure', range(len(paths)),
                format_func=lambda i: str(paths[i].relative_to(RESULTS)),
                key='gallery_figure')
            show_image(paths[selected])
        else:
            st.info('No figures are available for this selection.')

    with tables_tab:
        # Discover repository tables rather than maintaining a manually curated list.
        # Also include CSV result tables stored alongside figures.
        table_paths = sorted({p for folder in (RESULTS / 'tables', RESULTS / 'figures')
            if folder.is_dir() for pattern in ('*.csv', '*.csv.gz')
            for p in folder.rglob(pattern) if 'archive' not in p.parts})
        table_paths = [p for p in table_paths if
            region in p.name or not any(r in p.name for r in ('SA1', 'NSW1'))]
        categories = sorted({p.relative_to(RESULTS).parts[1]
            for p in table_paths if len(p.relative_to(RESULTS).parts) > 2})
        table_category = st.selectbox('Table category', ['All'] + categories,
            key='gallery_table_category')
        if table_category != 'All':
            table_paths = [p for p in table_paths if
                len(p.relative_to(RESULTS).parts) > 2 and
                p.relative_to(RESULTS).parts[1] == table_category]
        if not table_paths:
            st.info('No result tables are available for this selection.')
        else:
            selected_path = st.selectbox('Saved table', table_paths,
                format_func=lambda p: str(p.relative_to(RESULTS)),
                key='gallery_table_file')
            st.caption(f'Source: `results/{selected_path.relative_to(RESULTS)}`')
            try:
                # Preview a bounded number of rows so large seed-level output files
                # do not exhaust memory or delay the dashboard.
                preview = pd.read_csv(selected_path, nrows=500)
                st.dataframe(preview, use_container_width=True, hide_index=True)
                st.caption('Preview shows up to 500 rows from the selected repository table.')
                if selected_path.stat().st_size <= 20 * 1024 * 1024:
                    st.download_button('Download selected table',
                        data=selected_path.read_bytes(), file_name=selected_path.name,
                        mime='application/gzip' if selected_path.name.endswith('.gz') else 'text/csv',
                        key='gallery_table_download')
                else:
                    st.caption('Large source file: open the path above in the GitHub repository to obtain the complete data.')
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.warning(f'Unable to preview selected table: {exc}')



# Embedded escalation explorer: no additional local Python module is required.
REQUIRED = {"SETTLEMENTDATE", "region", "seed", "configuration", "y_true", "point_forecast"}
METHODS = {
    "Static split conformal": ("static_lower", "static_upper"),
    "Adaptive conformal (ACI)": ("aci_lower", "aci_upper"),
    "AgACI-BOA": ("agaci_lower", "agaci_upper"),
}


def _repo_root() -> Path:
    # Component normally copied into dashboard_components/ at repository root.
    return ROOT


@st.cache_data(show_spinner="Loading timestamp-level forecasts…", ttl=600)
def _load_intervals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"The interval data are missing columns: {sorted(missing)}")
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    df = df.dropna(subset=["SETTLEMENTDATE"]).sort_values("SETTLEMENTDATE")
    return df


def render_escalation_explorer(repo_root=None) -> None:
    """Render an independent Streamlit escalation section with interactive 5-day windows."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    interval_dir = root / "results/tables/uncertainty/agaci_boa_fixed_calibration"

    st.subheader("Five-day analysis of forecast review decisions")
    st.caption("Examine review prioritisation, prediction-interval coverage and retained forecasts across historical five-day evaluation windows.")

    a, b, c = st.columns([1, 1, 1])
    with a:
        region = st.selectbox("Region", ["SA1", "NSW1"], key="escalation_region")
    with b:
        seed = st.selectbox("Seed", [1, 2, 3, 4, 5], key="escalation_seed")
    with c:
        method = st.selectbox("Uncertainty method", list(METHODS), key="escalation_uq")

    path = interval_dir / f"seed_{seed}" / f"agaci_comparison_intervals_{region}.csv.gz"
    if not path.is_file():
        st.warning(f"Timestamp-level intervals were not found: `{path.relative_to(root) if path.is_relative_to(root) else path}`. The aggregated F5 CSV cannot provide a five-day timeline; these interval files are required.")
        return

    try:
        df = _load_intervals(str(path))
    except (OSError, ValueError) as exc:
        st.error(f"Cannot load interval data: {exc}")
        return

    configurations = sorted(df["configuration"].dropna().astype(str).unique())
    if not configurations:
        st.warning("No configurations found in interval data.")
        return
    configuration = st.selectbox("Adaptation configuration", configurations, key="escalation_configuration")
    df = df[df["configuration"].astype(str) == configuration].copy()
    lower_col, upper_col = METHODS[method]
    if not {lower_col, upper_col}.issubset(df.columns):
        st.error(f"Missing interval bounds for {method}: {lower_col}, {upper_col}")
        return

    # Calculate scores BEFORE slicing the viewing window; the first timestamp in
    # each window therefore uses the preceding observation when available.
    df = df.sort_values("SETTLEMENTDATE").drop_duplicates("SETTLEMENTDATE", keep="last").reset_index(drop=True)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["point_forecast"] = pd.to_numeric(df["point_forecast"], errors="coerce")
    df["lower"] = pd.to_numeric(df[lower_col], errors="coerce")
    df["upper"] = pd.to_numeric(df[upper_col], errors="coerce")
    df["width"] = df["upper"] - df["lower"]
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["lag1_error"] = (df["y_true"] - df["point_forecast"]).abs().shift(1)
    df["lag1_error"] = df["lag1_error"].fillna(0.0)
    df = df.dropna(subset=["y_true", "point_forecast", "lower", "upper"])
    if df.empty:
        st.warning("No valid observations after selecting the configuration and UQ method.")
        return

    score_name = st.radio("Review prioritisation signal", ["Previous 30-minute absolute error", "Current interval width"], horizontal=True, key="escalation_score")
    if score_name.startswith("Previous"):
        df["review_score"] = df["lag1_error"]
        st.caption("Causal score: at time t, the previous observed half-hour's absolute forecast error (t−1).")
    else:
        df["review_score"] = df["width"]
        st.caption("Interval-width score. Static conformal widths may be constant, so ties can limit prioritisation.")

    valid_scores = df["review_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_scores.empty:
        st.warning("There are no valid review-prioritisation scores.")
        return
    max_score = float(valid_scores.max())
    default_tau = float(valid_scores.quantile(0.8))
    threshold = st.slider(
        "Explore escalation threshold τ (review points whose score > τ)",
        min_value=0.0,
        max_value=max(max_score, 0.000001),
        value=min(default_tau, max(max_score, 0.000001)),
        key=f"escalation_threshold_{region}_{seed}_{configuration}_{score_name}",
        help="Interactive exploratory threshold, not a calibrated operating point. For final T6 claims, use your frozen evaluation thresholds.",
    )
    df["review"] = df["review_score"] > threshold

    # Windows are contiguous five-calendar-day blocks, with explicit navigation.
    start_day = df["SETTLEMENTDATE"].min().normalize()
    last_day = df["SETTLEMENTDATE"].max().normalize()
    max_offset = max(0, int((last_day - start_day).days) - 4)
    slider_key = f"esc_window_slider_{region}_{seed}_{configuration}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 0
    st.session_state[slider_key] = max(0, min(int(st.session_state[slider_key]), max_offset))

    p, offset_col, n = st.columns([1, 5, 1])
    with p:
        if st.button("◀ Previous 5d", key="esc_prev", disabled=st.session_state[slider_key] == 0):
            st.session_state[slider_key] = max(0, st.session_state[slider_key] - 5)
    with n:
        if st.button("Next 5d ▶", key="esc_next", disabled=st.session_state[slider_key] >= max_offset):
            st.session_state[slider_key] = min(max_offset, st.session_state[slider_key] + 5)
    with offset_col:
        st.slider("Window start (days after first test date)", 0, max_offset, key=slider_key)

    view_start = start_day + pd.Timedelta(days=int(st.session_state[slider_key]))
    view_end = view_start + pd.Timedelta(days=5)
    part = df[(df["SETTLEMENTDATE"] >= view_start) & (df["SETTLEMENTDATE"] < view_end)].copy()
    st.markdown(f"**Window:** {view_start:%d %b %Y} – {(view_end - pd.Timedelta(seconds=1)):%d %b %Y} · **{len(part):,} observations**")
    if part.empty:
        st.info("This five-day window has no observations. Move the slider to another date.")
        return

    retained = part[~part["review"]]
    review_rate = float(part["review"].mean())
    base_cov = float(part["covered"].mean())
    retained_cov = float(retained["covered"].mean()) if len(retained) else float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Human review", f"{review_rate:.1%}", f"{int(part['review'].sum()):,} forecasts")
    m2.metric("Before review coverage", f"{base_cov:.1%}")
    m3.metric("Retained coverage", f"{retained_cov:.1%}" if np.isfinite(retained_cov) else "Undefined")
    m4.metric("Coverage target", "90%", "Met" if np.isfinite(retained_cov) and retained_cov >= 0.9 else "Not met")

    # Figure U: forecast plus 90% interval, misses, grouped human-review stretches;
    # and a second panel for rolling coverage. Five-day window = 240 half-hours.
    rolling_n = st.select_slider('Rolling coverage window (half-hour observations)',
                                 options=[24, 48, 96, 240, 336], value=48, key='esc_roll_n')
    df['rolling_coverage'] = df['covered'].astype(float).rolling(rolling_n, min_periods=rolling_n).mean()
    part = df[(df['SETTLEMENTDATE'] >= view_start) & (df['SETTLEMENTDATE'] < view_end)].copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.075,
                        row_heights=[0.73, 0.27])
    x = part['SETTLEMENTDATE']
    fig.add_trace(go.Scatter(x=x, y=part['upper'], mode='lines', showlegend=False,
                             line=dict(width=0), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=part['lower'], mode='lines', name='90% prediction interval',
                             fill='tonexty', fillcolor='rgba(45,122,167,.16)',
                             line=dict(width=0), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=part['y_true'], name='Actual demand', mode='lines',
                             line=dict(color='#23394b', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=part['point_forecast'], name='Forecast', mode='lines',
                             line=dict(color='#1684a1', width=1.8)), row=1, col=1)
    misses = part[~part['covered']]
    if not misses.empty:
        fig.add_trace(go.Scatter(x=misses['SETTLEMENTDATE'], y=misses['y_true'],
                                 mode='markers', name='Outside prediction interval',
                                 marker=dict(symbol='circle-open', color='#bf4c51', size=10, line=dict(width=2))), row=1, col=1)
    # Shade consecutive reviewed runs. This is not a claim that the interval itself widens.
    review = part['review'].to_numpy(bool)
    timestamps = part['SETTLEMENTDATE'].reset_index(drop=True)
    run_start = None
    for i, active in enumerate(review):
        if active and run_start is None: run_start = i
        if run_start is not None and (not active or i == len(review)-1):
            stop = i if not active else i+1
            end_at = timestamps.iloc[stop] if stop < len(timestamps) else timestamps.iloc[-1] + pd.Timedelta(minutes=30)
            fig.add_vrect(x0=timestamps.iloc[run_start], x1=end_at, fillcolor='#efb36a',
                          opacity=.17, line_width=0, row=1, col=1)
            run_start = None
    reviewed = part[part['review']]
    if not reviewed.empty:
        fig.add_trace(go.Scatter(x=reviewed['SETTLEMENTDATE'], y=reviewed['point_forecast'],
                                 mode='markers', name='Sent for review',
                                 marker=dict(color='#bb762a', symbol='diamond', size=6, opacity=.85)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=part['rolling_coverage']*100, name='Rolling coverage',
                             mode='lines', line=dict(color='#c37d39', width=2)), row=2, col=1)
    fig.add_hline(y=90, line_color='#65717d', line_dash='dash', row=2, col=1)
    fig.update_yaxes(title_text='Demand (MW)', row=1, col=1)
    fig.update_yaxes(title_text='Coverage (%)', range=[0, 102], row=2, col=1)
    fig.update_xaxes(title_text='Settlement date', row=2, col=1)
    fig.update_layout(title='Figure U · Calibrated forecast and human escalation · five-day window',
                      height=670, hovermode='x unified', legend=dict(orientation='h', y=-.13),
                      margin=dict(l=30, r=20, t=65, b=90), plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Orange shading = timestamps flagged for review. Red circles = realised misses (evaluation only); '
               'the current observation error is never used to decide its own escalation. Rolling coverage uses '
               f'{rolling_n} half-hour observations and includes reviewed points for diagnostic evaluation.')
    if not np.isfinite(retained_cov):
        st.warning('All points were sent to human review; retained coverage is undefined.')
    elif retained_cov < .90:
        st.info('Retained coverage is below 90% in this window.')
    with st.expander('Inspect five-day decisions and threshold signal'):
        graph = go.Figure()
        graph.add_trace(go.Scatter(x=x, y=part['review_score'], name='Review signal', line=dict(color='#a66225')))
        graph.add_hline(y=threshold, line_dash='dash', line_color='#c44343', annotation_text='τ threshold')
        graph.update_layout(height=290, xaxis_title='Settlement date', yaxis_title='Score (MW)')
        st.plotly_chart(graph, use_container_width=True)
        table(part[['SETTLEMENTDATE','y_true','point_forecast','lower','upper','covered',
                    'review_score','review','rolling_coverage']],
              f'five_day_escalation_{region}_{seed}')
    st.caption('This is a retrospective, exploratory visualisation. Moving the window does not change the '
               'threshold. The dashboard does not simulate human corrections or claim operational calibration.')


def render_model_selection():
    section('Forecasting model selection · validation only')
    st.caption('Supervisor guidance: select a model using training-period validation; never choose it using the test stream. '
               'Gradient Boosting remains the fixed forecaster in the current demo for consistency and runtime.')
    region = region_control('ms_region')
    base = RESULTS / 'model_selection' / region
    for fname in ['validation_comparison.csv','validation_summary.csv']:
        path = base / fname
        if path.exists():
            st.subheader(fname.replace('_',' ').replace('.csv','').title())
            table(load(path), f'model_{region}_{fname[:-4]}')
    report = base / 'model_selection_report.md'
    if report.exists():
        with st.expander('Selection rationale and validation details'):
            st.markdown(report.read_text(encoding='utf-8'))
    st.info('These results document region-specific model-selection evidence; they do not replace GB in the current adaptation experiments.')

def render_f4():
    section('F4 · Rolling coverage trajectory')
    a,b=st.columns(2)
    with a: region=region_control('f4_region')
    with b: detector=st.selectbox('Detector', ['adwin','kswin','page_hinkley'],key='f4_detector')
    show_image(RESULTS / 'figures/uncertainty/f4'/f'F4_{region}_{detector}_four_arms.png')
    st.caption('Four arms, three UQ methods, five seeds. The 90% line represents nominal coverage. '
               'Event-regime annotations reflect the revised post-event buffer.')
    with st.expander('F4 underlying five-seed metrics'):
        p=RESULTS/'figures/uncertainty/f4/F4_coverage_width_mean_sd.csv'
        if p.exists(): table(filter_region(load(p),region),'f4_aggregate_'+region)

def render_f5():
    section('F5 · Human-review escalation trade-off')
    a,b,c=st.columns(3)
    with a: region=region_control('f5_region')
    with b: detector=st.selectbox('Detector',['adwin','kswin','page_hinkley'],key='f5_detector')
    with c: score=st.radio('Review score',['Interval width','Lagged nonconformity'],key='f5_score')
    folder='f5_width' if score=='Interval width' else 'f5_nonconformity'
    alternatives=[RESULTS/'figures/uncertainty'/folder,
                  RESULTS/'figures/uncertainty'/('f5_width_final' if score=='Interval width' else 'f5_nonconformity_final')]
    image=next((p/f'F5_{region}_{detector}.png' for p in alternatives if (p/f'F5_{region}_{detector}.png').exists()),alternatives[0]/f'F5_{region}_{detector}.png')
    show_image(image)
    st.caption('F5: review rate on the horizontal axis, retained coverage on the vertical axis. '
               'The earliest tested point reaching 90% is the reported minimum; an already adequate unreviewed '
               'stream requires 0% review. Missing targets are not assumed to be reachable.')
    base=image.parent
    with st.expander('F5 crossing points, figure audit and curve data'):
        for fname in ['F5_90pct_crossings.csv','F5_curve_points_five_seed.csv','F5_figure_audit.csv']:
            p=base/fname
            if p.exists():
                st.caption(fname)
                df=filter_region(load(p),region)
                table(df, 'f5_'+region+'_'+folder+'_'+fname[:-4])

def _t6_sources():
    """Discover completed T6 experiments, preserving score-specific provenance."""
    base=RESULTS/'tables'/'uncertainty'
    if not base.exists(): return []
    folders={p.parent for p in base.rglob('t6_aggregate_audit.csv') if 'archive' not in p.parts}
    folders |= {p.parent for p in base.rglob('t6_*_*.csv') if 'archive' not in p.parts}
    answer=[]
    for folder in sorted(folders):
        candidates=list(folder.glob('t6_SA1_*.csv'))+list(folder.glob('t6_NSW1_*.csv'))
        if not candidates: continue
        manifest=folder/'t6_manifest.json'
        desc=str(folder.relative_to(base)).lower()
        if manifest.exists():
            try:
                import json
                meta=json.loads(manifest.read_text())
                desc+=' '+str(meta.get('score','')).lower()+' '+str(meta.get('log_group','')).lower()
            except Exception: pass
        if any(term in desc for term in ['lag1','nonconformity','lagged']):
            kind='Lagged nonconformity (previous 30-minute error)'
        elif any(term in desc for term in ['width','interval','agaci','escalation','t6']):
            kind='Interval width'
        else: kind='Unclassified historical T6'
        answer.append((kind,folder))
    return answer


def render_t6():
    section('T6 · Human-review operating points')
    sources=_t6_sources()
    if not sources:
        st.warning('No T6 per-configuration tables found under results/tables/uncertainty.')
        st.caption('Expected separate T6 output folders for interval-width and lagged-nonconformity escalation.')
        return
    kinds=list(dict.fromkeys(x[0] for x in sources))
    a,b=st.columns(2)
    with a: region=region_control('t6_region_new')
    with b:
        preferred_signal='Lagged nonconformity (previous 30-minute error)'
        kind=st.selectbox('Escalation signal',kinds,
            index=kinds.index(preferred_signal) if preferred_signal in kinds else 0,
            key='t6_signal_new')
    matching=[folder for label,folder in sources if label==kind]
    # Select the final available results for this signal without a redundant experiment dropdown.
    folder=sorted(matching, key=lambda p: (
        'post7' not in str(p).lower(),
        't6_nonconformity_30min' not in str(p).lower() if kind.startswith('Lagged')
        else 't6_width' not in str(p).lower(),
        str(p)))[0]
    agg_path=folder/'t6_aggregate_audit.csv'
    if agg_path.exists():
        aggregate=filter_region(load(agg_path),region)
        configs=sorted(aggregate['configuration'].dropna().astype(str).unique().tolist()) if 'configuration' in aggregate else []
        methods=sorted(aggregate['uq_method'].dropna().astype(str).unique().tolist()) if 'uq_method' in aggregate else []
    else:
        aggregate=pd.DataFrame()
        files=list(folder.glob(f't6_{region}_*.csv'))
        # Parse from filenames only as a fallback for legacy folders.
        configs=sorted({f.stem[len('t6_'+region+'_'):].rsplit('_',2)[0] for f in files})
        methods=[]
    if not configs:
        st.warning('No T6 configurations for this region in this experiment.');return
    c1,c2=st.columns(2)
    with c1:
        preferred_config='D_180d_page_hinkley'
        config=st.selectbox('Adaptation configuration',configs,
            index=configs.index(preferred_config) if preferred_config in configs else 0,
            key='t6_config_new')
    with c2:
        if methods:
            preferred_method=next((m for m in methods if 'agaci' in m.lower()),None)
            method=st.selectbox('Uncertainty method',methods,
                index=methods.index(preferred_method) if preferred_method is not None else 0,
                key='t6_method_new')
        else: method=None
    st.caption('Operating points are measured on retained forecasts, without assuming a human corrected escalated predictions. Budget columns use the evaluated discrete thresholds.')
    if not aggregate.empty:
        data=aggregate[aggregate['configuration'].astype(str).eq(config)].copy()
        if method is not None: data=data[data['uq_method'].astype(str).eq(method)]
        if 'metric' in data and 'display' in data:
            order=['review rate for 0.90','review rate for 0.95','coverage @ 5%','coverage @ 10%','coverage @ 20%']
            pivot=data.pivot_table(index='regime',columns='metric',values='display',aggfunc='first')
            pivot=pivot.reindex(index=['pre-drift','drift','post-drift','all'],columns=order)
            st.subheader('T6 · Coverage targets and fixed review budgets')
            st.dataframe(pivot.reset_index(),use_container_width=True,hide_index=True)
            st.download_button('Download selected operating points',pivot.reset_index().to_csv(index=False),
                file_name=f'T6_{region}_{config}_{kind.split()[0]}.csv',mime='text/csv',key='t6_export_new')
            if 'drift' in pivot.index:
                items=st.columns(3)
                for col,label in zip(items,['review rate for 0.90','coverage @ 10%','coverage @ 20%']):
                    val=pivot.loc['drift',label]
                    col.metric(label, '—' if pd.isna(val) else str(val))
        with st.expander('Complete five-seed aggregate audit'):
            table(aggregate,f't6_audit_{region}_{kind.split()[0]}')
    else:
        files=sorted(folder.glob(f't6_{region}_{config}_*.csv'))
        if not files: st.warning('No T6 file for this selection.');return
        selected_file=st.selectbox('UQ method / results file',files,format_func=lambda x:x.stem,key='t6_file_new')
        table(load(selected_file),f't6_{region}_{config}')
    other={label for label,_ in sources if label!=kind}
    if other: st.caption('Switch the escalation signal above to compare the other protocol; do not interpret the two protocols as the same scoring rule.')

# ============================================================
# T7 — CROSS-TEAM TRIGGER COMPARISON (PROVISIONAL)
# ============================================================


# F6: compare accepted detector timestamps, never summary counts or synthetic alarms.
def render_f6(region):
    # Each region has its own two-row accepted-detection timeline.
    tolerance_h = 24
    highlight = True

    own_path = RESULTS / 'tables/detection/aemo_detection_changepoints.csv'
    if not own_path.is_file():
        st.warning('Team 42 timestamp-level detections are missing: results/tables/detection/aemo_detection_changepoints.csv')
        return

    # Load Team 41's timestamp-level Page-Hinkley outputs from the repository.
    # Their published CSV schema is timestamp, day, label; region is in the filename.
    cross_team = RESULTS / 'tables/cross_team'
    named = f'page_hinkley_standard_half_hourly_post_tune_{region}.csv'
    candidates = [cross_team / named,
                  RESULTS / 'tables/detection' / named,
                  cross_team / 'team41_accepted_detections.csv',
                  RESULTS / 'tables/detection/team41_accepted_detections.csv']
    candidate = next((p for p in candidates if p.is_file()), None)
    if candidate is None:
        # Search the existing GitHub results tree, without requiring a custom
        # location or mixing in archived/older experiment versions.
        hits = sorted(p for p in RESULTS.rglob(named) if 'archive' not in p.parts) if RESULTS.exists() else []
        candidate = hits[0] if hits else None
    if candidate is not None:
        other = pd.read_csv(candidate)
        source_description = str(candidate.relative_to(ROOT))
    else:
        uploaded = st.file_uploader(
            f'Team 41 · Page-Hinkley accepted detections ({region})',
            type=['csv'], key=f'f6_team41_upload_{region}')
        if uploaded is None:
            st.info('Team 41 detection file not found in GitHub results. Add '
                    f'`results/tables/cross_team/{named}` (original timestamp/day/label CSV). '
                    'Alternatively upload the file here for this session.')
            return
        other = pd.read_csv(uploaded)
        source_description = uploaded.name

    own = pd.read_csv(own_path)
    def standardise(raw, team, source=''):
        aliases = {
            'region': ('region', 'REGION', 'Region'),
            'time': ('SETTLEMENTDATE', 'detected_timestamp', 'detection_timestamp',
                     'detection_time', 'timestamp', 'alarm_timestamp',
                     'changepoint_timestamp', 'detected_at', 'datetime'),
            'detector': ('detector', 'method', 'detector_name'),
        }
        # Case-insensitive aliases also tolerate incidental surrounding spaces.
        lookup = {str(c).strip().lower(): c for c in raw.columns}
        found = {key: next((lookup.get(name.lower()) for name in names
                            if name.lower() in lookup), None)
                 for key, names in aliases.items()}
        if found['time'] is None:
            st.error(f'{team}: no detection timestamp column found. Columns: {list(raw.columns)}')
            return None
        d = raw.copy()
        if found['region'] is not None:
            d = d[d[found['region']].astype(str).str.strip().str.upper().eq(region)].copy()
        elif team == 'Team 41':
            # Region is unambiguous in the team's original filename.
            # Reject files that do not identify a region instead of assigning one.
            src = source.upper()
            if region not in src or any(r in src for r in ({'SA1','NSW1'} - {region})):
                st.error(f'Team 41: file has no region column; select the {region} filename, '
                         'or use a consolidated file with an explicit region column.')
                return None
        else:
            st.error('Team 42: region column is required.')
            return None
        if 'accepted' in d.columns:
            a = d['accepted'].astype(str).str.strip().str.lower()
            d = d[a.isin(['true','1','yes','accepted'])].copy()
        if 'status' in d.columns:
            statuses = d['status'].astype(str).str.strip().str.lower()
            if statuses.isin(['accepted','suppressed','rejected','raw']).any():
                d = d[statuses.eq('accepted')].copy()
      
        if found['detector'] is not None:
            detector_values = (d[found['detector']].astype(str).str.strip().str.lower()
                               .str.replace('-', '_', regex=False)
                               .str.replace(' ', '_', regex=False))
            d = d[detector_values.isin(['page_hinkley', 'pagehinkley', 'ph'])].copy()
        elif team == 'Team 42':
            st.error('Team 42: a detector column is required to verify Page-Hinkley timestamps.')
            return None
        # Team 41's supplied file has no detector column; its original file
        # is explicitly a Page-Hinkley export. Reject ambiguous filenames.
        elif 'page_hinkley' not in source.lower() and 'team41_accepted_detections' not in source.lower():
            st.error('Team 41: expected the Page-Hinkley detection export.')
            return None
        d['f6_time'] = pd.to_datetime(d[found['time']], errors='coerce')
        return (d.dropna(subset=['f6_time']).sort_values('f6_time')
                .drop_duplicates('f6_time').reset_index(drop=True))

    a = standardise(other, 'Team 41', source_description)
    b = standardise(own, 'Team 42')
    if a is None or b is None:
        return
    if a.empty or b.empty:
        st.warning(f'No accepted timestamps for {region}: Team 41={len(a)}, Team 42={len(b)}.')
        return
    # Verify against the accepted-alarm counts in Team 41's T7 summary.
    expected_team41 = {'SA1': 15, 'NSW1': 18}
    if len(a) != expected_team41[region]:
        st.warning(f'Team 41 file contains {len(a)} alarms, whereas its latest T7 summary reports '
                   f'{expected_team41[region]}. Check that the source is the final accepted-detector run.')

    # Greedy one-to-one nearest-timestamp pairing within the selected tolerance.
    # An accepted alarm cannot be paired with multiple alarms from the other team.
    hours = pd.Timedelta(hours=tolerance_h)
    pairs = []
    for i, t1 in enumerate(a['f6_time']):
        for j, t2 in enumerate(b['f6_time']):
            delta = abs(t1 - t2)
            if delta <= hours:
                pairs.append((delta, i, j))
    pairs.sort(key=lambda v: (v[0], v[1], v[2]))
    used_a, used_b, matches = set(), set(), []
    for delta, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matches.append((i, j, delta))
    a['matched'] = [i in used_a for i in range(len(a))]
    b['matched'] = [i in used_b for i in range(len(b))]

    earliest = min(a['f6_time'].min(), b['f6_time'].min()).date()
    latest = max(a['f6_time'].max(), b['f6_time'].max()).date()
    start = pd.Timestamp(earliest); end = pd.Timestamp(latest) + pd.Timedelta(days=1)
    av = a[a['f6_time'].between(start, end, inclusive='left')]
    bv = b[b['f6_time'].between(start, end, inclusive='left')]
    c = st.columns(3)
    c[0].metric('Team 41 accepted alarms', len(av))
    c[1].metric('Team 42 accepted alarms', len(bv))
    c[2].metric('Unmatched alarms in window', int((~av['matched']).sum() + (~bv['matched']).sum()))

    fig = go.Figure()
    for label, frame, pos, color in [('Team 41 · distribution shift',av,1,'#397ca0'),
                                      ('Team 42 · forecast error',bv,0,'#23816a')]:
        colors = ['#bd7231' if highlight and not m else color for m in frame['matched']]
        fig.add_trace(go.Scatter(x=frame['f6_time'], y=[pos]*len(frame),
            mode='markers', name=label, marker=dict(color=colors, size=13, symbol='line-ns-open', line=dict(width=2)),
            customdata=frame[['matched']].astype(str).to_numpy(),
            hovertemplate='%{x|%d %b %Y %H:%M}<br>Matched within tolerance: %{customdata[0]}<extra>%{fullData.name}</extra>'))
    fig.update_layout(height=380, title=f'{region} · Accepted Page–Hinkley detections',
        yaxis=dict(tickvals=[0,1], ticktext=['Team 42','Team 41'], range=[-.6,1.6], zeroline=False),
        xaxis_title='AEMO settlement date', hovermode='closest', plot_bgcolor='white',
        legend=dict(orientation='h',y=-.27), margin=dict(l=30,r=20,t=60,b=100))
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Amber markers indicate accepted alarms without a one-to-one corresponding alarm within the selected time tolerance. Correspondence describes temporal proximity, not causal equivalence or matching to documented AEMO events. This is a retrospective historical comparison, not a live detector execution.')
    with st.expander('Accepted alarms and correspondence audit'):
        audit = pd.concat([av.assign(team='Team 41'), bv.assign(team='Team 42')], ignore_index=True)
        table(audit[['team','f6_time','matched']].sort_values('f6_time'), 'f6_accepted_detection_audit_'+region)
    st.download_button('Download F6 interactive figure (HTML)', fig.to_html(include_plotlyjs='cdn'),
                       file_name=f'F6_{region}_trigger_disagreement.html', mime='text/html', key=f'f6_html_export_{region}')


def _t7_display_coverage(value):
    """Compact mean ± SD, preserving the underlying five-seed summary."""
    import re
    parts = re.split(r'\s*±\s*', str(value).strip())
    try:
        return ' ± '.join(f'{float(part):.3f}' for part in parts)
    except (ValueError, TypeError):
        return str(value)


def _t7_display_review_rate(value):
    """Show the mean review percentage only; don't display seed variability."""
    import re
    mean = re.split(r'\s*±\s*', str(value).strip())[0].strip()
    if mean.lower() in ('unreachable', '—', 'nan', ''):
        return '—' if mean.lower() in ('nan', '') else mean
    if mean.endswith('%'):
        try:
            return f'{float(mean[:-1]):.1f}%'
        except ValueError:
            return mean
    return mean


def render_t7():
    """Fixed, provenance-aware T7 comparison with both regions shown together."""
    section('Trigger comparison')
    st.caption('A comparative assessment of forecast-error and distribution-shift triggers under the reported Arm D protocol. Team 41 figures are transcribed from its shared T7 summary; Team 42 figures are read from this repository.')

    configuration = 'D_180d_page_hinkley'
    st.subheader('Experimental configuration and comparability')
    design = pd.DataFrame([
        ['Drift trigger', 'Distribution shift', 'Forecast error', 'Primary variable: distribution shift vs forecast error'],
        ['Detector', 'Page–Hinkley', 'Page–Hinkley', 'Consistent Page–Hinkley detection algorithm'],
        ['Forecasting model', 'XGBoost', 'HistGradientBoostingRegressor', 'Same gradient-boosting family; different model implementations'],
        ['Adaptation policy', 'Arm D · recent 180 days', 'Arm D · recent 180 days', 'Common Arm D, 180-day retraining window'],
        ['Dataset/Regions', 'SA1 and NSW1', 'SA1 and NSW1', 'Identical AEMO datasets (SA1 and NSW1) and evaluation periods'],
        ['Experiment seeds', '1–5', '1–5', 'Five-seed experimental protocol'],
        ['UQ method', 'AgACI', 'AgACI', 'AgACI-based methods; implementation equivalence unverified'],
        ['Escalation signals', 'Interval width; lagged nonconformity', 'Interval width; lagged 30-minute error', 'Comparable review criteria; methodological differences remain'],
        ['Regime definition', 'Documented events + 7-day post-event buffer', 'Documented events + 7-day post-event buffer', 'Consistent event-based definition with a seven-day buffer'],
    ], columns=['Setting', 'Team 41', 'Team 42', 'Comparison status'])
    st.dataframe(design, use_container_width=True, hide_index=True)

    team41 = {
        'SA1': {'MAE during drift': '281 ± 16 MW',
                'Worst 24h coverage': '0.71 ± 0.03',
                'Review rate for 0.90 coverage (interval width)': '0.0%',
                'Review rate for 0.90 coverage (nonconformity, 30 min)': '0.0%',
                'Detections raised': '15', 'Retrains triggered': '15'},
        'NSW1': {'MAE during drift': '907 ± 36 MW',
                 'Worst 24h coverage': '0.73 ± 0.02',
                 'Review rate for 0.90 coverage (interval width)': '0.0%',
                 'Review rate for 0.90 coverage (nonconformity, 30 min)': '0.0%',
                 'Detections raised': '18', 'Retrains triggered': '18'},
    }
    metric_names = list(team41['SA1'])
    st.subheader('T7 · Comparison outcomes (SA1 and NSW1)')

    def format_mean_sd(series, unit=''):
        values = pd.to_numeric(series, errors='coerce').dropna()
        if values.empty:
            return '—'
        sd = values.std(ddof=1) if len(values) > 1 else None
        result = f'{values.mean():,.2f}'
        if sd is not None:
            result += f' ± {sd:,.2f}'
        return result + unit

    def load_team42_region(region):
        team42 = {name: '—' for name in metric_names}
        provenance = {}
        issues = []

        t4_path = RESULTS / 'tables/adaptation/t4/t4_seed_level_metrics.csv'
        if t4_path.is_file():
            t4 = read_csv(str(t4_path))
            needed = {'region', 'configuration', 'seed', 'mae_drift', 'n_retrains'}
            if needed.issubset(t4.columns):
                sel = t4.loc[t4['region'].astype(str).str.upper().eq(region) &
                             t4['configuration'].astype(str).eq(configuration)].copy()
                if not sel.empty:
                    per_seed = sel.groupby('seed', as_index=False)[['mae_drift', 'n_retrains']].mean()
                    team42['MAE during drift'] = format_mean_sd(per_seed['mae_drift'], ' MW')
                    retrains = pd.to_numeric(per_seed['n_retrains'], errors='coerce')
                    team42['Retrains triggered'] = str(int(round(retrains.mean()))) if retrains.notna().any() else '—'
                    provenance[f'{region} · MAE and retrains'] = t4_path
                    if per_seed['seed'].nunique() != 5:
                        issues.append(f'{region}: T4 contains {per_seed["seed"].nunique()} seeds rather than five.')
                else:
                    issues.append(f'{region}: T4 has no {configuration} rows.')
            else:
                issues.append(f'{region}: T4 metrics file is missing expected columns.')
        else:
            issues.append(f'{region}: T4 seed-level metrics are not present.')

        cp_path = RESULTS / 'tables/detection/aemo_detection_changepoints.csv'
        if cp_path.is_file():
            cp = read_csv(str(cp_path))
            if {'region', 'detector', 'SETTLEMENTDATE'}.issubset(cp.columns):
                alarms = cp.loc[cp['region'].astype(str).str.upper().eq(region) &
                                cp['detector'].astype(str).str.lower().str.replace('-', '_', regex=False).eq('page_hinkley')]
                team42['Detections raised'] = str(alarms['SETTLEMENTDATE'].dropna().nunique())
                provenance[f'{region} · Accepted detections'] = cp_path
            else:
                issues.append(f'{region}: The changepoint CSV lacks region, detector, or SETTLEMENTDATE.')
        else:
            issues.append(f'{region}: T2 timestamp-level accepted detections are unavailable.')

        uq_base = RESULTS / 'tables/uncertainty'
        if uq_base.is_dir():
            t5_paths = [path for path in uq_base.rglob(f'T5_{region}_{configuration}.csv')
                        if not any(x in str(path).lower() for x in ('archive', 'simulation', 't5_aemo'))]
            post7_paths = [path for path in t5_paths if 'post7' in str(path).lower()]
            candidates = post7_paths if post7_paths else t5_paths
            if len(candidates) == 1:
                df5 = read_csv(str(candidates[0]))
                required = {'UQ method', 'regime', 'worst 24h coverage'}
                if required.issubset(df5.columns):
                    rows = df5.loc[df5['regime'].astype(str).str.lower().eq('drift') &
                                   df5['UQ method'].astype(str).str.lower().str.contains('agaci', na=False)]
                    if len(rows) == 1:
                        team42['Worst 24h coverage'] = _t7_display_coverage(rows.iloc[0]['worst 24h coverage'])
                        provenance[f'{region} · Worst 24h coverage'] = candidates[0]
                    else:
                        issues.append(f'{region}: T5 does not have exactly one AgACI drift-regime result.')
                else:
                    issues.append(f'{region}: T5 file does not contain the worst-24h coverage schema.')
            elif len(candidates) > 1:
                issues.append(f'{region}: Multiple current T5 sources exist; worst-24h coverage omitted.')
            else:
                issues.append(f'{region}: No matching T5 post-event table was found.')

            sources = _t6_sources()
            for signal, key in [('Interval width', 'Review rate for 0.90 coverage (interval width)'),
                                ('Lagged nonconformity', 'Review rate for 0.90 coverage (nonconformity, 30 min)')]:
                matching = [folder for kind, folder in sources if kind.lower().startswith(signal.lower())
                            and (folder / 't6_aggregate_audit.csv').is_file()]
                post7 = [folder for folder in matching if 'post7' in str(folder).lower()]
                if post7:
                    matching = post7
                if len(matching) != 1:
                    issues.append(f'{region}: {signal}: {len(matching)} unambiguous T6 audit sources; metric left blank.')
                    continue
                p6 = matching[0] / 't6_aggregate_audit.csv'
                df6 = read_csv(str(p6))
                required = {'region', 'configuration', 'uq_method', 'regime', 'metric', 'display'}
                if not required.issubset(df6.columns):
                    issues.append(f'{region}: {signal} T6 audit lacks expected fields.')
                    continue
                uq = df6['uq_method'].astype(str).str.lower().str.replace('-', '_', regex=False)
                rows = df6.loc[df6['region'].astype(str).str.upper().eq(region) &
                               df6['configuration'].astype(str).eq(configuration) &
                               uq.str.contains('agaci', na=False) &
                               df6['regime'].astype(str).str.lower().eq('drift') &
                               df6['metric'].astype(str).eq('review rate for 0.90')]
                if len(rows) == 1:
                    team42[key] = _t7_display_review_rate(rows.iloc[0]['display'])
                    provenance[f'{region} · {signal}'] = p6
                else:
                    issues.append(f'{region}: {signal}: no unique AgACI drift operating point.')

        return team42, provenance, issues

    team42_sa1, prov_sa1, issues_sa1 = load_team42_region('SA1')
    team42_nsw1, prov_nsw1, issues_nsw1 = load_team42_region('NSW1')
    provenance = {**prov_sa1, **prov_nsw1}
    issues = issues_sa1 + issues_nsw1

    comparison = pd.DataFrame({
        'Outcome': metric_names,
        'Team 41 · SA1': [team41['SA1'][m] for m in metric_names],
        'Team 41 · NSW1': [team41['NSW1'][m] for m in metric_names],
        'Team 42 · SA1': [team42_sa1[m] for m in metric_names],
        'Team 42 · NSW1': [team42_nsw1[m] for m in metric_names],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.download_button('Download T7 comparison (CSV)', comparison.to_csv(index=False).encode(),
                       file_name='T7_both_regions_fixed_page_hinkley_armD.csv',
                       mime='text/csv', key='t7_fixed_download')
    
    with st.expander('Data provenance and evaluation caveats'):
        st.write('**Fixed configuration:** Arm D (180 days), Page–Hinkley, seeds 1–5, AgACI-BOA and separate interval-width / lagged-error escalation metrics.')
        for label, path in provenance.items():
            st.caption(f'{label}: `{path.relative_to(ROOT)}`')
        for issue in issues:
            st.caption('Source check: ' + issue)


# Conclusion-page integrated historical replay; all observations come from repository files.
@st.cache_data(show_spinner=False)
def _demo_read_configuration(path, configuration):
    pieces = []
    for chunk in pd.read_csv(path, chunksize=75000):
        selected = chunk[chunk['configuration'].astype(str).eq(configuration)]
        if not selected.empty:
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def render_integrated_prototype():
    section('Integrated live prototype · historical stream replay')
    st.caption('A chronological replay of saved AEMO predictions and causal review signals; not a live AEMO feed. All data is read from experiment output files.')
    base = RESULTS / 'tables/uncertainty/agaci_boa_fixed_calibration'
    seed_dirs = sorted([d for d in base.glob('seed_*') if d.is_dir() and d.name[5:].isdigit()],
                       key=lambda d: int(d.name[5:])) if base.exists() else []
    if not seed_dirs:
        st.warning('No seed folders found in results/tables/uncertainty/agaci_boa_fixed_calibration/.')
        return
    a,b,c = st.columns(3)
    with a: seed_dir = st.selectbox('Experiment seed', seed_dirs, format_func=lambda d: d.name, key='conclusion_demo_seed')
    with b: region = st.selectbox('AEMO region', ['SA1','NSW1'], key='conclusion_demo_region')
    with c: method = st.selectbox('Uncertainty method', list(METHODS), index=2, key='conclusion_demo_method')
    file = seed_dir / f'agaci_comparison_intervals_{region}.csv.gz'
    if not file.is_file():
        st.warning(f'Interval source unavailable: {file.relative_to(ROOT)}')
        return
    @st.cache_data(show_spinner=False)
    def configurations(path):
        values=set()
        for chunk in pd.read_csv(path, usecols=['configuration'], chunksize=150000):
            values.update(chunk['configuration'].dropna().astype(str).unique())
        return sorted(values)
    names = configurations(str(file))
    if not names:
        st.warning('No adaptation configurations in interval file.')
        return
    c1,c2 = st.columns(2)
    with c1:
        configuration = st.selectbox('Adaptation configuration', names,
             index=names.index('C_page_hinkley') if 'C_page_hinkley' in names else 0,
             key='conclusion_demo_config')
    with c2:
        detector_names=['page_hinkley','adwin','kswin']
        fixed_detector = next((x for x in detector_names if configuration.endswith('_'+x)),None)
        if fixed_detector:
            st.text_input('Detector used by adaptation arm', value=fixed_detector, disabled=True, key='conclusion_demo_detector_fixed')
            detector=fixed_detector
        else:
            detector=st.selectbox('Detector overlay (not a retraining trigger for this arm)',
                        detector_names, key='conclusion_demo_detector')
    df = _demo_read_configuration(str(file), configuration)
    if df.empty:
        st.warning('No observations for the selected configuration.')
        return
    df['SETTLEMENTDATE']=pd.to_datetime(df['SETTLEMENTDATE'],errors='coerce')
    df=df.dropna(subset=['SETTLEMENTDATE']).sort_values('SETTLEMENTDATE').reset_index(drop=True)
    lo,hi=METHODS[method]
    needed=['SETTLEMENTDATE','y_true','point_forecast',lo,hi]
    if not set(needed).issubset(df.columns):
        st.error('Interval source is missing: '+', '.join(set(needed)-set(df.columns)))
        return
    for col in needed[1:]: df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df.dropna(subset=needed[1:]).copy()
    if df.empty:
        st.warning('No valid forecast/interval observations.'); return
    df['covered']=df['y_true'].between(df[lo],df[hi])
    df['width']=df[hi]-df[lo]
    # Shift across the COMPLETE selected series, not merely the displayed window.
    df['lagged_error']=(df['y_true']-df['point_forecast']).abs().shift(1)
    first_date=df['SETTLEMENTDATE'].min().normalize()
    last_date=df['SETTLEMENTDATE'].max().normalize()
    max_offset=max(0,(last_date-first_date).days-4)
    r1,r2=st.columns(2)
    with r1: signal=st.radio('Review-prioritisation signal', ['Lagged 30-minute error','Interval width'],
                             horizontal=True,key='conclusion_demo_signal')
    scores=df['lagged_error'] if signal.startswith('Lagged') else df['width']
    valid=scores.replace([np.inf,-np.inf],np.nan).dropna()
    if valid.empty:
        st.warning('No valid review scores available.');return
    with r2:
        percentile=st.slider('Exploratory review threshold · score percentile', 0,100,80,
                            key='conclusion_demo_percentile')
    tau=float(valid.quantile(percentile/100))
    df['review']=scores>tau
    # Use an actual calendar date rather than an opaque day offset.
    date_key=f'conclusion_demo_date_{seed_dir.name}_{region}'
    min_start=first_date.date()
    max_start=(first_date+pd.Timedelta(days=max_offset)).date()
    if date_key not in st.session_state:
        st.session_state[date_key]=min_start
    st.session_state[date_key]=min(max(st.session_state[date_key],min_start),max_start)
    prev,centre,nxt=st.columns([1,5,1])
    with prev:
        if st.button('◀ Previous 5d',key='conclusion_demo_prev',
                     disabled=st.session_state[date_key]<=min_start):
            st.session_state[date_key]=max(min_start,
                st.session_state[date_key]-pd.Timedelta(days=5))
    with nxt:
        if st.button('Next 5d ▶',key='conclusion_demo_next',
                     disabled=st.session_state[date_key]>=max_start):
            st.session_state[date_key]=min(max_start,
                st.session_state[date_key]+pd.Timedelta(days=5))
    with centre:
        st.date_input('Select historical replay date (start of five-day window)',
                      min_value=min_start,max_value=max_start,key=date_key,
                      format='DD/MM/YYYY')
    start=pd.Timestamp(st.session_state[date_key])
    end=start+pd.Timedelta(days=5)
    st.caption(f'Displaying {start:%d %b %Y} – {end-pd.Timedelta(days=1):%d %b %Y}')
    view=df[(df['SETTLEMENTDATE']>=start)&(df['SETTLEMENTDATE']<end)].copy()
    if view.empty:
        st.info('No observations in this window.');return
    cp_path=RESULTS/'tables/detection/aemo_detection_changepoints.csv'
    alarms=pd.DataFrame()
    if cp_path.is_file():
        cp=read_csv(str(cp_path))
        required={'region','detector','SETTLEMENTDATE'}
        if required.issubset(cp.columns):
            alarms=cp[cp['region'].astype(str).str.upper().eq(region)&
                      cp['detector'].astype(str).str.lower().eq(detector)].copy()
            alarms['SETTLEMENTDATE']=pd.to_datetime(alarms['SETTLEMENTDATE'],errors='coerce')
            alarms=alarms[(alarms['SETTLEMENTDATE']>=start)&(alarms['SETTLEMENTDATE']<end)]
    else:
        st.info('Detection markers require results/tables/detection/aemo_detection_changepoints.csv.')
    # Exact adaptation events for the selected seed, region and configuration.
    # Never infer model updates from detector alarms or summary counts.
    retrain_path = (RESULTS / 'tables/adaptation/revised/pgs242_128_five_seed_final'
                    / seed_dir.name / 'retraining_events.csv')
    retrains = pd.DataFrame()
    if retrain_path.is_file():
        event_log = read_csv(str(retrain_path))
        expected_events = {'region', 'configuration', 'retrain_timestamp',
                           'first_forecast_timestamp'}
        if expected_events.issubset(event_log.columns):
            match = (event_log['region'].astype(str).str.upper().eq(region) &
                     event_log['configuration'].astype(str).eq(configuration))
            if 'seed' in event_log.columns:
                match &= pd.to_numeric(event_log['seed'], errors='coerce').eq(int(seed_dir.name[5:]))
            retrains = event_log.loc[match].copy()
            for date_col in ('retrain_timestamp', 'first_forecast_timestamp'):
                retrains[date_col] = pd.to_datetime(retrains[date_col], errors='coerce')
            retrains = retrains.dropna(subset=['retrain_timestamp'])
            retrains = retrains[(retrains['retrain_timestamp'] >= start) &
                                (retrains['retrain_timestamp'] < end)].copy()
        else:
            st.warning('Retraining event log is missing required timestamp or configuration columns.')
    elif configuration != 'A':
        st.info('Retraining timeline unavailable: no matching seed-level retraining_events.csv in the adaptation results.')
    keep=view[~view['review']]
    metrics=st.columns(5)
    metrics[0].metric('Window MAE',f"{(view['y_true']-view['point_forecast']).abs().mean():.1f} MW")
    metrics[1].metric('Window coverage',f"{view['covered'].mean():.1%}")
    metrics[2].metric('Human-review flags',f"{view['review'].mean():.1%}")
    metrics[3].metric('Accepted detector alarms',str(len(alarms)))
    metrics[4].metric('Model retrains',str(len(retrains)))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'],y=view[hi],mode='lines',
                             line=dict(width=0),hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'],y=view[lo],mode='lines',
                             fill='tonexty',fillcolor='rgba(45,122,167,.16)',
                             line=dict(width=0),name='90% interval'))
    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'],y=view['y_true'],mode='lines',
                             line=dict(color='#23394b',width=2),name='Actual incoming demand'))
    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'],y=view['point_forecast'],mode='lines',
                             line=dict(color='#1684a1',width=2),name='Point forecast'))
    missed=view[~view['covered']]
    if not missed.empty:
        fig.add_trace(go.Scatter(x=missed['SETTLEMENTDATE'],y=missed['y_true'],mode='markers',
                      marker=dict(symbol='circle-open',color='#c44a50',size=9),name='Interval misses (retrospective)'))
    reviewed=view[view['review']]
    if not reviewed.empty:
        fig.add_trace(go.Scatter(x=reviewed['SETTLEMENTDATE'],y=reviewed['point_forecast'],
                      mode='markers',marker=dict(symbol='diamond',color='#bd762a',size=7),name='Escalation flag'))
    for timestamp in alarms['SETTLEMENTDATE'] if not alarms.empty else []:
        fig.add_vline(x=timestamp,line_color='#b34547',line_dash='dash',line_width=1.5)
    if not alarms.empty:
        fig.add_trace(go.Scatter(x=alarms['SETTLEMENTDATE'],
            y=[view['y_true'].max()]*len(alarms),mode='markers',
            marker=dict(symbol='triangle-down',color='#b34547',size=12),
            name='Accepted detector changepoint'))
    if not retrains.empty:
        for timestamp in retrains['retrain_timestamp']:
            fig.add_vline(x=timestamp, line_color='#25835a', line_dash='dot', line_width=2)
        fig.add_trace(go.Scatter(
            x=retrains['retrain_timestamp'],
            y=[view['y_true'].max() * 1.055] * len(retrains),
            mode='markers', marker=dict(symbol='square', color='#25835a', size=11),
            customdata=retrains[['first_forecast_timestamp']].astype(str).to_numpy(),
            hovertemplate='Model retrained: %{x}<br>First updated forecast: %{customdata[0]}<extra></extra>',
            name='Model retrained'))
        updated = retrains.dropna(subset=['first_forecast_timestamp']).copy()
        updated = updated[(updated['first_forecast_timestamp'] >= start) &
                          (updated['first_forecast_timestamp'] < end)]
        if not updated.empty:
            forecast_lookup = view.set_index('SETTLEMENTDATE')['point_forecast']
            updated['forecast_value'] = updated['first_forecast_timestamp'].map(forecast_lookup)
            updated = updated.dropna(subset=['forecast_value'])
            if not updated.empty:
                fig.add_trace(go.Scatter(
                    x=updated['first_forecast_timestamp'], y=updated['forecast_value'],
                    mode='markers', marker=dict(symbol='star', color='#25835a', size=12,
                                                line=dict(color='white', width=1)),
                    name='First forecast after retraining',
                    hovertemplate='First forecast after retraining<br>%{x}<br>%{y:.1f} MW<extra></extra>'))
    fig.update_layout(height=550,title=f'{region} · {configuration} · {seed_dir.name} · {start:%d %b %Y}–{end-pd.Timedelta(days=1):%d %b %Y}',
                      xaxis_title='Settlement date',yaxis_title='Demand (MW)',hovermode='x unified',
                      legend=dict(orientation='h',y=-.2),margin=dict(b=120))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Red dashed lines = accepted T2 changepoints; green dotted lines/squares = actual model retraining; green stars = first available forecast after retraining. For Arm A there are no retrains. Review thresholds are exploratory, interval misses retrospective, and human interventions are not simulated. Retraining markers are read from the selected seed-level adaptation log, never inferred from alarm counts.')
    with st.expander('Inspect current window observations and review decisions'):
        cols=['SETTLEMENTDATE','y_true','point_forecast',lo,hi,'covered','width','lagged_error','review']
        table(view[cols],f'integrated_replay_{region}_{seed_dir.name}_{configuration}')

STAGES = [
    '01 · Project Overview',
    '02 · Ethical Considerations',
    '03 · Forecast Degradation',
    '04 · Drift Detection',
    '05 · Forecast Adaptation',
    '06 · Uncertainty & Reliability',
    '07 · Human Escalation',
    '08 · Trigger Comparison',
    '09 · Integrated System & Research Synthesis',
]

st.markdown("""<style>
:root { --ink:#244766; --muted:#58718a; --blue:#668fb4; --teal:#6eafad; --border:#d7e4ed; }
.research-head {background:linear-gradient(112deg,#eef5fb,#ffffff 68%,#eef8f6);border:1px solid #b9d2e3;border-left:5px solid #668fb4;border-radius:18px;padding:24px 28px;margin:10px 0 19px;}
.research-head .eyebrow{font-size:.78rem;font-weight:800;letter-spacing:.09em;color:#4276a0;margin-bottom:8px;}
.research-head h2{font-size:1.75rem;color:#1b394f;margin:0 0 10px;font-weight:780;line-height:1.25;}
.research-head p{color:#52677c;line-height:1.65;margin:0;font-size:.99rem;}
.research-insight{background:#f0f7fa;border:1px solid #d3e5eb;border-left:4px solid #6eafad;padding:14px 19px;border-radius:11px;margin:19px 0;color:#315169;line-height:1.65;}
.research-insight strong{color:#244766;}
.research-section{color:#244766;font-weight:760;font-size:1.25rem;border-bottom:1px solid #dfe8ee;padding-bottom:9px;margin:27px 0 15px;}
.research-step{background:#f2f6f9;border:1px solid #dbe6ed;border-radius:13px;padding:16px 20px;margin:12px 0;color:#38566b;line-height:1.65;}
.research-step strong{color:#244766;}
.research-status{background:#f2f6f9;border:1px dashed #a8c9df;border-radius:13px;padding:19px 23px;color:#47637b;margin:12px 0;}
.research-status strong{color:#244766;}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:.92rem !important;}
</style>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="research-section">{title}</div>', unsafe_allow_html=True)

def stage_header(kicker, title, purpose):
    st.markdown(f'<div class="research-head"><div class="eyebrow">{kicker}</div><h2>{title}</h2><p>{purpose}</p></div>', unsafe_allow_html=True)

def insight(title, body):
    st.markdown(f'<div class="research-insight"><strong>{title}</strong> {body}</div>', unsafe_allow_html=True)

def note(body):
    st.markdown(f'<div class="research-step">{body}</div>', unsafe_allow_html=True)

def future_output(code, title, description, pattern=None):
    section(f'{code} · {title}')
    st.write(description)
    if pattern:
        folder=RESULTS / ('figures' if pattern.endswith('.png') else 'tables')
        found=[p for p in folder.rglob(pattern) if 'archive' not in p.parts] if folder.exists() else []
        if found:
            p=sorted(found)[0]
            st.caption(f'Found saved output: {p.relative_to(ROOT)}')
            if p.suffix.lower()=='.png':
                show_image(p, f'{code} · {title}')
            elif p.suffix.lower()=='.csv':
                table(load(p), f'{code}_results')
            if len(found)>1:
                with st.expander(f'Other {code} outputs ({len(found)-1})'):
                    for other in sorted(found)[1:]:
                        st.caption(str(other.relative_to(ROOT)))
                        if other.suffix.lower()=='.png': show_image(other)
                        elif other.suffix.lower()=='.csv': table(load(other), code+'_'+other.stem)
            return
    st.markdown('<div class="research-status"><strong>Results in progress</strong><div>The corresponding experimental output has not yet been added to the repository.</div></div>', unsafe_allow_html=True)


st.markdown("""<style>
    .overview-hero{background:linear-gradient(120deg,#eef5fb,#ffffff 58%,#eef8f6);border:1px solid #b9d2e3;border-radius:22px;padding:32px 35px;margin:12px 0 22px}
    .overview-eyebrow{font-size:.82rem;letter-spacing:.12em;font-weight:750;color:#426d95;margin-bottom:10px}
    .overview-hero h1{font-size:2.15rem;line-height:1.22;color:#10243a;margin:0 0 14px;font-weight:800}
    .overview-hero p{font-size:1.06rem;color:#42556b;line-height:1.65;margin:0 0 12px}
    .overview-chip{display:inline-block;background:#fff;border:1px solid #d9e7f2;border-radius:30px;padding:6px 12px;margin:5px 5px 0 0;color:#31506e;font-size:.84rem;font-weight:600}
    .overview-card{box-sizing:border-box;background:#f1f7ff;border:1px solid #d7e6fa;border-top:5px solid #668fb4;border-radius:18px;padding:23px 23px;margin:8px 0 16px;min-height:190px;box-shadow:0 5px 16px rgba(33,71,110,.035)}
    .overview-card h4{color:#143b60;margin:0 0 12px;font-size:1.10rem;font-weight:750}
    .overview-card p{color:#42556b;margin:0 0 8px;font-size:.96rem;line-height:1.62}
    .overview-card .ov-label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;font-weight:800;opacity:.83;margin-bottom:9px}
    .overview-card.teal{background:#effbf8;border-color:#cbece3;border-top-color:#6eafad}
    .overview-card.purple{background:#f7f3ff;border-color:#e4dafa;border-top-color:#8099b8}
    .overview-card.amber{background:#fff8eb;border-color:#f6e3be;border-top-color:#b58e63}
    .overview-card.blue{background:#f0f6ff;border-color:#d7e6fa;border-top-color:#668fb4}
    .overview-card.green{background:#f2faee;border-color:#dcebd1;border-top-color:#66a75a}
    .overview-card.equal{min-height:215px}
    .overview-card.ethics{min-height:245px}
    .overview-card.ethics p{margin-top:6px}
    .overview-card .ov-action{border-top:1px solid rgba(36,65,90,.13);padding-top:11px;margin-top:12px}
    .overview-note{background:linear-gradient(105deg,#eef6ff,#f4fbfa);border-left:5px solid #2878c8;padding:17px 20px;border-radius:12px;color:#284663;margin:14px 0 22px}
    .overview-section{color:#173c60;font-weight:800;letter-spacing:-.015em;border-bottom:1px solid #e4edf6;padding-bottom:10px;margin:36px 0 16px}
    .overview-timeline{display:flex;align-items:center;gap:7px;margin:15px 0 7px}
    .overview-timeline span{display:block;height:12px;border-radius:12px}
    .overview-kpi{background:linear-gradient(145deg,#f3f8ff,#fff);border:1px solid #dce8f4;border-radius:16px;padding:17px 20px;margin:0 0 12px}
    .overview-kpi .number{font-size:2.05rem;color:#1b5bb5;font-weight:800;line-height:1.2}
    .overview-kpi .heading{color:#244461;font-weight:750;font-size:.94rem;margin-bottom:8px}
    .overview-kpi .detail{color:#64758a;font-size:.80rem}
    .overview-flow{background:linear-gradient(100deg,#edf6ff,#f5f1ff,#edfbf6);border:1px solid #dbe5f4;border-radius:17px;padding:22px 24px;margin:14px 0 20px;color:#244663;font-weight:700;line-height:2}
    .overview-flow span{display:inline-block;background:white;border:1px solid #dce6f1;border-radius:9px;padding:4px 10px;margin:3px 1px}
</style>""", unsafe_allow_html=True)


st.sidebar.markdown('**CONCEPT DRIFT LAB**')
st.sidebar.caption('TEAM 42 · RESEARCH DEMONSTRATION')
page = st.sidebar.radio('Explore the research', STAGES, label_visibility='collapsed')
st.sidebar.divider()
with st.sidebar.expander('Research outputs index'):
    st.caption('F1 · Degradation  |  T1–T2 / F2 · Detection  |  T3–T4 / F3 · Adaptation  |  T5–T6 / F4–F5 · Uncertainty and review  |  T7 / F6 · Trigger comparison')
    st.caption("All experimental results, figures, and tables are sourced from the project's GitHub repository.")

if page == STAGES[0]:
    stage_header('ADELAIDE UNIVERSITY · ICT CAPSTONE · TEAM 42 · 2026',
                 'Detecting and Adapting to Concept Drift in Time-Series Forecasting',
                 'Can an electricity-demand forecaster recognise changing patterns, adapt when appropriate, and communicate whether its predictions remain reliable?')
    cols=st.columns(4)
    for c,n,label,detail in zip(cols,['2','3','4','5'],['AEMO regions','Drift detectors','Adaptation arms','Experiment seeds'],['SA1 and NSW1','ADWIN · KSWIN · PH','A · B · C · D','Repeated comparisons']):
        with c:
            st.markdown(f'<div class="overview-kpi"><div class="heading">{label}</div><div class="number">{n}</div><div class="detail">{detail}</div></div>', unsafe_allow_html=True)
    section('The research problem')
    st.write('Demand patterns change with consumption behaviour, seasonal effects and exceptional events. A previously accurate forecasting model can therefore lose reliability over time, even while continuing to produce predictions. This motivates drift monitoring, selective retraining and explicit uncertainty evaluation.')
    section('Three connected research questions')
    c1,c2,c3=st.columns(3)
    for c,accent,title,detail in [
        (c1,'blue','Detect change','How reliably and quickly do the detectors respond, and how many alerts remain unexplained?'),
        (c2,'teal','Adapt forecasts','Can drift-triggered retraining reduce errors without unnecessary computational work?'),
        (c3,'purple','Assess reliability','Do prediction intervals maintain coverage, and when might human review be warranted?')]:
        with c:
            st.markdown(f'<div class="overview-card equal {accent}"><h4>{title}</h4><p>{detail}</p></div>',unsafe_allow_html=True)
    section('Data and methodology')
    st.write('The experimental study combines **historical AEMO half-hourly electricity-demand observations** for SA1 and NSW1 with **synthetic change streams** whose changepoints are known.')
    splits=st.columns(3)
    for c,sty,title,dates,why in [
        (splits[0],'blue','TRAIN','2018–2019','Fit models and historical preprocessing'),
        (splits[1],'teal','CALIBRATE','Jan–Feb 2020','Set calibration references before testing'),
        (splits[2],'purple','TEST','Mar 2020 onward','Evaluate sequentially on held-out observations')]:
        with c:
            st.markdown(f'<div class="overview-card equal {sty}"><h4>{title}</h4><p><b>{dates}</b></p><p>{why}</p></div>',unsafe_allow_html=True)
    note('<strong>Experimental validity:</strong> Historical-only preprocessing and chronological testing limit look-ahead leakage; five seeds support repeatability. Synthetic changepoints are known, but AEMO documented events are not exhaustive ground truth.')
    section('System architecture')
    st.markdown('<div class="overview-flow"><span>Electricity demand</span> → <span>Forecast</span> → <span>Monitor drift</span> → <span>Retrain</span> → <span>Quantify uncertainty</span> → <span>Evaluate & review</span></div>',unsafe_allow_html=True)
    section('Methods & design decisions')
    st.caption('Four complementary components connect forecasting performance, change detection, model adaptation and predictive reliability.')
    st.markdown("""<style>
    .method-card{border:1px solid #d8e4ec;border-radius:16px;background:#f8fbfd;
        padding:18px 20px;min-height:182px;margin:4px 0 14px;box-shadow:0 2px 8px rgba(36,71,102,.035)}
    .method-head{display:flex;align-items:center;gap:12px;margin-bottom:12px}
    .method-icon{width:43px;height:43px;border-radius:12px;display:flex;
        align-items:center;justify-content:center;flex-shrink:0;background:#e6f0f8;color:#32658c}
    .method-icon.teal{background:#e3f3f1;color:#347e7b}
    .method-icon.navy{background:#e8edf4;color:#244766}
    .method-icon.steel{background:#eaf0f6;color:#547e9e}
    .method-icon svg{height:23px;width:23px;stroke:currentColor;fill:none;
        stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
    .method-card h4{margin:0;color:#244766;font-size:1.05rem;font-weight:750}
    .method-card p{margin:0;color:#485e70;font-size:.94rem;line-height:1.65}
    .method-card strong{color:#284e6e}
    </style>""",unsafe_allow_html=True)
    cards = [
        ('Forecasting','blue',
         '<path d="M3 3v18h18"/><path d="m6 16 5-5 4 3 5-7"/>',
         '<strong>Seasonal Naïve, SARIMAX and Gradient Boosting.</strong> Statistical baselines provide reference performance; gradient boosting with calendar and lag features supports adaptation experiments.'),
        ('Drift detection','teal',
         '<path d="M3 12h4l3-7 4 14 3-7h4"/><path d="M18 3v3"/>',
         '<strong>ADWIN, KSWIN and Page-Hinkley.</strong> Evaluated on synthetic streams with known changepoints and on AEMO demand using documented events as reference points.'),
        ('Adaptation','navy',
         '<path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M5.5 9a7 7 0 0 1 12-3L20 12"/><path d="M4 12l2.5 6a7 7 0 0 0 12-3"/>',
         '<strong>Four strategies (A–D).</strong> No retraining; every 30 days; drift-triggered full-history retraining; and drift-triggered recent-window retraining.'),
        ('Uncertainty','steel',
         '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/>',
         '<strong>Static and adaptive conformal prediction.</strong> Prediction intervals target nominal 90% coverage; both empirical coverage and interval width are evaluated.'),
    ]
    for start in (0,2):
        cols=st.columns(2)
        for col,(heading,accent,paths,description) in zip(cols,cards[start:start+2]):
            with col:
                st.markdown(f'''<div class="method-card">
                <div class="method-head"><span class="method-icon {accent}">
                <svg viewBox="0 0 24 24" aria-hidden="true">{paths}</svg></span>
                <h4>{heading}</h4></div><p>{description}</p></div>''',unsafe_allow_html=True)
    section('Follow the evidence')
    st.write('The investigation proceeds from observed forecast degradation to drift detection, retraining, uncertainty, human review and cross-team comparison. Each stage connects its research question to experimental evidence.')
    with st.expander('Project coordination and experimental reproducibility'):
        st.write('The team integrates data preparation, detector evaluation, adaptation, accuracy and cost analysis, and conformal uncertainty. Evaluation conventions are aligned with Team 41 for an interpretable cross-team comparison. The required outputs are T1–T7, F1–F6 and machine-readable experiment logs.')
        run_path=RESULTS/'runs.csv'
        if run_path.is_file():
            log=load(run_path)
            st.caption(f'Experiment log: {len(log):,} rows · results/runs.csv')
            table(log.head(100), 'runs_preview')
        else:
            st.caption('Experiment-level logs are not included in this local dashboard copy.')

elif page == STAGES[1]:
    stage_header('STAGE 02 · ETHICAL, SOCIETAL & SECURITY ANALYSIS',
                 'Ethical and Security Implications of Concept-Drift Adaptation',
                 'Concept-drift adaptation introduces a fundamental trade-off between responsiveness to changing data and the stability of forecasting decisions. This study considers the ethical and security implications of that trade-off, particularly unreliable drift alarms, unnecessary model updates, uncertainty miscalibration and limitations in human oversight.')

    section('Ethical significance of the research')
    st.write(
        'Electricity-demand forecasting is consequential because inaccurate or poorly communicated predictions may affect planning decisions. '
        'Within this study, ethical responsibility concerns not only forecast error but also **the interpretation of drift alarms, '
        'the justification for model updates, the reliability of uncertainty estimates and the consistency of forecasting reliability '
        'across the evaluated regions**. These questions are investigated through historical experiments rather than claims about operational outcomes.'
    )

    section('Research-grounded ethical and security considerations')
    cases = [
        ('01 · False certainty about detected change',
         'An alarm from ADWIN, KSWIN or Page–Hinkley signals a potential change according to the detector’s statistical criterion, '
         'but does not independently establish its cause or correspondence with a documented real-world event. '
         'Equating an unmatched detection with a confirmed false alarm—or a matched detection with proof of causation—would overstate the evidence.',
         'T2 event matching and chance-baseline analysis distinguish detector behaviour from incomplete historical event annotations. '
         'This supports transparent interpretation without treating the AEMO event catalogue as exhaustive ground truth.'),
        ('02 · The consequences of unnecessary adaptation',
         'Retraining after accepted alarms may consume computational resources without proportionate gains in predictive accuracy, '
         'while leaving a model unchanged may allow forecasting performance to deteriorate. This raises questions of resource stewardship '
         'and whether automated updates justify their computational expenditure and changes in model behaviour.',
         'Arms A–D compare no retraining, periodic updates, drift-triggered full-history retraining and drift-triggered recent-window retraining. '
         'T3 and T4 assess forecasting error alongside update frequency, training volume and runtime, making the benefit–cost trade-off explicit.'),
        ('03 · Misleading confidence and limits of human oversight',
         'A nominal 90% prediction interval can conceal poor coverage over shorter periods, creating misleading confidence during changing demand conditions. '
         'Human-review prioritisation adds a further trade-off: flagging more forecasts increases the potential oversight workload, '
         'without demonstrating that those forecasts were corrected by a person.',
         'T5 and F4 evaluate Static, ACI and AgACI-BOA through empirical coverage, interval width and temporal reliability. '
         'T6 and F5 quantify retained-forecast coverage against review rates. These are retrospective evaluations of proposed review flags, '
         'not measurements of actual human intervention or its effectiveness.'),
        ('04 · Data integrity and regional context',
         'Missing, out-of-order, corrupted or deliberately manipulated demand observations could affect lagged forecasting features '
         'and drift-monitoring decisions. SA1 and NSW1 also exhibit different demand conditions, so a common adaptation policy '
         'need not achieve comparable forecasting reliability in both regions.',
         'Chronological splits, documented preprocessing, experiment logs and region-specific evaluation support traceability '
         'and prevent regional differences from being hidden in pooled results. These methodological measures do not constitute '
         'a defence against adversarial data manipulation. Input-integrity validation and adversarial stress-testing are plausible '
         'additional safeguards, but were not evaluated in this study; nor was fairness across population groups.')
    ]
    for heading, issue, response in cases:
        with st.expander(heading, expanded=True):
            st.markdown(f'**Ethical, societal or security concern:** {issue}')
            st.markdown(f'**Analytical response and supporting evidence:** {response}')

    section('Synthesis of the ethical analysis')
    st.write(
        'The central implication of this analysis is that **reliable drift detection, model adaptation and nominal uncertainty '
        'calibration do not individually establish forecasting trustworthiness**. Their ethical significance lies in how statistical '
        'evidence informs model-update decisions, how uncertainty is communicated, and how reliability is balanced against computational '
        'and human-review costs. Evaluating these components jointly provides an auditable basis for assessing adaptive forecasting '
        'on historical SA1 and NSW1 data, while recognising the limits of the evidence concerning security robustness, regional '
        'generalisation and effective human intervention.'
    )

elif page == STAGES[2]:
    stage_header('STAGE 03 · F1', 'Why does forecasting performance degrade?',
                 'Use the rolling forecast error of frozen baseline models to establish the problem and its relationship to documented demand events.')
    note('<strong>Research purpose:</strong> A sustained or abrupt increase in rolling MAE motivates monitoring, but does not by itself establish the cause of the change.')
    render_0()
    insight('Research significance:', 'Rolling MAE tracks how prediction error changes over time. Event annotations offer historical context, but do not establish that an event caused an increase in error.')

elif page == STAGES[3]:
    stage_header('STAGE 04 · T1 · T2 · F2', 'Can meaningful changes be detected?',
                 'First test the algorithms against known synthetic changepoints, then evaluate their alignment with documented real-world AEMO events.')
    mode=st.radio('Evidence', ['T1 · Synthetic benchmark', 'T2 / F2 · AEMO event matching'], horizontal=True, key='stage_detect')
    if mode.startswith('T1'):
        note('<strong>Why synthetic first?</strong> Known changepoints allow detection delay, missed events and false alarms to be measured directly, including a no-drift control.')
        render_1()
        insight('Research significance:', 'The known synthetic changepoints make detection delays, missed changes and false alarms directly measurable across sudden, gradual, recurring and no-drift conditions.')
    else:
        note('<strong>Important limitation:</strong> The historical event catalogue is not exhaustive ground truth. An unmatched detection should not automatically be labelled a false positive.')
        render_2()
        insight('Research significance:', 'Event matching describes how often alerts coincide with documented changes; unmatched detections remain distinct from confirmed false alarms because the event catalogue is not exhaustive.')

elif page == STAGES[4]:
    stage_header('STAGE 05 · F3 · T3 · T4', 'When is retraining worthwhile?',
                 'Compare frozen, periodic and drift-triggered forecasting on the same test stream, then examine regime-specific accuracy and computational cost.')
    note('<strong>Four-arm design:</strong> A — never retrain; B — every 30 days; C — on detected drift using full history; D — on detected drift using a recent window. Arms A and B are essential controls.')
    mode=st.radio('Explore adaptation', ['Interactive forecasts', 'F3 · Four-arm comparison', 'T3 · Forecast accuracy', 'T4 · Adaptation cost', 'Model selection'], horizontal=True, key='stage_adapt')
    if mode=='Interactive forecasts':
        render_6()
        insight('Research significance:', 'The forecast trajectories show actual and predicted electricity demand for the selected region, run and retraining policy. Models may initially produce identical predictions before their first retraining event.')
    elif mode.startswith('F3'):
        render_5()
        insight('Research significance:', 'The four forecasting policies are compared against the same demand stream. Retraining markers distinguish scheduled updates from detector-triggered adaptation.')
    elif mode.startswith('T3'):
        render_3()
        insight('Research significance:', 'Regime-specific MAE separates pre-drift, drift and post-drift performance; repeated seeds quantify variability rather than relying on a single run.')
    elif mode.startswith('T4'):
        render_4()
        insight('Research significance:', 'Retraining count, cumulative training samples and runtime contextualise MAE differences. More frequent retraining is not necessarily more effective or efficient.')
    else:
        render_model_selection()

elif page == STAGES[5]:
    stage_header('STAGE 06 · T5 · F4', 'Do prediction intervals remain reliable during drift?',
                 'Study static and adaptive conformal uncertainty, including interval coverage, width and how calibration changes over time.')
    render_7()
    insight('Research significance:', 'Coverage measures how often observed demand falls inside prediction intervals; width captures their precision. Rolling and worst-window coverage reveal failures that may be hidden by full-period averages.')
    render_f4()

elif page == STAGES[6]:
    stage_header('STAGE 07 · T6 · F5', 'Evaluating Human-Review Escalation in Adaptive Forecasting',
                 'Assess review-prioritisation strategies by examining prediction-interval coverage, review rates and their trade-offs across operating thresholds.')
    view=st.radio('Escalation view', ['Five-day interactive explorer','F5 · Review–coverage curves','T6 · Operating-point table'], horizontal=True, key='stage_esc_view')
    if view.startswith('Five-day'):
        render_escalation_explorer(ROOT)
    elif view.startswith('F5'):
        render_f5()
    else:
        render_t6()
    insight('Ethical implication:', 'Escalation trades automation against human workload. A safeguard is meaningful only if its achieved coverage and review burden can be measured.')

elif page == STAGES[7]:
    stage_header('STAGE 08 · T7 · F6 · COMPARATIVE ANALYSIS',
                 'How does drift-trigger selection influence adaptive forecasting?',
                 'Team 41 and Team 42 evaluate distribution-shift and forecast-error triggers, respectively. The comparison examines drift-period accuracy, uncertainty reliability, review workload and retraining behaviour, while explicitly accounting for methodological differences between the experimental pipelines.')
    render_t7()
    section('F6 · Trigger disagreement timeline')
    f6_region = st.radio('Region shown in F6', ['SA1', 'NSW1'], horizontal=True, key='f6_region_selector')
    render_f6(f6_region)
    insight('Research significance:', 'The comparison connects trigger behaviour to downstream drift MAE, worst-window coverage, review workload and retraining frequency.')

elif page == STAGES[8]:
    stage_header('STAGE 09 · SYSTEM INTEGRATION & RESEARCH SYNTHESIS',
                 'From drift detection to reliable adaptive forecasting',
                 'This integrated research prototype brings together forecasting, drift monitoring, model adaptation, predictive uncertainty and human-review escalation. Using timestamped outputs from the SA1 and NSW1 experiments, the replay makes the relationships between these components observable and auditable.')

    # The integrated historical replay is the centrepiece of this final stage.
    render_integrated_prototype()

    section('Research synthesis')
    st.write(
        'The combined evaluation demonstrates why concept drift must be assessed beyond detector alarms alone. '
        'The value of adaptation depends on whether retraining improves forecast accuracy relative to its computational cost, '
        'while prediction-interval coverage and human-review workload provide complementary measures of reliability.'
    )
    st.caption(
        'The interactive view replays recorded experimental outputs rather than rerunning the models. '
        'Findings are based on historical SA1 and NSW1 data; wider operational generalisation requires further validation.'
    )

    # Keep the figure gallery as the final element of the Synthesis page.
    with st.expander('Supporting experiment outputs, figures and tables'):
        render_8()
