from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Concept Drift Lab | Team 42', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FINAL = RESULTS / 'tables/adaptation/revised/pgs242_128_five_seed_final'

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
        st.image(str(path), caption=caption or path.name, use_column_width=True)
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
    regime = st.radio('Regime evaluation', ['Full', 'Local'], horizontal=True)
    path = FINAL / 't3_full_and_local' / f't3_{regime.lower()}.csv'
    df = filter_region(load(path), region)
    st.caption('Values are five-seed mean ± SD as stored in the official T3 table.')
    table(df, 't3_'+regime.lower()+'_'+region)
    for name in ['t3_seed_metrics_full.csv' if regime=='Full' else 't3_seed_metrics_local.csv', 'regime_observation_counts.csv']:
        with st.expander(name):
            table(filter_region(load(FINAL / 't3_full_and_local' / name), region), 't3_'+region+'_'+name[:-4])

def render_4():
    st.header('T4 · Adaptation cost')
    region = region_control('t4_region')
    df = filter_region(load(FINAL / 't4/t4_adaptation_cost.csv'), region)
    st.caption('Official five-seed summary. Negative MAE gain vs Arm A indicates lower (better) MAE.')
    table(df, 't4_cost_'+region)
    detail = filter_region(load(FINAL / 'summary_all_seeds.csv'), region)
    if not detail.empty:
        st.subheader('Interactive seed-level comparisons')
        seed = st.selectbox('Seed', sorted(detail['seed'].dropna().unique().tolist()), key='t4_seed')
        subset = detail[detail['seed']==seed]
        number_chart(subset, 'configuration', 'overall_mae', 'MAE by adaptation configuration')
        number_chart(subset, 'configuration', 'n_retrains', 'Retraining events by configuration')
        table(subset, 't4_seed_'+region+'_'+str(seed))
    show_image(RESULTS / 'figures/adaptation/T4_cost_vs_accuracy.png', 'T4 · Adaptation cost versus forecasting accuracy')

def render_5():
    st.header('F3 · Four-arm comparison')
    region = region_control('f3_region')
    detector = st.selectbox('Detector', ['adwin','kswin','page_hinkley'], key='f3_detector')
    show_image(RESULTS / f'figures/adaptation/F3_four_arms/F3_{region}_four_arm_comparison_{detector}_180d.png')
    st.caption('F3 · Saved four-arm comparison figure from the experiment repository.')
    df = filter_region(load(FINAL / 'summary_all_seeds.csv'), region)
    if not df.empty:
        df = df[(df['arm'].isin(['A','B'])) | (df['detector'].astype(str)==detector)]
        st.subheader('Five-seed experiment rows for selected detector')
        table(df, 'f3_five_seed_'+region+'_'+detector)

def render_6():
    st.header('Interactive actual vs forecast')
    region = region_control('forecast_region')
    seed = st.selectbox('Seed', [1,2,3,4,5], key='forecast_seed')
    path = FINAL / f'seed_{seed}' / f'forecasts_{region}.csv.gz'
    @st.cache_data(show_spinner='Loading forecast data…')
    def load_forecasts(p):
        data = pd.read_csv(p, parse_dates=['SETTLEMENTDATE'])
        return data
    if path.exists():
        df = load_forecasts(str(path))
        config = choice(df, 'configuration', 'Adaptation configuration', 'forecast_config')
        if config:
            df = df[df['configuration'].astype(str)==config].copy()
        if not df.empty:
            low = df['SETTLEMENTDATE'].min().date()
            high = df['SETTLEMENTDATE'].max().date()
            dates = st.date_input('Date range', value=(low, min(low + pd.Timedelta(days=14),high)), min_value=low, max_value=high, key='forecast_dates')
            if len(dates)==2:
                start, end = dates
                view = df[(df['SETTLEMENTDATE'].dt.date >= start) & (df['SETTLEMENTDATE'].dt.date <= end)].copy()
                if not view.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'], y=view['y_true'], name='Actual demand', mode='lines'))
                    fig.add_trace(go.Scatter(x=view['SETTLEMENTDATE'], y=view['y_pred'], name='Forecast', mode='lines'))
                    if 'retrain_indicator' in view:
                        retrains = view[view['retrain_indicator'].astype(str).str.lower().isin(['true','1'])]
                        if len(retrains):
                            fig.add_trace(go.Scatter(x=retrains['SETTLEMENTDATE'], y=retrains['y_pred'], name='Retraining', mode='markers', marker={'symbol':'diamond','size':10}))
                    fig.update_layout(height=550, xaxis_title='Settlement date', yaxis_title='Electricity demand (MW)', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric('Selected period MAE', f"{(view['y_true']-view['y_pred']).abs().mean():.2f} MW")
                    st.caption('Period MAE is calculated interactively from the selected rows, not the official full-test T3 value.')
                    table(view, f'forecast_{region}_seed{seed}_{config}')
                else:
                    st.info('No forecasts in selected dates.')
            else:
                st.info('Choose a start and end date.')
    else:
        st.warning('Forecast file is unavailable.')

def render_7():
    region = region_control('t5_region')
    t5_dir = RESULTS / 'tables/uncertainty/t5_aemo/tables'
    files = sorted(t5_dir.glob(f'T5_{region}_*.csv')) if t5_dir.exists() else []
    if not files:
        st.info('T5 results are not available for this region.')
        return

    configurations = {p.stem.removeprefix(f'T5_{region}_'): p for p in files}
    config = st.selectbox('Adaptation configuration', list(configurations), key='t5_configuration')
    df = load(configurations[config])
    required = ['UQ method', 'regime', 'coverage', 'mean width', 'normalised width', 'worst 24h coverage']
    if df.empty or not all(col in df.columns for col in required):
        st.warning('The selected T5 table does not have the expected evaluation columns.')
        return

    def split_mean_sd(series):
        parts = series.astype(str).str.split('±', n=1, expand=True)
        mean = pd.to_numeric(parts[0].str.strip(), errors='coerce')
        sd = pd.to_numeric(parts[1].str.strip(), errors='coerce') if parts.shape[1] > 1 else pd.Series(0.0, index=series.index)
        return mean, sd.fillna(0)

    numeric = df.copy()
    metrics = ['coverage', 'mean width', 'normalised width', 'worst 24h coverage']
    for metric in metrics:
        numeric[metric], numeric[metric + '_sd'] = split_mean_sd(df[metric])
    regimes = ['pre-drift', 'drift', 'post-drift']
    numeric['regime'] = pd.Categorical(numeric['regime'], categories=regimes, ordered=True)
    numeric = numeric.sort_values(['regime', 'UQ method'])

    methods = numeric['UQ method'].dropna().unique().tolist()
    selected_methods = st.multiselect('Uncertainty method', methods, default=methods, key='t5_methods')
    plot_data = numeric[numeric['UQ method'].isin(selected_methods)].copy()
    if plot_data.empty:
        st.info('Select an uncertainty method to view the results.')
        return

    drift_data = plot_data[plot_data['regime'] == 'drift']
    metric_cols = st.columns(3)
    metric_cols[0].metric('Drift coverage', f"{drift_data['coverage'].mean():.2%}" if len(drift_data) else '—')
    metric_cols[1].metric('Drift mean interval width', f"{drift_data['mean width'].mean():.1f} MW" if len(drift_data) else '—')
    metric_cols[2].metric('Worst 24h coverage in drift', f"{drift_data['worst 24h coverage'].min():.2%}" if len(drift_data) else '—')

    colors = ['#244766', '#6eafad', '#668fb4', '#8c9fae']

    def t5_chart(metric, title, percent=False, target=False):
        fig = go.Figure()
        for i, method in enumerate(selected_methods):
            rows = plot_data[plot_data['UQ method'] == method]
            fig.add_trace(go.Bar(
                name=method.replace('_', ' ').title(),
                x=rows['regime'].astype(str),
                y=rows[metric] * (100 if percent else 1),
                error_y=dict(type='data', array=rows[metric + '_sd'] * (100 if percent else 1), visible=True),
                marker_color=colors[i % len(colors)],
                hovertemplate='%{x}<br>%{y:.2f}' + ('%' if percent else '') + '<extra>%{fullData.name}</extra>',
            ))
        if target:
            fig.add_hline(y=90, line_dash='dash', line_color='#9b6470', annotation_text='90% target', annotation_position='top left')
        fig.update_layout(title=title, barmode='group', height=410, plot_bgcolor='white',
                          paper_bgcolor='white', font_color='#244766',
                          xaxis_title='Evaluation regime', yaxis_title='Coverage (%)' if percent else title,
                          legend_title_text='Method', margin=dict(t=75, b=45))
        fig.update_xaxes(categoryorder='array', categoryarray=regimes)
        st.plotly_chart(fig, use_container_width=True)

    t5_chart('coverage', 'Empirical coverage', percent=True, target=True)
    t5_chart('worst 24h coverage', 'Worst rolling 24-hour coverage', percent=True)
    t5_chart('mean width', 'Mean prediction interval width (MW)')
    t5_chart('normalised width', 'Normalised prediction interval width')
    st.caption('Bars show five-seed means with ±1 standard deviation. The 90% line is the nominal coverage target; worst-window results describe observed short-period reliability.')
    st.subheader('T5 · Regime-level results')
    st.dataframe(df[df['UQ method'].isin(selected_methods)], use_container_width=True, hide_index=True)
    st.download_button('Download T5 table (CSV)', df.to_csv(index=False).encode(),
                       file_name=f'T5_{region}_{config}.csv', mime='text/csv', key='t5_download')

def render_8():
    region = region_control('gallery_region')
    paths = sorted((RESULTS / 'figures').rglob('*.png'))
    paths = [p for p in paths if 'archive' not in p.parts and (region in p.name or not any(r in p.name for r in ['SA1','NSW1']))]
    category = st.selectbox('Figure category', ['All','degradation','detection','adaptation','diagnosis'], key='gallery_category')
    if category != 'All':
        paths = [p for p in paths if category in p.parts]
    if paths:
        selected = st.selectbox('Saved figure', range(len(paths)),
                                format_func=lambda i: str(paths[i].relative_to(RESULTS)),
                                key='gallery_figure')
        show_image(paths[selected])
    else:
        st.info('No figures are available for this selection.')


STAGES = [
    '01 · Project Overview',
    '02 · Forecast Degradation',
    '03 · Drift Detection',
    '04 · Forecast Adaptation',
    '05 · Uncertainty & Reliability',
    '06 · Human Escalation',
    '07 · Trigger Comparison',
    '08 · Ethics & Limitations',
    '09 · Conclusions & Future Work',
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
    st.caption('Results are read from the local experiment repository.')

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
    stage_header('STAGE 02 · F1', 'Why does forecasting performance degrade?',
                 'Use the rolling forecast error of frozen baseline models to establish the problem and its relationship to documented demand events.')
    note('<strong>Research purpose:</strong> A sustained or abrupt increase in rolling MAE motivates monitoring, but does not by itself establish the cause of the change.')
    render_0()
    insight('Research significance:', 'Rolling MAE tracks how prediction error changes over time. Event annotations offer historical context, but do not establish that an event caused an increase in error.')

elif page == STAGES[2]:
    stage_header('STAGE 03 · T1 · T2 · F2', 'Can meaningful changes be detected?',
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

elif page == STAGES[3]:
    stage_header('STAGE 04 · F3 · T3 · T4', 'When is retraining worthwhile?',
                 'Compare frozen, periodic and drift-triggered forecasting on the same test stream, then examine regime-specific accuracy and computational cost.')
    note('<strong>Four-arm design:</strong> A — never retrain; B — every 30 days; C — on detected drift using full history; D — on detected drift using a recent window. Arms A and B are essential controls.')
    mode=st.radio('Explore adaptation', ['Interactive forecasts', 'F3 · Four-arm comparison', 'T3 · Forecast accuracy', 'T4 · Adaptation cost'], horizontal=True, key='stage_adapt')
    if mode=='Interactive forecasts':
        render_6()
        insight('Research significance:', 'The forecast trajectories show actual and predicted electricity demand for the selected region, run and retraining policy. Models may initially produce identical predictions before their first retraining event.')
    elif mode.startswith('F3'):
        render_5()
        insight('Research significance:', 'The four forecasting policies are compared against the same demand stream. Retraining markers distinguish scheduled updates from detector-triggered adaptation.')
    elif mode.startswith('T3'):
        render_3()
        insight('Research significance:', 'Regime-specific MAE separates pre-drift, drift and post-drift performance; repeated seeds quantify variability rather than relying on a single run.')
    else:
        render_4()
        insight('Research significance:', 'Retraining count, cumulative training samples and runtime contextualise MAE differences. More frequent retraining is not necessarily more effective or efficient.')

elif page == STAGES[4]:
    stage_header('STAGE 05 · T5 · F4', 'Do prediction intervals remain reliable during drift?',
                 'Study static and adaptive conformal uncertainty, including interval coverage, width and how calibration changes over time.')
    section('T5 · Five-seed evaluation by regime')
    render_7()
    insight('Research significance:', 'Coverage measures how often observed demand falls inside prediction intervals; width captures their precision. Rolling and worst-window coverage reveal failures that may be hidden by full-period averages.')
    future_output('F4','Rolling coverage trajectory',
        'Rolling empirical coverage against the 90% target, with documented shifts annotated.', 'F4*.png')

elif page == STAGES[5]:
    stage_header('STAGE 06 · T6 · F5', 'When should forecasts be escalated for human review?',
                 'Evaluate the trade-off between the fraction of cases referred for review and the coverage achieved at different operating points.')
    note('<strong>Human-review operating points:</strong> Review rate measures the proportion of forecasts referred for intervention. A target coverage level can be unreachable at some review budgets.')
    future_output('T6','Escalation operating points',
        'Review rates required for 90% and 95% coverage, alongside coverage at fixed review budgets.', 'T6*.csv')
    future_output('F5','Escalation trade-off curves',
        'Empirical coverage versus human-review rate, separated by stable and drift regimes.', 'F5*.png')
    insight('Ethical implication:', 'Escalation trades automation against human workload. A safeguard is meaningful only if its achieved coverage and review burden can be measured.')

elif page == STAGES[6]:
    stage_header('STAGE 07 · T7 · F6', 'Does the choice of drift trigger change downstream outcomes?',
                 'Compare Team 42 and Team 41 on a shared experimental protocol, separating trigger behaviour from other experimental differences.')
    note('<strong>Cross-team experimental design:</strong> Both trigger approaches are assessed with aligned forecasting models, features, data splits, seeds and retraining budgets so that trigger effects can be isolated.')
    future_output('T7','Cross-team trigger comparison',
        'Cross-team drift MAE, worst 24-hour coverage, review rate, detector alerts and retraining counts.', 'T7*.csv')
    future_output('F6','Trigger disagreement timeline',
        'Detection timelines for both trigger methods, including periods of disagreement.', 'F6*.png')
    insight('Research significance:', 'The comparison connects trigger behaviour to downstream drift MAE, worst-window coverage, review workload and retraining frequency.')

elif page == STAGES[7]:
    stage_header('STAGE 08 · RESPONSIBLE AI', 'Ethical, societal and security implications',
                 'Apply reliability, transparency, security and human-oversight lenses to specific risks identified in adaptive electricity-demand forecasting.')
    risks=[
        ('Unnecessary detector alerts','Spurious triggers can waste computational resources or create false confidence in model changes.','Compare matched and unmatched detections, constrain false alarms and use a retraining refractory period.','Implemented evaluation controls; operational alert policy is future work.'),
        ('Regional generalisation','A strategy evaluated on SA1 may not behave similarly in NSW1 or new market regimes.','Report regional results separately; validate transfer before using one region as evidence for another.','Region-specific reporting is available.'),
        ('Corrupt or missing measurements','Data integrity problems could affect both forecasts and drift-triggered retraining.','Validate ordering and missingness; retain auditable experiment outputs and monitor input quality.','Input-integrity monitoring is not part of the current research prototype.'),
        ('Overconfident forecasts','A nominal 90% interval can still fail badly during short periods of change.','Monitor rolling and worst-case coverage alongside interval width; evaluate human escalation rules.','T5/T6 and F4/F5 are still being finalised.'),
    ]
    for idx,(title,risk,mitigation,status) in enumerate(risks,1):
        with st.expander(f'{idx:02d} · {title}',expanded=True):
            st.markdown(f'**Real-world risk:** {risk}\n\n**Justified mitigation:** {mitigation}\n\n**Current scope:** {status}')
    insight('Responsible interpretation:', 'Historical experiment outputs support research evaluation; grid operations, automated alert handling and human-review deployment remain outside the present system.')

elif page == STAGES[8]:
    stage_header('STAGE 09 · SYNTHESIS', 'Research conclusions & future work',
                 'Connect the experimental findings back to the research objectives, practical value, methodological limitations and responsible deployment requirements.')
    section('What the available evidence can answer')
    for title,body in [
        ('Detection','Synthetic tests characterise responsiveness and false alarms. Documented AEMO events provide useful but incomplete context for real-world drift matching.'),
        ('Adaptation','The four-arm comparison evaluates whether drift-triggered retraining can justify its accuracy gains relative to periodic retraining and the cost of updates.'),
        ('Uncertainty','Static and adaptive conformal intervals are evaluated against a 90% target; regime-specific and worst-window results will provide a more stringent view of reliability.'),
        ('Human review and cross-team comparison','T6/F5 will quantify the human workload of escalation. T7/F6 will examine whether differing triggers change downstream forecasting outcomes.')]:
        note(f'<strong>{title}:</strong> {body}')
    section('Current limitations and next steps')
    st.write('Event annotations are incomplete reference data, not exhaustive changepoint truth. The generality of results beyond SA1 and NSW1 has not been established. Upcoming required outputs **T5–T7 and F4–F6** will strengthen regime-level reliability, escalation and cross-team comparison.')
    with st.expander('Supporting experiment outputs and figure gallery'):
        render_8()
    insight('Overall conclusion:', 'Useful adaptation cannot be determined from detection counts alone. Forecast accuracy, retraining effort, interval reliability and human-review requirements together determine the practical value of the approach.')
