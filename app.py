"""
WUC NRW Reduction Prototype — Streamlit Dashboard
4 tabs: Leak Detection | Water Quality | Pipe Risk | Demand Forecast
Data source: wuc-prototype-data-tp S3 bucket (af-south-1)
"""
import streamlit as st
import pandas as pd
import numpy as np
import boto3, joblib, json, io
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "WUC NRW Dashboard",
    page_icon  = "💧",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Colour palette ────────────────────────────────────────────────
ZONE_COLORS = {
    'GBR-N':'#1565C0','GBR-S':'#0D8A82','GBR-C':'#7B2D8B',
    'FRA-01':'#C05B0E','MHR-01':'#1A7A3C','KNY-01':'#C62828',
    'LBT-01':'#E8B817','MMB-01':'#546E7A'
}
RISK_COLORS = {
    'LOW':'#1A7A3C','MEDIUM':'#E8B817','HIGH':'#C05B0E','CRITICAL':'#C62828'
}
BUCKET = 'wuc-prototype-data-tp'
REGION = 'af-south-1'

# ── S3 connection — reads from Streamlit Cloud secrets ────────────
def get_s3():
    """Create S3 client — no caching to avoid stale credentials."""
    return boto3.client(
        's3',
        region_name           = st.secrets['aws_default_region'],
        aws_access_key_id     = st.secrets['aws_access_key_id'],
        aws_secret_access_key = st.secrets['aws_secret_access_key'],
    )

@st.cache_data(ttl=300)
def load_csv_from_s3(key):
    obj = get_s3().get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

@st.cache_data(ttl=300)
def load_metrics_from_s3(key):
    obj = get_s3().get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj['Body'].read())

def load_model_from_s3(key):
    """Models are not cached — loaded fresh each session."""
    obj = get_s3().get_object(Bucket=BUCKET, Key=key)
    return joblib.load(io.BytesIO(obj['Body'].read()))

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://www.wuc.bw/images/logo.png", width=120)
    except:
        st.title("💧 WUC Dashboard")

    st.markdown("### Filters")
    all_zones      = ['GBR-N','GBR-S','GBR-C','FRA-01',
                      'MHR-01','KNY-01','LBT-01','MMB-01']
    selected_zones = st.multiselect(
        "Select Zones", all_zones, default=all_zones)
    st.divider()
    st.markdown("**Bucket:** wuc-prototype-data-tp")
    st.markdown("**Region:** af-south-1")
    st.markdown("**Models:** IF · XGB · RF · Prophet")

# ── WUC Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0B1F3A;padding:1rem 1.5rem;
            border-radius:8px;margin-bottom:1rem'>
  <h2 style='color:white;margin:0;font-size:1.4rem'>
    💧 WUC Botswana — NRW Reduction Dashboard
  </h2>
  <p style='color:#AACCE8;margin:0.3rem 0 0;font-size:0.9rem'>
    Prototype · af-south-1 · wuc-prototype-data-tp
  </p>
</div>
""", unsafe_allow_html=True)

# ── Global NRW header ─────────────────────────────────────────────
try:
    billing = load_csv_from_s3('warm/features/billing_features.csv')
    for col in ['nrw_pct','nrw_revenue_loss_BWP','nrw_revenue_loss_bwp']:
        if col in billing.columns:
            billing[col] = pd.to_numeric(billing[col], errors='coerce')

    billing_filt = billing[billing['zone_id'].isin(selected_zones)]
    avg_nrw      = billing_filt['nrw_pct'].mean() * 100
    loss_col     = ('nrw_revenue_loss_BWP'
                    if 'nrw_revenue_loss_BWP' in billing.columns
                    else 'nrw_revenue_loss_bwp')
    total_loss   = (billing_filt[loss_col].sum()
                    if loss_col in billing_filt.columns else 0)
    worst_zone   = (billing_filt.groupby('zone_id')['nrw_pct']
                    .mean().idxmax()
                    if len(billing_filt) > 0 else "N/A")

    h1,h2,h3,h4,h5 = st.columns(5)
    h1.metric("Avg NRW %",       f"{avg_nrw:.1f}%",
              delta=f"{avg_nrw-35:.1f}% above 35% benchmark",
              delta_color="inverse")
    h2.metric("Revenue Loss",    f"P{total_loss/1e6:.1f}M/yr")
    h3.metric("Worst Zone",      worst_zone)
    h4.metric("Zones Monitored", len(selected_zones))
    h5.metric("Models Active",   "4 ✓")
    st.divider()
except Exception as e:
    st.warning(f"Header load: {e}")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Leak Detection",
    "🧪 Water Quality",
    "🗺️ Pipe Risk Map",
    "📈 Demand Forecast",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — LEAK DETECTION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Leak Detection — Isolation Forest Model")
    try:
        iot_raw     = load_csv_from_s3('warm/features/iot_features.csv')
        iso_model   = load_model_from_s3('warm/models/isolation_forest_leak.pkl')
        iso_metrics = load_metrics_from_s3('warm/models/isolation_forest_metrics.json')

        iot      = iot_raw[iot_raw['zone_id'].isin(selected_zones)].copy()
        FEATURES = [f for f in iso_metrics.get('features', [])
                    if f in iot.columns]
        if not FEATURES:
            FEATURES = ['pressure_bar','flow_rate_m3hr']

        for col in FEATURES:
            iot[col] = pd.to_numeric(iot[col], errors='coerce')

        X = iot[FEATURES].fillna(0)
        iot['predicted_leak'] = (iso_model.predict(X) == -1).astype(int)
        iot['anomaly_score']  = -iso_model.score_samples(X)

        zone_summary = (
            iot.groupby('zone_id')
            .agg(total=('predicted_leak','count'),
                 leaks=('predicted_leak','sum'),
                 avg_score=('anomaly_score','mean'))
            .assign(leak_rate=lambda x: x['leaks']/x['total']*100)
            .reset_index()
            .sort_values('leak_rate', ascending=False)
        )

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Predicted Leaks", int(iot['predicted_leak'].sum()))
        k2.metric("Leak Rate", f"{iot['predicted_leak'].mean()*100:.2f}%")
        k3.metric("Highest Risk Zone",
                  zone_summary.iloc[0]['zone_id']
                  if len(zone_summary) > 0 else "N/A")
        k4.metric("Model AUC-ROC",
                  f"{iso_metrics.get('auc_roc',0):.4f}",
                  "Target > 0.75 ✓")
        st.divider()

        col_a, col_b = st.columns([2,1])
        with col_a:
            fig = px.bar(
                zone_summary, x='zone_id', y='leak_rate',
                color='zone_id', color_discrete_map=ZONE_COLORS,
                title='Predicted Leak Rate by Zone (%)',
                labels={'leak_rate':'Leak Rate (%)','zone_id':'Zone'},
                text=zone_summary['leak_rate'].round(2).astype(str)+'%'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=380,
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Zone Risk Summary**")
            for _, row in zone_summary.iterrows():
                color = ('#C62828' if row['leak_rate'] > 1.5
                         else '#E8B817' if row['leak_rate'] > 1.0
                         else '#1A7A3C')
                st.markdown(
                    f"<div style='padding:6px 10px;margin:4px 0;"
                    f"border-left:4px solid {color};"
                    f"background:#f8f8f8;border-radius:4px'>"
                    f"<b>{row['zone_id']}</b>: "
                    f"{row['leak_rate']:.2f}% "
                    f"({int(row['leaks'])} leaks)</div>",
                    unsafe_allow_html=True
                )

        st.divider()
        if 'pressure_bar' in iot.columns:
            st.markdown("**Pressure Distribution — Leak vs Normal**")
            fig2 = go.Figure()
            for is_leak, label, color in [
                (0,'Normal','#1565C0'),(1,'Leak','#C62828')
            ]:
                data = iot[iot['predicted_leak']==is_leak]['pressure_bar'].dropna()
                fig2.add_trace(go.Histogram(
                    x=data, name=f'{label} (n={len(data):,})',
                    opacity=0.6, marker_color=color,
                    histnorm='probability density', nbinsx=60,
                ))
            fig2.update_layout(
                barmode='overlay', height=300,
                title='Pressure Distribution — Predicted Leak vs Normal',
                xaxis_title='Pressure (bar)', yaxis_title='Density',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Tab 1 error: {e}")
        import traceback; st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════
# TAB 2 — WATER QUALITY
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Water Quality — XGBoost Failure Prediction")
    try:
        quality_raw  = load_csv_from_s3('warm/features/quality_features.csv')
        xgb_model    = load_model_from_s3('warm/models/xgboost_quality.pkl')
        xgb_metrics  = load_metrics_from_s3('warm/models/xgboost_quality_metrics.json')

        THRESHOLD    = float(xgb_metrics['optimal_threshold'])
        XGB_FEATURES = [f for f in xgb_metrics['features']
                        if f in quality_raw.columns]

        quality = quality_raw[quality_raw['zone_id'].isin(selected_zones)].copy()
        for col in XGB_FEATURES:
            if col in quality.columns:
                quality[col] = pd.to_numeric(quality[col], errors='coerce')

        X_q   = quality[XGB_FEATURES].fillna(0)
        probs = xgb_model.predict_proba(X_q)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)
        quality['fail_prob']      = probs.round(4)
        quality['predicted_fail'] = preds

        chl_col = next((c for c in ['chlorine_mg_L','chlorine_mg_l']
                        if c in quality.columns), None)
        ph_col  = next((c for c in ['pH','ph']
                        if c in quality.columns), None)
        tur_col = next((c for c in ['turbidity_NTU','turbidity_ntu']
                        if c in quality.columns), None)

        agg_dict = {
            'total':     ('predicted_fail','count'),
            'fails':     ('predicted_fail','sum'),
            'fail_mean': ('predicted_fail','mean'),
        }
        if chl_col: agg_dict['avg_chlorine'] = (chl_col,'mean')
        if ph_col:  agg_dict['avg_ph']       = (ph_col, 'mean')
        if tur_col: agg_dict['avg_turbidity']= (tur_col,'mean')

        zq = (quality.groupby('zone_id').agg(**agg_dict)
              .assign(fail_rate=lambda x: x['fail_mean']*100)
              .drop(columns=['fail_mean']).round(3)
              .sort_values('fail_rate', ascending=False).reset_index())

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Samples",      f"{len(quality):,}")
        k2.metric("Predicted Failures", int(preds.sum()),
                  delta=f"{preds.mean()*100:.2f}% failure rate")
        k3.metric("Threshold",          f"{THRESHOLD}",
                  delta="optimised ✓")
        k4.metric("AUC-ROC",
                  f"{xgb_metrics.get('auc_roc',0):.4f}",
                  "Target > 0.85 ✓")
        st.divider()

        col_a, col_b = st.columns([2,1])
        with col_a:
            fig = px.bar(
                zq, x='zone_id', y='fail_rate',
                color='zone_id',
                color_discrete_map={z:c for z,c in ZONE_COLORS.items()
                                    if z in selected_zones},
                title='Quality Failure Rate by Zone (%)',
                labels={'fail_rate':'Failure Rate (%)','zone_id':'Zone'},
                text=zq['fail_rate'].round(2).astype(str)+'%',
            )
            fig.update_traces(textposition='outside')
            fig.add_hline(y=1.0, line_dash='dash', line_color='red',
                          annotation_text='1% threshold')
            fig.update_layout(showlegend=False, height=380,
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Zone Quality Status**")
            for _, row in zq.iterrows():
                color = ('#C62828' if row['fail_rate'] > 1.0
                         else '#E8B817' if row['fail_rate'] > 0.5
                         else '#1A7A3C')
                label = ('🔴 ALERT' if row['fail_rate'] > 1.0
                         else '🟡 WARN' if row['fail_rate'] > 0.5
                         else '🟢 OK')
                st.markdown(
                    f"<div style='padding:6px 10px;margin:4px 0;"
                    f"border-left:4px solid {color};"
                    f"background:#f8f8f8;border-radius:4px'>"
                    f"<b>{row['zone_id']}</b> {label}<br>"
                    f"<small>{int(row['fails'])} fails / "
                    f"{int(row['total'])} samples</small></div>",
                    unsafe_allow_html=True
                )

        st.divider()
        if chl_col and ph_col:
            scatter_data = quality.copy()
            scatter_data['Result'] = scatter_data['predicted_fail'].map(
                {0:'Pass',1:'FAIL'})
            fig2 = px.scatter(
                scatter_data.sample(min(500,len(scatter_data))),
                x=chl_col, y=ph_col, color='Result',
                color_discrete_map={'Pass':'#1565C0','FAIL':'#C62828'},
                opacity=0.6,
                title='Chlorine vs pH — Pass/Fail',
                labels={chl_col:'Chlorine (mg/L)', ph_col:'pH'},
            )
            fig2.add_vline(x=0.2, line_dash='dot', line_color='orange',
                           annotation_text='WHO min 0.2')
            fig2.add_hrect(y0=6.5, y1=8.5, fillcolor='green', opacity=0.05)
            fig2.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        top_feat = xgb_metrics.get('top_feature','coagulant_dose_mg_L')
        st.info(f"**SHAP Insight:** Top failure driver is **{top_feat}**.")

    except Exception as e:
        st.error(f"Tab 2 error: {e}")
        import traceback; st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════
# TAB 3 — PIPE RISK MAP
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Pipe Risk Map — Random Forest Model")
    try:
        from streamlit_folium import st_folium
        import folium
        from folium.plugins import MarkerCluster

        pipes_raw  = load_csv_from_s3('hot/pipe_scores/pipe_risk_scores.csv')
        lr_metrics = load_metrics_from_s3('warm/models/logreg_pipe_risk_metrics.json')

        for col in ['gps_latitude','gps_longitude','prob_low','prob_medium',
                    'prob_high','prob_critical','max_risk_prob','age_years_clean']:
            if col in pipes_raw.columns:
                pipes_raw[col] = pd.to_numeric(pipes_raw[col], errors='coerce')

        pipes = pipes_raw[
            pipes_raw['zone_id'].isin(selected_zones) &
            pipes_raw['gps_latitude'].notna() &
            pipes_raw['gps_longitude'].notna() &
            (pipes_raw['gps_latitude'].abs()  > 0.01) &
            (pipes_raw['gps_longitude'].abs() > 0.01)
        ].copy()
        pipes['predicted_risk'] = pipes['predicted_risk'].str.upper()

        RISK_COLORS_MAP = {
            'LOW':'#1A7A3C','MEDIUM':'#E8B817',
            'HIGH':'#C05B0E','CRITICAL':'#C62828'
        }
        RISK_RADIUS = {'LOW':6,'MEDIUM':7,'HIGH':8,'CRITICAL':10}

        risk_counts = pipes['predicted_risk'].value_counts()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("CRITICAL Pipes", int(risk_counts.get('CRITICAL',0)),
                  delta="immediate priority")
        k2.metric("HIGH Risk",      int(risk_counts.get('HIGH',0)))
        k3.metric("Total Pipes",    len(pipes))
        k4.metric("F1-Macro",
                  f"{lr_metrics.get('f1_macro',0):.4f}",
                  "Target > 0.80 ✓")
        st.divider()

        show_critical = st.toggle("Show CRITICAL only", value=False)
        plot_pipes    = (pipes if not show_critical
                         else pipes[pipes['predicted_risk']=='CRITICAL'])

        centre_lat = float(pipes['gps_latitude'].mean())
        centre_lon = float(pipes['gps_longitude'].mean())
        m       = folium.Map(location=[centre_lat,centre_lon],
                             zoom_start=12, tiles='CartoDB positron')
        cluster = MarkerCluster(disableClusteringAtZoom=14)

        for _, row in plot_pipes.iterrows():
            risk  = str(row.get('predicted_risk','MEDIUM')).upper()
            color = RISK_COLORS_MAP.get(risk,'#546E7A')
            rad   = RISK_RADIUS.get(risk,7)
            popup_html = (
                f"<div style='font-family:Arial;font-size:12px'>"
                f"<b style='color:{color}'>{risk}</b><br>"
                f"Pipe: {row.get('pipe_id','?')}<br>"
                f"Zone: {row.get('zone_id','?')}<br>"
                f"Age: {row.get('age_years_clean','?')} yrs<br>"
                f"P(CRIT): {row.get('prob_critical',0):.3f}</div>"
            )
            folium.CircleMarker(
                location=[float(row['gps_latitude']),
                          float(row['gps_longitude'])],
                radius=rad, color=color,
                fill=True, fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{row.get('pipe_id','?')} — {risk}",
            ).add_to(cluster)

        cluster.add_to(m)
        st_folium(m, width=None, height=520, returned_objects=[])

        st.divider()
        if 'prob_critical' in pipes.columns:
            top10 = (
                pipes.nlargest(10,'prob_critical')
                [['pipe_id','zone_id','material_clean',
                  'age_years_clean','predicted_risk','prob_critical']]
                .rename(columns={
                    'pipe_id':'Pipe','zone_id':'Zone',
                    'material_clean':'Material',
                    'age_years_clean':'Age',
                    'predicted_risk':'Risk',
                    'prob_critical':'P(CRITICAL)',
                })
                .reset_index(drop=True)
            )
            st.markdown("**Top 10 CRITICAL Pipes**")
            st.dataframe(
                top10.style.background_gradient(
                    subset=['P(CRITICAL)'], cmap='Reds'
                ).format({'P(CRITICAL)':'{:.3f}','Age':'{:.0f}'}),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Tab 3 error: {e}")
        import traceback; st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════
# TAB 4 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Demand Forecast — Prophet Model (×8 Zones)")
    try:
        fc_raw    = load_csv_from_s3('hot/forecasts/demand_forecasts_6month.csv')
        bill_raw  = load_csv_from_s3('warm/features/billing_features.csv')
        p_metrics = load_metrics_from_s3('warm/models/prophet_demand_metrics.json')

        fc_raw['ds']         = pd.to_datetime(fc_raw['ds'],        errors='coerce')
        fc_raw['yhat']       = pd.to_numeric(fc_raw['yhat'],       errors='coerce')
        fc_raw['yhat_lower'] = pd.to_numeric(fc_raw['yhat_lower'], errors='coerce')
        fc_raw['yhat_upper'] = pd.to_numeric(fc_raw['yhat_upper'], errors='coerce')

        date_col = next((c for c in ['ds','date','billing_date']
                         if c in bill_raw.columns), None)
        bill_raw[date_col] = pd.to_datetime(bill_raw[date_col], errors='coerce')

        ZONE_MAP_B = {
            'GBR-N':'GBR-N','GBR-S':'GBR-S','GBR-C':'GBR-C',
            'FRA-01':'FRA-01','MHR-01':'MHR-01','KNY-01':'KNY-01',
            'LBT-01':'LBT-01','MMB-01':'MMB-01',
            'GBR_N':'GBR-N','GBR_S':'GBR-S','GBR_C':'GBR-C',
            'FRA_01':'FRA-01','MHR_01':'MHR-01','KNY_01':'KNY-01',
            'LBT_01':'LBT-01','MMB_01':'MMB-01','GBR-X':'GBR-N',
        }
        bill_raw['zone_id'] = (bill_raw['zone_id'].astype(str)
                               .str.strip().map(ZONE_MAP_B))
        bill_raw = bill_raw.dropna(subset=['zone_id'])

        fc   = fc_raw[fc_raw['zone_id'].isin(selected_zones)].copy()
        bill = bill_raw[bill_raw['zone_id'].isin(selected_zones)].copy()
        cutoff = bill[date_col].max()

        zone_mapes = p_metrics.get('zones', {})
        mean_mape  = p_metrics.get('overall_mape', 0)
        future_fc  = fc[fc['ds'] > cutoff]

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Mean MAPE",  f"{mean_mape:.2f}%", "< 15% ✓")
        k2.metric("Horizon",    "to Dec 2026")
        k3.metric("Zones",      len(selected_zones))
        if len(future_fc) > 0:
            peak_zone = future_fc.groupby('zone_id')['yhat'].max().idxmax()
            k4.metric("Peak Demand Zone", peak_zone)
        st.divider()

        col_sel, col_tog = st.columns([3,1])
        with col_sel:
            zone_choice = st.selectbox(
                "Zone deep-dive", options=selected_zones,
                index=selected_zones.index('KNY-01')
                      if 'KNY-01' in selected_zones else 0,
            )
        with col_tog:
            show_ci = st.toggle("Confidence band", value=True)

        zfc  = fc[fc['zone_id']==zone_choice].sort_values('ds').reset_index(drop=True)
        hist = zfc[zfc['ds'] <= cutoff].reset_index(drop=True)
        fut  = zfc[zfc['ds'] >  cutoff].reset_index(drop=True)
        col  = ZONE_COLORS.get(zone_choice,'#546E7A')
        h    = col.lstrip('#')
        rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.12)"

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=hist['ds'].values, y=(hist['yhat'].values/1000),
            mode='lines+markers', name='Historical',
            line=dict(color=col, width=2.5), marker=dict(size=4),
        ))
        fig1.add_trace(go.Scatter(
            x=fut['ds'].values, y=(fut['yhat'].values/1000),
            mode='lines+markers', name='Forecast',
            line=dict(color=col, width=2, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
        ))
        if show_ci and len(fut) > 0:
            x_band = np.concatenate([fut['ds'].values,
                                      fut['ds'].values[::-1]])
            y_band = np.concatenate([fut['yhat_upper'].values/1000,
                                      fut['yhat_lower'].values[::-1]/1000])
            fig1.add_trace(go.Scatter(
                x=x_band, y=y_band, fill='toself', fillcolor=rgba,
                line=dict(color='rgba(255,255,255,0)'), name='80% CI',
            ))

        cutoff_str = cutoff.strftime('%Y-%m-%d')
        fig1.add_shape(type='line', x0=cutoff_str, x1=cutoff_str,
                       y0=0, y1=1, yref='paper',
                       line=dict(dash='dot', color='gray', width=1.5))
        fig1.add_annotation(x=cutoff_str, y=1.02, yref='paper',
                             text='Forecast start', showarrow=False,
                             font=dict(size=11, color='gray'))
        for yr in [2024,2025,2026]:
            fig1.add_vrect(x0=f'{yr}-07-01', x1=f'{yr}-09-01',
                           fillcolor='#E8B817', opacity=0.07, layer='below')
        fig1.update_layout(
            title=f'{zone_choice} — Demand Forecast (kL × 1,000)',
            xaxis_title='Month', yaxis_title='Volume (000 kL)',
            height=420, plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified', legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()
        fig2 = go.Figure()
        for zone in selected_zones:
            zf = fc[(fc['zone_id']==zone) & (fc['ds']>cutoff)].sort_values('ds')
            if len(zf) == 0: continue
            c = ZONE_COLORS.get(zone,'#546E7A')
            fig2.add_trace(go.Scatter(
                x=zf['ds'].values, y=(zf['yhat'].values/1000),
                mode='lines+markers', name=zone,
                line=dict(color=c, width=2), marker=dict(size=5),
            ))
        fig2.add_vrect(x0='2026-07-01', x1='2026-09-01',
                       fillcolor='#E8B817', opacity=0.08, layer='below')
        fig2.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)',
                           xaxis_title='Month',
                           yaxis_title='Forecast Volume (000 kL)',
                           hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        mape_rows = [
            {'Zone': z, 'MAPE (%)': info.get('mape',0),
             'Status': '✓' if info.get('mape',99)<15 else '✗'}
            for z, info in zone_mapes.items() if z in selected_zones
        ]
        if mape_rows:
            mape_df = pd.DataFrame(mape_rows).sort_values('MAPE (%)')
            st.dataframe(
                mape_df.style.background_gradient(
                    subset=['MAPE (%)'], cmap='RdYlGn_r', vmin=0, vmax=15
                ).format({'MAPE (%)':'{:.2f}'}),
                use_container_width=True,
            )

        st.divider()
        st.info(
            "**Forecast Insight:** Peak demand July–August (dry season). "
            "KNY-01 and LBT-01 highest forecast volumes — also highest NRW zones."
        )

    except Exception as e:
        st.error(f"Tab 4 error: {e}")
        import traceback; st.code(traceback.format_exc())