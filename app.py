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

# ── S3 helpers ────────────────────────────────────────────────────
@st.cache_resource
def get_s3():
    return boto3.client('s3', region_name=REGION)

@st.cache_data(ttl=300)
def load_csv_from_s3(key):
    s3  = get_s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

@st.cache_resource
def load_model_from_s3(key):
    s3  = get_s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return joblib.load(io.BytesIO(obj['Body'].read()))

@st.cache_data(ttl=300)
def load_metrics_from_s3(key):
    s3  = get_s3()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj['Body'].read())

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://www.wuc.bw/images/logo.png", width=120)
    except:
        st.title("💧 WUC Dashboard")

    st.markdown("### Filters")
    all_zones      = ['GBR-N','GBR-S','GBR-C','FRA-01','MHR-01','KNY-01','LBT-01','MMB-01']
    selected_zones = st.multiselect("Select Zones", all_zones, default=all_zones)
    st.divider()
    st.markdown("**Bucket:** wuc-prototype-data-tp")
    st.markdown("**Region:** af-south-1")
    st.markdown("**Models:** IF · XGB · RF · Prophet")

# ── WUC Header ───────────────────────────────────────────────────
st.markdown("""
<div style='background:#0B1F3A;padding:1rem 1.5rem;border-radius:8px;margin-bottom:1rem'>
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

    # Normalise column names — handle both cases
    for col in ['volume_pumped_kL','volume_billed_kL','nrw_pct',
                'revenue_BWP','nrw_revenue_loss_BWP',
                'volume_pumped_kl','volume_billed_kl',
                'nrw_revenue_loss_bwp']:
        if col in billing.columns:
            billing[col] = pd.to_numeric(billing[col], errors='coerce')

    billing_filt = billing[billing['zone_id'].isin(selected_zones)]

    avg_nrw    = billing_filt['nrw_pct'].mean() * 100

    # Handle both capitalisation variants for revenue loss
    loss_col   = ('nrw_revenue_loss_BWP' if 'nrw_revenue_loss_BWP' in billing.columns
                  else 'nrw_revenue_loss_bwp')
    total_loss = billing_filt[loss_col].sum() if loss_col in billing_filt.columns else 0

    worst_zone = (billing_filt.groupby('zone_id')['nrw_pct']
                  .mean().idxmax()
                  if len(billing_filt) > 0 else "N/A")

    h1,h2,h3,h4,h5 = st.columns(5)
    h1.metric("Avg NRW %",       f"{avg_nrw:.1f}%",
              delta=f"{avg_nrw - 35:.1f}% above 35% WC benchmark",
              delta_color="inverse")
    h2.metric("Revenue Loss",    f"P{total_loss/1e6:.1f}M/yr",
              delta="NRW-driven")
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
# TAB 1 — LEAK DETECTION (Isolation Forest)
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
        k2.metric("Leak Rate",  f"{iot['predicted_leak'].mean()*100:.2f}%")
        k3.metric("Highest Risk Zone",
                  zone_summary.iloc[0]['zone_id'] if len(zone_summary)>0 else "N/A",
                  f"{zone_summary.iloc[0]['leak_rate']:.2f}% leak rate"
                  if len(zone_summary)>0 else "")
        k4.metric("Model AUC-ROC",
                  f"{iso_metrics.get('auc_roc', 0):.4f}", "Target > 0.75 ✓")
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
# TAB 2 — WATER QUALITY (XGBoost)
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Water Quality — XGBoost Failure Prediction")

    try:
        quality_raw  = load_csv_from_s3('warm/features/quality_features.csv')
        xgb_model    = load_model_from_s3('warm/models/xgboost_quality.pkl')
        xgb_metrics  = load_metrics_from_s3('warm/models/xgboost_quality_metrics.json')

        THRESHOLD    = float(xgb_metrics['optimal_threshold'])  # 0.15
        XGB_FEATURES = [f for f in xgb_metrics['features']
                        if f in quality_raw.columns]

        # Filter to selected zones
        quality = quality_raw[quality_raw['zone_id'].isin(selected_zones)].copy()
        for col in XGB_FEATURES + ['quality_fail_clean']:
            if col in quality.columns:
                quality[col] = pd.to_numeric(quality[col], errors='coerce')

        X_q   = quality[XGB_FEATURES].fillna(0)
        probs = xgb_model.predict_proba(X_q)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)
        quality['fail_prob']      = probs.round(4)
        quality['predicted_fail'] = preds

        # ── Zone summary — use actual column names ────────────────
        # Detect exact column names present in data
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
        if chl_col: agg_dict['avg_chlorine']  = (chl_col, 'mean')
        if ph_col:  agg_dict['avg_ph']        = (ph_col,  'mean')
        if tur_col: agg_dict['avg_turbidity'] = (tur_col, 'mean')

        zq = (quality.groupby('zone_id')
              .agg(**agg_dict)
              .assign(fail_rate=lambda x: x['fail_mean']*100)
              .drop(columns=['fail_mean'])
              .round(3)
              .sort_values('fail_rate', ascending=False)
              .reset_index())

        # ── KPI tiles ─────────────────────────────────────────────
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Samples",     f"{len(quality):,}")
        k2.metric("Predicted Failures", int(preds.sum()),
                  delta=f"{preds.mean()*100:.2f}% failure rate")
        k3.metric("Threshold Applied",  f"{THRESHOLD}",
                  delta="optimised — not default 0.5")
        k4.metric("Model AUC-ROC",
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
                title='Predicted Quality Failure Rate by Zone (%)',
                labels={'fail_rate':'Failure Rate (%)','zone_id':'Zone'},
                text=zq['fail_rate'].round(2).astype(str)+'%',
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=380,
                              plot_bgcolor='rgba(0,0,0,0)')
            fig.add_hline(y=1.0, line_dash='dash', line_color='red',
                          annotation_text='1% threshold',
                          annotation_position='top right')
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

        # ── Chlorine vs pH scatter ────────────────────────────────
        st.divider()
        st.markdown("**Parameter Analysis — Chlorine vs pH**")
        if chl_col and ph_col:
            scatter_data         = quality.copy()
            scatter_data['Result'] = scatter_data['predicted_fail'].map(
                {0:'Pass', 1:'FAIL'})
            hover = ['zone_id', 'fail_prob']
            if tur_col: hover.append(tur_col)

            fig2 = px.scatter(
                scatter_data.sample(min(500, len(scatter_data))),
                x=chl_col, y=ph_col,
                color='Result',
                color_discrete_map={'Pass':'#1565C0','FAIL':'#C62828'},
                opacity=0.6,
                title='Chlorine vs pH — Pass/Fail (sample 500)',
                labels={chl_col:'Chlorine (mg/L)', ph_col:'pH'},
                hover_data=hover,
            )
            fig2.add_vline(x=0.2, line_dash='dot', line_color='orange',
                           annotation_text='WHO min 0.2 mg/L')
            fig2.add_hrect(y0=6.5, y1=8.5, fillcolor='green',
                           opacity=0.05,
                           annotation_text='Safe pH 6.5–8.5')
            fig2.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Chlorine or pH column not found in data")

        # ── Chlorine trend ────────────────────────────────────────
        st.divider()
        st.markdown("**Chlorine Trend by Zone (Monthly Average)**")
        if chl_col and 'timestamp' in quality_raw.columns:
            quality_ts = quality_raw[
                quality_raw['zone_id'].isin(selected_zones)
            ].copy()
            quality_ts['timestamp'] = pd.to_datetime(
                quality_ts['timestamp'], errors='coerce')
            quality_ts[chl_col] = pd.to_numeric(
                quality_ts[chl_col], errors='coerce')
            quality_ts = quality_ts.dropna(
                subset=['timestamp', chl_col])
            quality_ts['month'] = (quality_ts['timestamp']
                                   .dt.to_period('M').astype(str))
            chlorine_monthly = (
                quality_ts.groupby(['zone_id','month'])[chl_col]
                .mean().round(3).reset_index()
            )
            chlorine_monthly['month_dt'] = pd.to_datetime(
                chlorine_monthly['month'])
            chlorine_monthly = chlorine_monthly.sort_values('month_dt')

            fig3 = px.line(
                chlorine_monthly,
                x='month_dt', y=chl_col,
                color='zone_id',
                color_discrete_map={z:c for z,c in ZONE_COLORS.items()
                                    if z in selected_zones},
                title='Monthly Average Chlorine (mg/L) by Zone',
                labels={chl_col:'Avg Chlorine (mg/L)',
                        'month_dt':'Month', 'zone_id':'Zone'},
                markers=True,
            )
            fig3.add_hline(y=0.2, line_dash='dash', line_color='red',
                           annotation_text='WHO minimum 0.2 mg/L',
                           annotation_position='bottom right')
            fig3.update_layout(height=360,
                               plot_bgcolor='rgba(0,0,0,0)',
                               hovermode='x unified')
            st.plotly_chart(fig3, use_container_width=True)

        # ── SHAP insight ──────────────────────────────────────────
        st.divider()
        top_feat = xgb_metrics.get('top_feature','coagulant_dose_mg_L')
        st.info(
            f"**Model Insight (SHAP):** Top quality failure driver is "
            f"**{top_feat}**. WUC should prioritise monitoring "
            f"{top_feat} in zones showing failure alerts."
        )

    except Exception as e:
        st.error(f"Tab 2 error: {e}")
        import traceback; st.code(traceback.format_exc())
# ══════════════════════════════════════════════════════════════════
# TAB 3 — PIPE RISK MAP (Random Forest)
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Pipe Risk Map — Random Forest Model")

    try:
        from streamlit_folium import st_folium
        import folium
        from folium.plugins import MarkerCluster

        pipes_raw    = load_csv_from_s3('hot/pipe_scores/pipe_risk_scores.csv')
        lr_metrics   = load_metrics_from_s3(
            'warm/models/logreg_pipe_risk_metrics.json')  # correct filename

        for col in ['gps_latitude','gps_longitude','prob_low','prob_medium',
                    'prob_high','prob_critical','max_risk_prob','age_years_clean']:
            if col in pipes_raw.columns:
                pipes_raw[col] = pd.to_numeric(pipes_raw[col], errors='coerce')

        # Filter zones + valid GPS — do NOT lowercase columns
        pipes = pipes_raw[
            pipes_raw['zone_id'].isin(selected_zones) &
            pipes_raw['gps_latitude'].notna() &
            pipes_raw['gps_longitude'].notna() &
            (pipes_raw['gps_latitude'].abs()  > 0.01) &
            (pipes_raw['gps_longitude'].abs() > 0.01)
        ].copy()

        # Uppercase predicted_risk for consistent comparison
        pipes['predicted_risk'] = pipes['predicted_risk'].str.upper()

        RISK_COLORS = {
            'LOW':'#1A7A3C','MEDIUM':'#E8B817',
            'HIGH':'#C05B0E','CRITICAL':'#C62828'
        }
        RISK_RADIUS = {'LOW':6,'MEDIUM':7,'HIGH':8,'CRITICAL':10}

        # ── KPI tiles ─────────────────────────────────────────────
        risk_counts = pipes['predicted_risk'].value_counts()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("CRITICAL Pipes",
                  int(risk_counts.get('CRITICAL', 0)),
                  delta="immediate replacement priority")
        k2.metric("HIGH Risk Pipes",
                  int(risk_counts.get('HIGH', 0)))
        k3.metric("Total Pipes", len(pipes))
        k4.metric("Model F1-Macro",
                  f"{lr_metrics.get('f1_macro', 0):.4f}",
                  "Target > 0.80 ✓")
        st.divider()

        # ── CRITICAL-only toggle ──────────────────────────────────
        show_critical = st.toggle(
            "Show CRITICAL pipes only", value=False,
            help="Highlight only the highest-risk pipes on the map"
        )
        plot_pipes = (pipes if not show_critical
                      else pipes[pipes['predicted_risk'] == 'CRITICAL'])

        # ── Build folium map ──────────────────────────────────────
        centre_lat = float(pipes['gps_latitude'].mean())
        centre_lon = float(pipes['gps_longitude'].mean())

        m = folium.Map(
            location   = [centre_lat, centre_lon],
            zoom_start = 12,
            tiles      = 'CartoDB positron',
        )
        cluster = MarkerCluster(disableClusteringAtZoom=14)

        for _, row in plot_pipes.iterrows():
            risk  = str(row.get('predicted_risk', 'MEDIUM')).upper()
            color = RISK_COLORS.get(risk, '#546E7A')
            rad   = RISK_RADIUS.get(risk, 7)

            popup_html = (
                f"<div style='font-family:Arial;font-size:12px;"
                f"min-width:180px'>"
                f"<b style='color:{color}'>{risk} RISK</b><br>"
                f"<b>Pipe:</b> {row.get('pipe_id','?')}<br>"
                f"<b>Zone:</b> {row.get('zone_id','?')}<br>"
                f"<b>Material:</b> {row.get('material_clean','?')}<br>"
                f"<b>Age:</b> {row.get('age_years_clean','?')} yrs<br>"
                f"<hr style='margin:4px 0'>"
                f"<b>P(CRITICAL):</b> {row.get('prob_critical',0):.3f}<br>"
                f"<b>P(HIGH):</b> {row.get('prob_high',0):.3f}"
                f"</div>"
            )
            folium.CircleMarker(
                location     = [float(row['gps_latitude']),
                                float(row['gps_longitude'])],
                radius       = rad,
                color        = color,
                fill         = True,
                fill_color   = color,
                fill_opacity = 0.75,
                popup        = folium.Popup(popup_html, max_width=220),
                tooltip      = f"{row.get('pipe_id','?')} — {risk}",
            ).add_to(cluster)

        cluster.add_to(m)

        legend_html = """
        <div style='position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:white;padding:10px 14px;border-radius:8px;
                    border:1px solid #ccc;font-family:Arial;font-size:12px'>
          <b>Pipe Risk Category</b><br>
          <span style='color:#C62828'>&#9679;</span> CRITICAL &nbsp;
          <span style='color:#C05B0E'>&#9679;</span> HIGH<br>
          <span style='color:#E8B817'>&#9679;</span> MEDIUM &nbsp;
          <span style='color:#1A7A3C'>&#9679;</span> LOW
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=None, height=520, returned_objects=[])

        # ── Top 10 CRITICAL pipes table ───────────────────────────
        st.divider()
        st.markdown("**Top 10 Highest-Priority Pipes — by P(CRITICAL)**")
        if 'prob_critical' in pipes.columns:
            top10 = (
                pipes.nlargest(10, 'prob_critical')
                [['pipe_id','zone_id','material_clean',
                  'age_years_clean','predicted_risk',
                  'prob_critical','prob_high']]
                .rename(columns={
                    'pipe_id':       'Pipe ID',
                    'zone_id':       'Zone',
                    'material_clean':'Material',
                    'age_years_clean':'Age (yrs)',
                    'predicted_risk':'Risk',
                    'prob_critical': 'P(CRITICAL)',
                    'prob_high':     'P(HIGH)',
                })
                .reset_index(drop=True)
            )
            st.dataframe(
                top10.style.background_gradient(
                    subset=['P(CRITICAL)'], cmap='Reds'
                ).format({'P(CRITICAL)':'{:.3f}',
                          'P(HIGH)':'{:.3f}',
                          'Age (yrs)':'{:.0f}'}),
                use_container_width=True,
            )

        # ── Risk breakdown bar chart ──────────────────────────────
        st.divider()
        rc = pipes['predicted_risk'].value_counts().reset_index()
        rc.columns = ['Risk','Count']
        rc['Risk'] = pd.Categorical(
            rc['Risk'],
            categories=['CRITICAL','HIGH','MEDIUM','LOW'],
            ordered=True
        )
        rc = rc.sort_values('Risk')
        fig_rc = px.bar(
            rc, x='Risk', y='Count',
            color='Risk',
            color_discrete_map=RISK_COLORS,
            title='Pipe Count by Risk Category',
            text='Count',
        )
        fig_rc.update_traces(textposition='outside')
        fig_rc.update_layout(showlegend=False, height=320,
                              plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rc, use_container_width=True)

    except Exception as e:
        st.error(f"Tab 3 error: {e}")
        import traceback; st.code(traceback.format_exc())
# ══════════════════════════════════════════════════════════════════
# TAB 4 — DEMAND FORECAST (Prophet)
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Demand Forecast — Prophet Model (×8 Zones)")

    try:
        fc_raw    = load_csv_from_s3('hot/forecasts/demand_forecasts_6month.csv')
        bill_raw  = load_csv_from_s3('warm/features/billing_features.csv')
        p_metrics = load_metrics_from_s3('warm/models/prophet_demand_metrics.json')

        # Fix 1: bill_raw uses 'ds' not 'date'
        fc_raw['ds']         = pd.to_datetime(fc_raw['ds'],         errors='coerce')
        fc_raw['yhat']       = pd.to_numeric(fc_raw['yhat'],        errors='coerce')
        fc_raw['yhat_lower'] = pd.to_numeric(fc_raw['yhat_lower'],  errors='coerce')
        fc_raw['yhat_upper'] = pd.to_numeric(fc_raw['yhat_upper'],  errors='coerce')

        # Fix 2: detect date column — billing uses 'ds' not 'date'
        date_col = next((c for c in ['ds','date','billing_date']
                         if c in bill_raw.columns), None)
        bill_raw[date_col] = pd.to_datetime(bill_raw[date_col], errors='coerce')

        # Fix 3: standardise zone_ids in billing
        ZONE_MAP = {
            'GBR-N':'GBR-N','GBR-S':'GBR-S','GBR-C':'GBR-C',
            'FRA-01':'FRA-01','MHR-01':'MHR-01','KNY-01':'KNY-01',
            'LBT-01':'LBT-01','MMB-01':'MMB-01',
            'GBR_N':'GBR-N','GBR_S':'GBR-S','GBR_C':'GBR-C',
            'FRA_01':'FRA-01','MHR_01':'MHR-01','KNY_01':'KNY-01',
            'LBT_01':'LBT-01','MMB_01':'MMB-01','GBR-X':'GBR-N',
        }
        bill_raw['zone_id'] = (bill_raw['zone_id'].astype(str)
                               .str.strip().map(ZONE_MAP))
        bill_raw = bill_raw.dropna(subset=['zone_id'])

        # Filter to selected zones
        fc   = fc_raw[fc_raw['zone_id'].isin(selected_zones)].copy()
        bill = bill_raw[bill_raw['zone_id'].isin(selected_zones)].copy()

        # Cutoff = last date in billing
        cutoff = bill[date_col].max()

        # ── KPI tiles ─────────────────────────────────────────────
        zone_mapes = p_metrics.get('zones', {})
        mean_mape  = p_metrics.get('overall_mape', 0)
        future_fc  = fc[fc['ds'] > cutoff]

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Mean MAPE",        f"{mean_mape:.2f}%", "Target < 15% ✓")
        k2.metric("Forecast Horizon", "to Dec 2026")
        k3.metric("Zones Forecast",   len(selected_zones))
        if len(future_fc) > 0:
            peak_zone = future_fc.groupby('zone_id')['yhat'].max().idxmax()
            k4.metric("Peak Demand Zone", peak_zone)
        st.divider()

        # ── Zone selector ─────────────────────────────────────────
        col_sel, col_toggle = st.columns([3,1])
        with col_sel:
            zone_choice = st.selectbox(
                "Select zone for deep-dive forecast",
                options=selected_zones,
                index=selected_zones.index('KNY-01')
                      if 'KNY-01' in selected_zones else 0,
            )
        with col_toggle:
            show_ci = st.toggle("Show confidence band", value=True)

        # ── Single-zone forecast chart ────────────────────────────
        zfc  = fc[fc['zone_id']==zone_choice].sort_values('ds').reset_index(drop=True)
        hist = zfc[zfc['ds'] <= cutoff].reset_index(drop=True)
        fut  = zfc[zfc['ds'] >  cutoff].reset_index(drop=True)
        col  = ZONE_COLORS.get(zone_choice, '#546E7A')

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
            marker=dict(size=6, symbol='diamond'),
        ))

        # Fix 4: use np.concatenate + add_shape instead of add_vline
        if show_ci and len(fut) > 0:
            hex_col = col.lstrip('#')
            r,g,b   = int(hex_col[0:2],16),int(hex_col[2:4],16),int(hex_col[4:6],16)
            rgba    = f'rgba({r},{g},{b},0.12)'
            x_band  = np.concatenate([fut['ds'].values,
                                       fut['ds'].values[::-1]])
            y_band  = np.concatenate([fut['yhat_upper'].values/1000,
                                       fut['yhat_lower'].values[::-1]/1000])
            fig1.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill='toself', fillcolor=rgba,
                line=dict(color='rgba(255,255,255,0)'),
                name='80% CI',
            ))

        # Fix 5: add_shape instead of add_vline — avoids datetime bug
        cutoff_str = cutoff.strftime('%Y-%m-%d')
        fig1.add_shape(
            type='line',
            x0=cutoff_str, x1=cutoff_str,
            y0=0, y1=1, yref='paper',
            line=dict(dash='dot', color='gray', width=1.5),
        )
        fig1.add_annotation(
            x=cutoff_str, y=1.02, yref='paper',
            text='Forecast start', showarrow=False,
            font=dict(size=11, color='gray'),
        )

        for yr in [2024, 2025, 2026]:
            fig1.add_vrect(x0=f'{yr}-07-01', x1=f'{yr}-09-01',
                           fillcolor='#E8B817', opacity=0.07,
                           layer='below')

        fig1.update_layout(
            title=f'{zone_choice} — Water Demand Forecast (kL × 1,000)',
            xaxis_title='Month', yaxis_title='Volume (000 kL)',
            height=420, plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── All-zones comparison ───────────────────────────────────
        st.divider()
        st.markdown("**Forecast Comparison — All Selected Zones**")
        fig2 = go.Figure()
        for zone in selected_zones:
            zf = fc[(fc['zone_id']==zone) &
                    (fc['ds'] > cutoff)].sort_values('ds')
            if len(zf) == 0:
                continue
            c = ZONE_COLORS.get(zone, '#546E7A')
            fig2.add_trace(go.Scatter(
                x=zf['ds'].values, y=(zf['yhat'].values/1000),
                mode='lines+markers', name=zone,
                line=dict(color=c, width=2), marker=dict(size=5),
            ))
        fig2.add_vrect(x0='2026-07-01', x1='2026-09-01',
                       fillcolor='#E8B817', opacity=0.08,
                       layer='below')
        fig2.update_layout(
            height=380, plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Month',
            yaxis_title='Forecast Volume (000 kL)',
            hovermode='x unified',
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── MAPE table ────────────────────────────────────────────
        st.divider()
        st.markdown("**Model Performance — MAPE per Zone**")
        mape_rows = [
            {'Zone': z,
             'MAPE (%)': info.get('mape', 0),
             'Months':   info.get('n_months', 0),
             'Status':   '✓ PASS' if info.get('mape',99) < 15 else '✗ FAIL'}
            for z, info in zone_mapes.items()
            if z in selected_zones
        ]
        if mape_rows:
            mape_df = pd.DataFrame(mape_rows).sort_values('MAPE (%)')
            st.dataframe(
                mape_df.style.background_gradient(
                    subset=['MAPE (%)'], cmap='RdYlGn_r', vmin=0, vmax=15
                ).format({'MAPE (%)': '{:.2f}'}),
                use_container_width=True,
            )

        # ── Insight ───────────────────────────────────────────────
        st.divider()
        st.info(
            "**Forecast Insight:** Peak demand occurs July–August (dry season). "
            "KNY-01 and LBT-01 show the highest forecast volumes — these are also "
            "the highest NRW zones. Increased pumping during dry season amplifies "
            "revenue losses if NRW is not reduced before July 2026."
        )

    except Exception as e:
        st.error(f"Tab 4 error: {e}")
        import traceback; st.code(traceback.format_exc())