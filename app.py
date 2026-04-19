"""
app.py
------
WUC Non-Revenue Water Intelligence Dashboard
Streamlit entry point — UI layout only.

All data loading  →  src/data_access.py
All model loading →  src/model_loader.py
All feature prep  →  src/feature_utils.py
All chart logic   →  src/chart_utils.py
"""

import logging

import streamlit as st
import pandas as pd

from src.data_access import (
    load_sensor_data,
    load_pipe_data,
    load_forecast_data,
    load_metrics,
)
from src.model_loader import get_leak_model, get_quality_model, get_pipe_risk_model
from src.feature_utils import (
    align_features,
    encode_zone_id,
    find_date_column,
    safe_map_centre,
    validate_zone_selection,
    LEAK_FEATURES,
    QUALITY_FEATURES,
    PIPE_RISK_FEATURES,
)
from src.chart_utils import (
    leak_risk_bar,
    pipe_risk_map,
    demand_forecast_line,
    derive_forecast_insight,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WUC NRW Intelligence",
    page_icon="💧",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💧 WUC Non-Revenue Water Intelligence")
st.caption("Prototype — Rooteddeck × Water Utilities Corporation Botswana")
st.divider()


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading sensor data…")
def _sensor_data() -> pd.DataFrame:
    return load_sensor_data()


@st.cache_data(show_spinner="Loading pipe inventory…")
def _pipe_data() -> pd.DataFrame:
    return load_pipe_data()


@st.cache_data(show_spinner="Loading demand forecast…")
def _forecast_data() -> pd.DataFrame:
    return load_forecast_data()


sensor_df = _sensor_data()
pipe_df = _pipe_data()
forecast_df = _forecast_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    # Zone selector — derived from available data, not hard-coded
    all_zones: list = []
    if not sensor_df.empty and "zone_id" in sensor_df.columns:
        all_zones = sorted(sensor_df["zone_id"].dropna().unique().tolist())

    selected_zones = st.multiselect(
        "Zones",
        options=all_zones,
        default=all_zones,
        help="Select one or more zones. Charts update based on your selection.",
    )

    st.caption("ℹ️ Select at least one zone to view analysis.")
    st.divider()
    st.caption("Prototype v1.0 — Rooteddeck")


# ── Zone validation guard ─────────────────────────────────────────────────────
if not validate_zone_selection(selected_zones):
    st.warning("Please select at least one zone from the sidebar to display results.")
    st.stop()

# Apply zone filter
filtered_sensor = sensor_df[sensor_df["zone_id"].isin(selected_zones)] if not sensor_df.empty else sensor_df
filtered_pipe = pipe_df[pipe_df["zone_id"].isin(selected_zones)] if (not pipe_df.empty and "zone_id" in pipe_df.columns) else pipe_df


# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_leak, tab_quality, tab_pipe, tab_demand = st.tabs([
    "🔴 Leak Detection",
    "🔵 Water Quality",
    "🟡 Pipe Risk",
    "📈 Demand Forecast",
])


# ── TAB 1: Leak Detection ─────────────────────────────────────────────────────
with tab_leak:
    st.subheader("Leak Detection")

    leak_model = get_leak_model()

    if leak_model is None:
        st.error("Leak detection model could not be loaded. Check logs for details.")
    elif filtered_sensor.empty:
        st.info("No sensor data available for the selected zones.")
    else:
        try:
            encoded = encode_zone_id(filtered_sensor)
            X = align_features(encoded, LEAK_FEATURES)
            filtered_sensor = filtered_sensor.copy()
            filtered_sensor["leak_probability"] = leak_model.predict_proba(X)[:, 1]

            # KPIs derived from actual predictions
            n_high = (filtered_sensor["leak_probability"] >= 0.7).sum()
            n_medium = ((filtered_sensor["leak_probability"] >= 0.4) & (filtered_sensor["leak_probability"] < 0.7)).sum()
            mean_prob = filtered_sensor["leak_probability"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("High-risk readings", f"{n_high:,}")
            col2.metric("Medium-risk readings", f"{n_medium:,}")
            col3.metric("Mean leak probability", f"{mean_prob:.1%}")

            leak_metrics = load_metrics("leak_detection")
            if leak_metrics:
                with st.expander("Model performance metrics"):
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("F1 Score", f"{leak_metrics.get('f1', 'N/A')}")
                    mc2.metric("AUC-ROC", f"{leak_metrics.get('auc', 'N/A')}")
                    mc3.metric("Precision", f"{leak_metrics.get('precision', 'N/A')}")

            st.plotly_chart(
                leak_risk_bar(filtered_sensor),
                use_container_width=True,
            )
        except Exception as exc:
            logger.error("Leak detection error: %s", exc)
            st.error("An error occurred during leak prediction. See logs.")


# ── TAB 2: Water Quality ──────────────────────────────────────────────────────
with tab_quality:
    st.subheader("Water Quality Monitoring")

    quality_model = get_quality_model()

    if quality_model is None:
        st.error("Water quality model could not be loaded.")
    elif filtered_sensor.empty:
        st.info("No sensor data available for the selected zones.")
    else:
        try:
            encoded = encode_zone_id(filtered_sensor)
            X = align_features(encoded, QUALITY_FEATURES)
            filtered_sensor = filtered_sensor.copy()
            filtered_sensor["quality_flag"] = quality_model.predict(X)

            flag_counts = filtered_sensor["quality_flag"].value_counts()
            st.bar_chart(flag_counts)

            quality_metrics = load_metrics("water_quality")
            if quality_metrics:
                with st.expander("Model performance metrics"):
                    st.json(quality_metrics)
        except Exception as exc:
            logger.error("Water quality error: %s", exc)
            st.error("An error occurred during quality prediction. See logs.")


# ── TAB 3: Pipe Risk ──────────────────────────────────────────────────────────
with tab_pipe:
    st.subheader("Pipe Risk Assessment")
    st.caption("Model: XGBoost gradient-boosted classifier (binary risk stratification)")

    pipe_model = get_pipe_risk_model()

    if pipe_model is None:
        st.error("Pipe risk model could not be loaded.")
    elif filtered_pipe.empty:
        st.info("No pipe data available for the selected zones.")
    else:
        try:
            X_pipe = align_features(filtered_pipe, PIPE_RISK_FEATURES)
            filtered_pipe = filtered_pipe.copy()
            filtered_pipe["risk_score"] = pipe_model.predict_proba(X_pipe)[:, 1]

            # KPIs
            n_high_risk = (filtered_pipe["risk_score"] >= 0.7).sum()
            mean_risk = filtered_pipe["risk_score"].mean()

            col1, col2 = st.columns(2)
            col1.metric("High-risk pipe segments", f"{n_high_risk:,}")
            col2.metric("Mean risk score", f"{mean_risk:.2f}")

            # Map — safe_map_centre handles empty filtered results
            centre = safe_map_centre(filtered_pipe)
            st.plotly_chart(
                pipe_risk_map(filtered_pipe, centre=centre),
                use_container_width=True,
            )

            pipe_metrics = load_metrics("pipe_risk")
            if pipe_metrics:
                with st.expander("Model performance metrics"):
                    st.json(pipe_metrics)

        except Exception as exc:
            logger.error("Pipe risk error: %s", exc)
            st.error("An error occurred during pipe risk scoring. See logs.")


# ── TAB 4: Demand Forecast ────────────────────────────────────────────────────
with tab_demand:
    st.subheader("Demand Forecasting")
    st.caption("Based on Prophet-generated baseline projections. Real-time re-fitting not available in this prototype.")

    if forecast_df.empty:
        st.info("Forecast data could not be loaded.")
    else:
        # Resolve date column — handle missing gracefully
        date_col = find_date_column(forecast_df)
        if date_col is None:
            st.error(
                "Forecast data does not contain a recognisable date column. "
                f"Available columns: {list(forecast_df.columns)}"
            )
        else:
            # Zone selector for forecast (single zone)
            forecast_zones: list = []
            if "zone_id" in forecast_df.columns:
                forecast_zones = sorted(forecast_df["zone_id"].dropna().unique().tolist())
                forecast_zones = [z for z in forecast_zones if z in selected_zones]

            if not forecast_zones:
                st.info("No forecast data available for the currently selected zones.")
            else:
                selected_forecast_zone = st.selectbox(
                    "Select zone for forecast view",
                    options=forecast_zones,
                )
                zone_forecast = forecast_df[forecast_df["zone_id"] == selected_forecast_zone]

                # Data-driven insight — not hard-coded
                insight = derive_forecast_insight(zone_forecast, date_col)
                st.info(insight)

                st.plotly_chart(
                    demand_forecast_line(zone_forecast, date_col, zone_label=selected_forecast_zone),
                    use_container_width=True,
                )

                forecast_metrics = load_metrics("demand_forecast")
                if forecast_metrics:
                    with st.expander("Forecast accuracy metrics"):
                        st.json(forecast_metrics)
