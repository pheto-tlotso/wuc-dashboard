"""
src/chart_utils.py
------------------
Reusable Plotly chart builders for the WUC NRW dashboard.

All chart functions accept a DataFrame and return a Plotly Figure,
keeping visualisation logic out of app.py.

Narrative insight messages are generated from actual data values
rather than hard-coded strings.
"""

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Theme ─────────────────────────────────────────────────────────────────────
_TEMPLATE = "plotly_dark"
_PALETTE = {
    "low":    "#2ECC71",
    "medium": "#F39C12",
    "high":   "#E74C3C",
    "accent": "#3498DB",
    "neutral": "#95A5A6",
}


def leak_risk_bar(df: pd.DataFrame,
                  zone_col: str = "zone_id",
                  risk_col: str = "leak_probability") -> go.Figure:
    """
    Bar chart of predicted leak probability by zone.
    Bars are colour-coded by risk band derived from actual values.
    """
    if df.empty or risk_col not in df.columns:
        return _empty_figure("No leak risk data available for the selected zones.")

    zone_agg = (
        df.groupby(zone_col)[risk_col]
        .mean()
        .reset_index()
        .sort_values(risk_col, ascending=False)
    )

    def _band(p):
        if p >= 0.7:
            return "High"
        if p >= 0.4:
            return "Medium"
        return "Low"

    zone_agg["risk_band"] = zone_agg[risk_col].apply(_band)
    color_map = {"High": _PALETTE["high"], "Medium": _PALETTE["medium"], "Low": _PALETTE["low"]}

    fig = px.bar(
        zone_agg,
        x=zone_col,
        y=risk_col,
        color="risk_band",
        color_discrete_map=color_map,
        labels={risk_col: "Mean Leak Probability", zone_col: "Zone"},
        template=_TEMPLATE,
    )
    fig.update_layout(
        showlegend=True,
        legend_title_text="Risk Band",
        margin=dict(t=24, b=0, l=0, r=0),
    )
    return fig


def pipe_risk_map(df: pd.DataFrame,
                  lat_col: str = "latitude",
                  lon_col: str = "longitude",
                  risk_col: str = "risk_score",
                  centre: tuple[float, float] = (-24.654, 25.908)) -> go.Figure:
    """
    Scatter mapbox of pipe risk scores.
    Requires a non-empty DataFrame — callers should guard with safe_map_centre.
    """
    required = {lat_col, lon_col, risk_col}
    if df.empty or not required.issubset(df.columns):
        return _empty_figure("No pipe risk data available for the selected filters.")

    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=risk_col,
        color_continuous_scale="RdYlGn_r",
        size=risk_col,
        size_max=12,
        zoom=10,
        center={"lat": centre[0], "lon": centre[1]},
        mapbox_style="carto-darkmatter",
        labels={risk_col: "Risk Score"},
        template=_TEMPLATE,
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    return fig


def demand_forecast_line(df: pd.DataFrame,
                         date_col: str,
                         yhat_col: str = "yhat",
                         yhat_lower_col: str = "yhat_lower",
                         yhat_upper_col: str = "yhat_upper",
                         zone_label: str = "") -> go.Figure:
    """
    Line chart with confidence band for Prophet demand forecast output.

    Parameters
    ----------
    df : pd.DataFrame  — forecast output for a single zone
    date_col : str     — column containing forecast dates
    zone_label : str   — used in the chart title (optional)
    """
    if df.empty:
        return _empty_figure("No forecast data available for the selected zone.")

    missing = [c for c in [date_col, yhat_col] if c not in df.columns]
    if missing:
        return _empty_figure(f"Forecast data is missing columns: {missing}")

    fig = go.Figure()

    # Confidence band
    if yhat_lower_col in df.columns and yhat_upper_col in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df[date_col], df[date_col].iloc[::-1]]),
            y=pd.concat([df[yhat_upper_col], df[yhat_lower_col].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[yhat_col],
        mode="lines",
        line=dict(color=_PALETTE["accent"], width=2),
        name="Forecast",
    ))

    title = f"Demand Forecast — {zone_label}" if zone_label else "Demand Forecast"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Predicted Demand (m³)",
        template=_TEMPLATE,
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def derive_forecast_insight(df: pd.DataFrame,
                             date_col: str,
                             yhat_col: str = "yhat") -> str:
    """
    Generate a short data-driven insight string from forecast output.

    Returns a plain-English sentence describing what the forecast
    actually shows — replaces any hard-coded commentary.
    """
    if df.empty or date_col not in df.columns or yhat_col not in df.columns:
        return "No forecast data available to generate an insight."

    peak_row = df.loc[df[yhat_col].idxmax()]
    trough_row = df.loc[df[yhat_col].idxmin()]
    mean_demand = df[yhat_col].mean()

    try:
        peak_date = pd.to_datetime(peak_row[date_col]).strftime("%d %b %Y")
        trough_date = pd.to_datetime(trough_row[date_col]).strftime("%d %b %Y")
    except Exception:
        peak_date = str(peak_row[date_col])
        trough_date = str(trough_row[date_col])

    return (
        f"Forecast peak demand of {peak_row[yhat_col]:,.0f} m³ is expected on {peak_date}. "
        f"Lowest demand of {trough_row[yhat_col]:,.0f} m³ is forecast for {trough_date}. "
        f"Mean demand over the forecast horizon is {mean_demand:,.0f} m³."
    )


def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return a blank figure with a centred message — used when data is missing."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#95A5A6"),
    )
    fig.update_layout(
        template=_TEMPLATE,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(t=0, b=0, l=0, r=0),
    )
    return fig
