"""
src/feature_utils.py
--------------------
Feature preprocessing, alignment, and validation utilities.

Keeps all feature engineering logic out of app.py and ensures
that the features passed to models exactly match those used during training.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Feature sets (must match training artefacts) ─────────────────────────────

LEAK_FEATURES = [
    "flow_rate", "pressure_variance", "hour_of_day",
    "day_of_week", "zone_id_encoded", "anomaly_score",
]

QUALITY_FEATURES = [
    "turbidity", "ph_level", "chlorine_residual",
    "temperature_c", "conductivity", "zone_id_encoded",
]

PIPE_RISK_FEATURES = [
    "pipe_age_years", "material_encoded", "diameter_mm",
    "pressure_zone", "repair_count", "soil_corrosivity_index",
]


def align_features(df: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    """
    Ensure df contains exactly the required feature columns, in order.

    Adds zero-filled columns for any feature missing from df (with a warning),
    and drops any extra columns not in the required list.

    Parameters
    ----------
    df : pd.DataFrame
    required_features : list[str]

    Returns
    -------
    pd.DataFrame  — subset/reordered to required_features
    """
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        logger.warning("Features missing from input, filling with 0: %s", missing)
        for col in missing:
            df = df.copy()
            df[col] = 0
    return df[required_features]


def encode_zone_id(df: pd.DataFrame, col: str = "zone_id") -> pd.DataFrame:
    """
    Encode zone_id strings to integers.

    Expects zone IDs in the format 'ZONE_001', 'ZONE_002', etc.
    Strips the prefix and converts to int. Rows with unrecognised
    format are set to -1 and a warning is logged.

    Parameters
    ----------
    df : pd.DataFrame
    col : str  column name containing raw zone IDs

    Returns
    -------
    pd.DataFrame  with an added 'zone_id_encoded' column
    """
    df = df.copy()
    if col not in df.columns:
        logger.warning("Column '%s' not found — zone_id_encoded set to -1", col)
        df["zone_id_encoded"] = -1
        return df

    def _parse(val):
        try:
            return int(str(val).replace("ZONE_", ""))
        except (ValueError, AttributeError):
            return -1

    df["zone_id_encoded"] = df[col].apply(_parse)
    n_invalid = (df["zone_id_encoded"] == -1).sum()
    if n_invalid:
        logger.warning("%d rows had unrecognised zone_id format", n_invalid)
    return df


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the name of the first date-like column in df, or None.

    Checks column names containing 'date', 'time', 'timestamp', 'ds'
    (Prophet convention). Returns None if no candidate is found,
    so callers can handle the missing-column case explicitly rather
    than assuming one always exists.
    """
    candidates = [c for c in df.columns
                  if any(kw in c.lower() for kw in ("date", "time", "timestamp", "ds"))]
    if not candidates:
        logger.warning("No date column found in DataFrame. Available: %s", list(df.columns))
        return None
    return candidates[0]


def safe_map_centre(df: pd.DataFrame,
                    lat_col: str = "latitude",
                    lon_col: str = "longitude",
                    default_lat: float = -24.654,
                    default_lon: float = 25.908) -> tuple[float, float]:
    """
    Return (lat, lon) centre of the filtered pipe dataset.

    Falls back to Gaborone coordinates if df is empty or columns are missing,
    so the map always initialises correctly regardless of filter state.

    Parameters
    ----------
    df : pd.DataFrame
    lat_col, lon_col : str
    default_lat, default_lon : float  — fallback (Gaborone CBD)

    Returns
    -------
    (lat, lon) tuple of floats
    """
    if df.empty:
        logger.info("Pipe data is empty — using default map centre (Gaborone).")
        return default_lat, default_lon

    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning(
            "Columns '%s' / '%s' not found — using default map centre.", lat_col, lon_col
        )
        return default_lat, default_lon

    valid = df[[lat_col, lon_col]].dropna()
    if valid.empty:
        return default_lat, default_lon

    return float(valid[lat_col].mean()), float(valid[lon_col].mean())


def validate_zone_selection(selected_zones: list) -> bool:
    """
    Return True if at least one zone is selected.

    Use this before any downstream filtering to avoid passing empty
    selections into model prediction or chart generation.
    """
    return bool(selected_zones)
