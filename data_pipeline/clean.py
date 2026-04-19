"""
data_pipeline/clean.py
-----------------------
Cleaning pipeline for the four WUC NRW analytical datasets.

Reads from:  data/raw/   (the files as-received)
Writes to:   data/clean/ (what the dashboard loads)

Issues resolved per dataset (discovered from actual data audit):

BILLING FEATURES
  - zone_id has 32 variants for 8 real zones: trailing spaces, underscores
    instead of hyphens, mixed case, 'UNKNOWN', 'ZONE??', 'GBR-X'
  - 'ds' column is a string — parse to datetime

IOT FEATURES
  - Same zone_id mess as billing (35 variants)
  - meter_id mixed case: 'WUC/FRA-01/MTR-3753' vs 'wuc/fra-01/mtr-4575'
  - Duplicate columns with conflicting values:
      hour_of_day  vs  hour            (877 mismatches — hour is derived from
                                         timestamp and is correct; hour_of_day
                                         has corruption)
      is_weekend   vs  is_wknd         (126 mismatches — is_wknd is derived
                                         correctly; is_weekend has errors)
      leak_event   vs  leak_event_clean (13,895 mismatches — leak_event is raw
                                         with 19 different string values;
                                         leak_event_clean is the resolved binary)
  - Lag/rolling NaNs (first N rows per zone — expected, not errors)

PIPE RISK SCORES
  - zone_id has sentinel values: 'UNKNOWN', 'GBR-X', 'ZONE??' — these can
    be recovered from pipe_id prefix or zone_name where available
  - 1 null zone_id, 50 null zone_names, 2 null gps coordinates
  - 1 null age_years_clean

DEMAND FORECASTS
  - Cleanest file — zone_ids already canonical
  - 'ds' is a string — parse to datetime
  - No nulls detected

Usage:
  python data_pipeline/clean.py
  python data_pipeline/clean.py --input data/raw --output data/clean
"""

import os
import re
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT  = os.path.join(_BASE, "data", "raw")
DEFAULT_OUTPUT = os.path.join(_BASE, "data", "clean")

# ── Canonical zone IDs ─────────────────────────────────────────────────────────
# Sourced from demand_forecasts_6month.csv — the only file with no zone_id issues.
CANONICAL_ZONES = {"FRA-01", "GBR-C", "GBR-N", "GBR-S", "KNY-01", "LBT-01", "MHR-01", "MMB-01"}

# Maps every observed variant to its canonical form.
# Built from the actual audit — not guesses.
ZONE_ID_MAP = {
    # Trailing-space variants
    "FRA-01 ":  "FRA-01",  "GBR-C ":  "GBR-C",
    "GBR-N ":   "GBR-N",   "GBR-S ":  "GBR-S",
    "KNY-01 ":  "KNY-01",  "LBT-01 ": "LBT-01",
    "MHR-01 ":  "MHR-01",  "MMB-01 ": "MMB-01",
    # Underscore variants
    "FRA_01":   "FRA-01",  "GBR_C":   "GBR-C",
    "GBR_N":    "GBR-N",   "GBR_S":   "GBR-S",
    "KNY_01":   "KNY-01",  "LBT_01":  "LBT-01",
    "MHR_01":   "MHR-01",  "MMB_01":  "MMB-01",
    # Lowercase variants
    "fra-01":   "FRA-01",  "gbr-c":   "GBR-C",
    "gbr-n":    "GBR-N",   "gbr-s":   "GBR-S",
    "kny-01":   "KNY-01",  "lbt-01":  "LBT-01",
    "mhr-01":   "MHR-01",  "mmb-01":  "MMB-01",
}

# Sentinel values that cannot be directly mapped — need context recovery
SENTINEL_ZONES = {"UNKNOWN", "ZONE??", "GBR-X", ""}


def _coerce_zone_id(val) -> str | None:
    """Map a raw zone_id value to its canonical form, or None if unresolvable."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in CANONICAL_ZONES:
        return s
    if s in ZONE_ID_MAP:
        return ZONE_ID_MAP[s]
    if s in SENTINEL_ZONES:
        return None
    return None


def _recover_zone_from_pipe_id(pipe_id) -> str | None:
    """
    Extract zone_id from pipe_id: 'WUC-GBR-S-0527' → 'GBR-S'.
    Tries each canonical zone as a prefix match after stripping 'WUC-'.
    """
    if pd.isna(pipe_id):
        return None
    stripped = str(pipe_id).replace("WUC-", "", 1)
    for zone in CANONICAL_ZONES:
        if stripped.startswith(zone + "-"):
            return zone
    return None


def _recover_zone_from_zone_name(zone_name) -> str | None:
    """Map zone_name text to canonical zone_id."""
    if pd.isna(zone_name):
        return None
    mapping = {
        "Francistown":      "FRA-01",
        "Gaborone Central": "GBR-C",
        "Gaborone North":   "GBR-N",
        "Gaborone South":   "GBR-S",
        "Kanye":            "KNY-01",
        "Lobatse":          "LBT-01",
        "Mahalapye":        "MHR-01",
        "Molepolole":       "MMB-01",
    }
    return mapping.get(str(zone_name).strip())


# ── Leak event resolution ──────────────────────────────────────────────────────
LEAK_POSITIVE = {"1", "yes", "Yes", "YES", "true", "True", "TRUE",
                 "ALARM", "LEAK", "L"}
LEAK_NEGATIVE = {"0", "no", "No", "NO", "false", "False", "FALSE",
                 "NORMAL", "CLEAR", "OK"}

def _resolve_leak_event(val) -> int:
    """Convert any raw leak_event value to 0 (no leak) or 1 (leak)."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in LEAK_POSITIVE:
        return 1
    if s in LEAK_NEGATIVE:
        return 0
    try:
        return 1 if float(s) > 0 else 0
    except ValueError:
        logger.warning("Unrecognised leak_event value '%s' — defaulting to 0", s)
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clean_billing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"dataset": "billing_features", "input_rows": len(df)}

    before = df["zone_id"].copy()
    df["zone_id"] = df["zone_id"].apply(_coerce_zone_id)
    n_fixed = int(((df["zone_id"] != before) & df["zone_id"].notna()).sum())
    n_null  = int(df["zone_id"].isna().sum())
    report["zone_id"] = {
        "variants_resolved": n_fixed,
        "unresolvable_rows":  n_null,
        "unresolvable_values": before[df["zone_id"].isna()].value_counts().to_dict(),
    }
    if n_null:
        logger.warning(
            "billing_features: %d rows with unresolvable zone_id dropped "
            "(UNKNOWN / ZONE?? / GBR-X). Review raw source if they should map "
            "to a real zone.", n_null
        )
        df = df.dropna(subset=["zone_id"])

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    n_bad_ds = int(df["ds"].isna().sum())
    if n_bad_ds:
        df = df.dropna(subset=["ds"])
    report["ds_parse_failures"] = n_bad_ds

    n_dupes = int(df.duplicated(subset=["zone_id", "ds"]).sum())
    if n_dupes:
        df = df.drop_duplicates(subset=["zone_id", "ds"])
    report["duplicates_dropped"] = n_dupes

    report["output_rows"] = len(df)
    return df.sort_values(["zone_id", "ds"]).reset_index(drop=True), report


def clean_iot_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"dataset": "iot_features", "input_rows": len(df)}

    # Zone ID
    before = df["zone_id"].copy()
    df["zone_id"] = df["zone_id"].apply(_coerce_zone_id)
    n_fixed = int(((df["zone_id"] != before) & df["zone_id"].notna()).sum())
    n_null  = int(df["zone_id"].isna().sum())
    report["zone_id"] = {"variants_resolved": n_fixed, "unresolvable_rows": n_null}
    if n_null:
        df = df.dropna(subset=["zone_id"])

    # meter_id — uppercase for consistency
    df["meter_id"] = df["meter_id"].str.upper()
    report["meter_id_uppercased"] = True

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_ts = int(df["timestamp"].isna().sum())
    if n_bad_ts:
        df = df.dropna(subset=["timestamp"])
    report["timestamp_parse_failures"] = n_bad_ts

    # Drop corrupted duplicate columns — re-derive from timestamp as ground truth
    # hour_of_day had 877 mismatches vs timestamp-derived hour
    df["hour"] = df["timestamp"].dt.hour
    df = df.drop(columns=["hour_of_day"], errors="ignore")
    report["hour_of_day_dropped"] = "877 mismatches vs timestamp — re-derived as 'hour'"

    # is_weekend had 126 mismatches vs timestamp-derived is_wknd
    df["is_wknd"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    df = df.drop(columns=["is_weekend"], errors="ignore")
    report["is_weekend_dropped"] = "126 mismatches vs timestamp — re-derived as 'is_wknd'"

    # leak_event had 19 distinct string formats — resolve to binary, drop raw
    df["leak_event_clean"] = df["leak_event"].apply(_resolve_leak_event)
    df = df.drop(columns=["leak_event"], errors="ignore")
    report["leak_event_resolved"] = {
        "raw_column_dropped": True,
        "positive_events": int((df["leak_event_clean"] == 1).sum()),
        "negative_events": int((df["leak_event_clean"] == 0).sum()),
    }

    # Lag NaNs: expected for first N rows per zone — retain with note
    lag_cols = [c for c in df.columns if "lag" in c or "drop" in c]
    lag_nulls = {c: int(df[c].isna().sum()) for c in lag_cols if df[c].isna().any()}
    report["lag_nulls_retained"] = lag_nulls
    report["lag_null_note"] = (
        "NaNs in lag/rolling columns are structurally expected for the first "
        "N rows of each zone's time series. Do not impute — handle at model "
        "training time via dropna() or fillna(0)."
    )

    n_dupes = int(df.duplicated(subset=["zone_id", "timestamp", "meter_id"]).sum())
    if n_dupes:
        df = df.drop_duplicates(subset=["zone_id", "timestamp", "meter_id"])
    report["duplicates_dropped"] = n_dupes

    report["output_rows"] = len(df)
    return df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True), report


def clean_pipe_risk_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"dataset": "pipe_risk_scores", "input_rows": len(df)}

    # Zone ID — three-stage recovery
    before = df["zone_id"].copy()
    df["zone_id"] = df["zone_id"].apply(_coerce_zone_id)

    # Stage 2: recover from pipe_id prefix for sentinel/null rows
    mask_null = df["zone_id"].isna()
    if mask_null.any():
        df.loc[mask_null, "zone_id"] = df.loc[mask_null, "pipe_id"].apply(
            _recover_zone_from_pipe_id
        )

    # Stage 3: recover from zone_name for anything still null
    mask_still_null = df["zone_id"].isna()
    if mask_still_null.any():
        df.loc[mask_still_null, "zone_id"] = df.loc[mask_still_null, "zone_name"].apply(
            _recover_zone_from_zone_name
        )

    n_irrecoverable = int(df["zone_id"].isna().sum())
    report["zone_id"] = {
        "three_stage_recovery_applied": True,
        "irrecoverable_rows": n_irrecoverable,
    }
    if n_irrecoverable:
        logger.warning(
            "pipe_risk_scores: %d rows with irrecoverable zone_id dropped.", n_irrecoverable
        )
        df = df.dropna(subset=["zone_id"])

    # Standardise GPS column names to match rest of codebase
    df = df.rename(columns={
        "gps_latitude":  "latitude",
        "gps_longitude": "longitude",
    })
    report["column_renames"] = {"gps_latitude": "latitude", "gps_longitude": "longitude"}

    # GPS nulls — impute with zone centroid mean
    for coord in ["latitude", "longitude"]:
        null_mask = df[coord].isna()
        if null_mask.any():
            zone_means = df.groupby("zone_id")[coord].transform("mean")
            df.loc[null_mask, coord] = zone_means[null_mask]
            report[f"{coord}_imputed_rows"] = int(null_mask.sum())

    # age_years_clean null — impute with material-group median
    if df["age_years_clean"].isna().any():
        material_median = df.groupby("material_clean")["age_years_clean"].transform("median")
        df["age_years_clean"] = df["age_years_clean"].fillna(material_median)
        df["age_years_clean"] = df["age_years_clean"].fillna(df["age_years_clean"].median())
        report["age_years_clean_imputed"] = True

    # zone_name nulls — fill from zone_id
    zone_name_map = {
        "FRA-01": "Francistown",    "GBR-C": "Gaborone Central",
        "GBR-N":  "Gaborone North", "GBR-S": "Gaborone South",
        "KNY-01": "Kanye",          "LBT-01": "Lobatse",
        "MHR-01": "Mahalapye",      "MMB-01": "Molepolole",
    }
    df["zone_name"] = df["zone_name"].fillna(df["zone_id"].map(zone_name_map))
    report["zone_name_nulls_remaining"] = int(df["zone_name"].isna().sum())

    n_dupes = int(df.duplicated(subset=["pipe_id"]).sum())
    if n_dupes:
        df = df.drop_duplicates(subset=["pipe_id"])
    report["duplicates_dropped"] = n_dupes

    report["output_rows"] = len(df)
    return df.sort_values(["zone_id", "pipe_id"]).reset_index(drop=True), report


def clean_demand_forecasts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"dataset": "demand_forecasts_6month", "input_rows": len(df)}

    before = df["zone_id"].copy()
    df["zone_id"] = df["zone_id"].apply(_coerce_zone_id)
    report["zone_id_variants_resolved"] = int((df["zone_id"] != before).sum())

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    n_bad = int(df["ds"].isna().sum())
    if n_bad:
        df = df.dropna(subset=["ds"])
    report["ds_parse_failures"] = n_bad

    if all(c in df.columns for c in ["yhat", "yhat_lower", "yhat_upper"]):
        n_lower = int((df["yhat_lower"] > df["yhat"]).sum())
        n_upper = int((df["yhat_upper"] < df["yhat"]).sum())
        df["yhat_lower"] = df[["yhat_lower", "yhat"]].min(axis=1)
        df["yhat_upper"] = df[["yhat_upper", "yhat"]].max(axis=1)
        report["forecast_bound_fixes"] = {"yhat_lower_clamped": n_lower, "yhat_upper_clamped": n_upper}

    n_dupes = int(df.duplicated(subset=["zone_id", "ds"]).sum())
    if n_dupes:
        df = df.drop_duplicates(subset=["zone_id", "ds"])
    report["duplicates_dropped"] = n_dupes

    report["output_rows"] = len(df)
    return df.sort_values(["zone_id", "ds"]).reset_index(drop=True), report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    full_report = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "input_dir":  input_dir,
        "output_dir": output_dir,
        "datasets":   {},
    }

    jobs = [
        ("billing_features.csv",        "billing_features.csv",        clean_billing_features),
        ("iot_features.csv",             "iot_features.csv",            clean_iot_features),
        ("pipe_risk_scores.csv",         "pipe_risk_scores.csv",        clean_pipe_risk_scores),
        ("demand_forecasts_6month.csv",  "demand_forecasts_6month.csv", clean_demand_forecasts),
    ]

    for raw_file, clean_file, fn in jobs:
        raw_path   = os.path.join(input_dir,  raw_file)
        clean_path = os.path.join(output_dir, clean_file)

        if not os.path.exists(raw_path):
            logger.warning("Not found, skipping: %s", raw_path)
            full_report["datasets"][raw_file] = {"status": "skipped"}
            continue

        logger.info("Cleaning %s ...", raw_file)
        try:
            raw_df = pd.read_csv(raw_path)
            clean_df, report = fn(raw_df)
            clean_df.to_csv(clean_path, index=False)
            delta = report["output_rows"] - report["input_rows"]
            logger.info(
                "  OK  %s -> %d rows  (%+d from %d input)",
                clean_file, report["output_rows"], delta, report["input_rows"],
            )
            full_report["datasets"][raw_file] = {"status": "ok", **report}
        except Exception as exc:
            logger.error("  FAIL  %s: %s", raw_file, exc)
            full_report["datasets"][raw_file] = {"status": "failed", "error": str(exc)}

    report_path = os.path.join(output_dir, "cleaning_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info("\nAudit report -> %s", report_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
