"""
data_pipeline/generate_synthetic_data.py
-----------------------------------------
Synthetic data generator for the WUC NRW prototype.

FIXED VERSION — all column names are standardised at the point of generation.
This means the cleaning pipeline receives consistently named data, and the
dashboard never has to deal with zone_id vs zone-id vs ZONE_ID mismatches.

Column naming conventions enforced here:
  - All column names: lowercase, underscore-separated (snake_case)
  - Zone IDs: always 'zone_id' (never 'zone-id', 'ZONE_ID', 'Zone_ID')
  - Zone ID values: always 'ZONE_001' format (zero-padded, 3 digits)
  - Timestamps: always 'timestamp' (never 'time', 'date_time', 'Timestamp')
  - All measurement columns: lowercase with units in name where helpful

Run this script to regenerate all synthetic datasets:
  python data_pipeline/generate_synthetic_data.py

Outputs written to: data/synthetic/
  - sensor_readings.csv
  - pipe_inventory.csv
  - demand_forecast.csv
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_ZONES = 8
N_SENSOR_ROWS = 5000
N_PIPE_ROWS = 400
FORECAST_HORIZON_DAYS = 90

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

# Canonical zone ID format — defined once, used everywhere
ZONE_IDS = [f"ZONE_{str(i).zfill(3)}" for i in range(1, N_ZONES + 1)]
# e.g. ['ZONE_001', 'ZONE_002', ..., 'ZONE_008']

PIPE_MATERIALS = ["uPVC", "steel", "cast_iron", "HDPE", "asbestos_cement"]
MATERIAL_CODES = {m: i for i, m in enumerate(PIPE_MATERIALS)}

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zone_series(n: int) -> pd.Series:
    """Return a Series of random zone IDs using the canonical format."""
    return pd.Series(np.random.choice(ZONE_IDS, size=n), name="zone_id")


def _timestamp_series(n: int,
                       start: str = "2024-01-01",
                       freq_minutes: int = 15) -> pd.Series:
    """Return a Series of evenly-spaced timestamps."""
    base = pd.Timestamp(start)
    timestamps = [base + timedelta(minutes=i * freq_minutes) for i in range(n)]
    return pd.Series(timestamps, name="timestamp")


# ── Dataset 1: Sensor Readings ─────────────────────────────────────────────────

def generate_sensor_readings() -> pd.DataFrame:
    """
    Simulated IoT sensor readings from distribution network monitoring points.

    Columns (all snake_case, all named consistently):
      timestamp         — datetime of reading
      zone_id           — canonical ZONE_XXX identifier
      sensor_id         — unique sensor string
      flow_rate         — flow in m³/h
      pressure_bar      — line pressure in bar
      pressure_variance — rolling variance proxy
      turbidity         — NTU
      ph_level          — 0–14
      chlorine_residual — mg/L
      temperature_c     — degrees Celsius
      conductivity      — µS/cm
      anomaly_score     — pre-computed isolation forest score
      hour_of_day       — 0–23 (derived from timestamp)
      day_of_week       — 0=Monday (derived from timestamp)
      leak_label        — 1 if simulated leak event, else 0
    """
    n = N_SENSOR_ROWS
    zones = _zone_series(n)
    timestamps = _timestamp_series(n)

    flow_base = np.random.normal(loc=45, scale=8, size=n).clip(min=0)
    pressure_base = np.random.normal(loc=3.5, scale=0.4, size=n).clip(min=0.5)

    # Inject ~8% leak events — these drive model training
    leak_mask = np.random.rand(n) < 0.08
    flow_base[leak_mask] += np.random.normal(loc=15, scale=4, size=leak_mask.sum())
    pressure_base[leak_mask] -= np.random.normal(loc=0.8, scale=0.2, size=leak_mask.sum())
    pressure_base = pressure_base.clip(min=0.1)

    df = pd.DataFrame({
        "timestamp":         timestamps,
        "zone_id":           zones,
        "sensor_id":         [f"SENSOR_{str(i).zfill(4)}" for i in range(n)],
        "flow_rate":         flow_base.round(3),
        "pressure_bar":      pressure_base.round(3),
        "pressure_variance": np.abs(np.random.normal(loc=0.05, scale=0.02, size=n)).round(4),
        "turbidity":         np.random.exponential(scale=1.2, size=n).round(2),
        "ph_level":          np.random.normal(loc=7.2, scale=0.4, size=n).clip(6.0, 9.0).round(2),
        "chlorine_residual": np.random.normal(loc=0.5, scale=0.15, size=n).clip(0.0, 2.0).round(3),
        "temperature_c":     np.random.normal(loc=22, scale=3, size=n).clip(5, 40).round(1),
        "conductivity":      np.random.normal(loc=420, scale=60, size=n).clip(50, 900).round(1),
        "anomaly_score":     np.random.uniform(-0.3, 0.3, size=n).round(4),
        "hour_of_day":       pd.to_datetime(timestamps).dt.hour,
        "day_of_week":       pd.to_datetime(timestamps).dt.dayofweek,
        "leak_label":        leak_mask.astype(int),
    })

    return df


# ── Dataset 2: Pipe Inventory ──────────────────────────────────────────────────

def generate_pipe_inventory() -> pd.DataFrame:
    """
    Simulated pipe segment inventory and inspection records.

    Columns:
      pipe_id               — unique segment identifier
      zone_id               — canonical ZONE_XXX identifier
      latitude              — WGS84 latitude (Gaborone area)
      longitude             — WGS84 longitude
      pipe_age_years        — years since installation
      material              — pipe material string
      material_encoded      — integer code for ML
      diameter_mm           — internal diameter in mm
      pressure_zone         — integer zone category (1–4)
      repair_count          — historical repairs on this segment
      soil_corrosivity_index — 0.0–1.0 index
      last_inspection_date  — date string
      risk_label            — 1 = high risk segment (for model training)
    """
    n = N_PIPE_ROWS
    zones = _zone_series(n)
    materials = np.random.choice(PIPE_MATERIALS, size=n)

    # Gaborone area coordinates with small random scatter
    lats = np.random.normal(loc=-24.654, scale=0.05, size=n)
    lons = np.random.normal(loc=25.908, scale=0.06, size=n)

    age = np.random.exponential(scale=18, size=n).clip(1, 60).astype(int)
    repair_count = np.random.poisson(lam=1.5, size=n)
    corrosivity = np.random.beta(2, 5, size=n).round(3)

    # Risk label: older pipes, more repairs, higher corrosivity → higher risk
    risk_score_raw = (age / 60) * 0.4 + (repair_count / 10) * 0.3 + corrosivity * 0.3
    risk_label = (risk_score_raw > 0.35).astype(int)

    base_date = datetime(2024, 1, 1)
    inspection_dates = [
        (base_date - timedelta(days=int(np.random.exponential(scale=180)))).strftime("%Y-%m-%d")
        for _ in range(n)
    ]

    df = pd.DataFrame({
        "pipe_id":                [f"PIPE_{str(i).zfill(4)}" for i in range(n)],
        "zone_id":                zones,
        "latitude":               lats.round(6),
        "longitude":              lons.round(6),
        "pipe_age_years":         age,
        "material":               materials,
        "material_encoded":       [MATERIAL_CODES[m] for m in materials],
        "diameter_mm":            np.random.choice([50, 75, 100, 150, 200, 300], size=n),
        "pressure_zone":          np.random.randint(1, 5, size=n),
        "repair_count":           repair_count,
        "soil_corrosivity_index": corrosivity,
        "last_inspection_date":   inspection_dates,
        "risk_label":             risk_label,
    })

    return df


# ── Dataset 3: Demand Forecast ─────────────────────────────────────────────────

def generate_demand_forecast() -> pd.DataFrame:
    """
    Simulated Prophet forecast output per zone.

    Columns match Prophet's standard output convention:
      ds          — forecast date (Prophet convention)
      zone_id     — canonical ZONE_XXX identifier
      yhat        — point forecast (m³)
      yhat_lower  — lower confidence bound
      yhat_upper  — upper confidence bound

    Note: 'ds' is the Prophet-standard date column name and is preserved
    intentionally. find_date_column() in feature_utils.py recognises it.
    """
    records = []
    base_date = datetime(2024, 6, 1)

    for zone in ZONE_IDS:
        zone_base_demand = np.random.uniform(800, 2000)
        for day in range(FORECAST_HORIZON_DAYS):
            date = base_date + timedelta(days=day)
            # Weekday effect
            weekday_factor = 1.0 if date.weekday() < 5 else 0.75
            # Seasonal trend (simple sine)
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * day / 30)
            # Noise
            noise = np.random.normal(0, 30)

            yhat = zone_base_demand * weekday_factor * seasonal + noise
            margin = np.random.uniform(40, 80)

            records.append({
                "ds":         date.strftime("%Y-%m-%d"),
                "zone_id":    zone,
                "yhat":       round(max(yhat, 0), 1),
                "yhat_lower": round(max(yhat - margin, 0), 1),
                "yhat_upper": round(yhat + margin, 1),
            })

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic datasets...")

    sensor_df = generate_sensor_readings()
    sensor_path = os.path.join(OUTPUT_DIR, "sensor_readings_raw.csv")
    sensor_df.to_csv(sensor_path, index=False)
    print(f"  ✓ Sensor readings: {len(sensor_df):,} rows → {sensor_path}")

    pipe_df = generate_pipe_inventory()
    pipe_path = os.path.join(OUTPUT_DIR, "pipe_inventory_raw.csv")
    pipe_df.to_csv(pipe_path, index=False)
    print(f"  ✓ Pipe inventory:  {len(pipe_df):,} rows → {pipe_path}")

    forecast_df = generate_demand_forecast()
    forecast_path = os.path.join(OUTPUT_DIR, "demand_forecast_raw.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"  ✓ Demand forecast: {len(forecast_df):,} rows → {forecast_path}")

    print("\nAll datasets generated. Run clean.py next to produce validated outputs.")


if __name__ == "__main__":
    main()
