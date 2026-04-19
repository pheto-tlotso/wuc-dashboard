# Model Notes — WUC NRW Prototype

## Model Inventory

| Section | Algorithm | Artefact File | Task |
|---|---|---|---|
| Leak Detection | XGBoost (binary classifier) | `xgb_leak_detection.joblib` | Predict leak probability per reading |
| Water Quality | XGBoost (multi-class classifier) | `xgb_water_quality.joblib` | Flag quality deviations by type |
| Pipe Risk | XGBoost (binary classifier) | `xgb_pipe_risk.joblib` | Risk-score pipe segments for inspection |
| Demand Forecast | Prophet (time series) | Precomputed CSV via `demand_forecast.csv` | Zone-level demand projection |

> **Naming correction**: An earlier version of the app labelled the pipe risk section as "Random Forest" and an earlier metrics file was named `logistic_regression_output.json`. Both were incorrect. All references have been updated to reflect the actual algorithm: XGBoost gradient-boosted classifier.

---

## Feature Descriptions

### Leak Detection
| Feature | Description |
|---|---|
| `flow_rate` | Instantaneous flow reading (m³/h) |
| `pressure_variance` | Rolling pressure variance over 1-hour window |
| `hour_of_day` | Hour extracted from timestamp (0–23) |
| `day_of_week` | Day index (0=Monday) |
| `zone_id_encoded` | Integer-encoded zone identifier |
| `anomaly_score` | Isolation Forest anomaly score (pre-computed) |

### Pipe Risk
| Feature | Description |
|---|---|
| `pipe_age_years` | Years since installation |
| `material_encoded` | Pipe material as integer code |
| `diameter_mm` | Internal pipe diameter |
| `pressure_zone` | Pressure zone category (integer) |
| `repair_count` | Number of historical repairs on this segment |
| `soil_corrosivity_index` | Soil corrosivity index at pipe location |

---

## Demand Forecasting Notes

Demand forecasts are generated offline using Facebook Prophet and stored as CSV artefacts. The dashboard reads these precomputed outputs — it does not refit Prophet at runtime.

This is a deliberate prototype-stage decision. Real-time Prophet refitting would require:
- A scheduled Lambda/Glue job to refresh forecasts on a set cadence
- A more structured S3 path convention per zone and forecast horizon
- Validation logic for newly generated outputs before they replace existing ones

These are planned for a production iteration.

---

## Known Limitations

1. All data is synthetic — performance metrics reflect synthetic validation sets
2. Zone ID format must be `ZONE_001`, `ZONE_002`, etc. (see `feature_utils.encode_zone_id`)
3. No real-time sensor stream integration in this prototype
4. Model SHAP explanations are available offline but not yet surfaced in the dashboard UI
