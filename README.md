# 💧 WUC NRW Reduction Prototype

**Client:** Water Utilities Corporation Botswana (WUC)  
**Built by:** Rooteddeck Analytics — Tlotso T Pheto ( Revenue and Operational Analyst)  
**Sprint:** 17 March – 20 April 2026 · 5 weeks · 25 working days  
**Stack:** Python · AWS · Streamlit · Power BI  
**Region:** AWS af-south-1 (Cape Town)

> ⚠️ **IP Notice:** This prototype was built as part of a client engagement for Water Utilities Corporation Botswana through Rooteddeck Analytics. All intellectual property belongs to Rooteddeck Analytics. This repository documents the technical implementation for portfolio and handover purposes only.

---

## Overview

Botswana's Water Utilities Corporation loses an estimated **40% of treated water** as Non-Revenue Water (NRW) — representing over **P1.25 billion** in annual revenue at risk. This prototype demonstrates how a cloud-based machine learning pipeline can detect leaks, predict water quality failures, score pipe risk, and forecast demand across 8 distribution zones.

The prototype was built in a single 5-week sprint by one junior data analyst using AWS free-tier services, Python, and open-source ML libraries.

---

## Live Dashboard

🔗 **Streamlit Dashboard:** [wuc-dashboard.streamlit.app](https://share.streamlit.io)  
📊 **Power BI Executive Report:** Available as `.pbix` file in `/reports/`

---

## Business Impact

| Metric | Value |
|--------|-------|
| NRW rate detected | ~40% across 8 zones |
| Projected annual saving at 3% NRW reduction | P37.5M+ |
| Revenue base | P1.25B/yr |
| Cloud investment payback | < 1 day |
| Zones monitored | 8 (GBR-N, GBR-S, GBR-C, FRA-01, MHR-01, KNY-01, LBT-01, MMB-01) |

---

## Architecture

```
IoT Core (mock MQTT)
    │
    ▼
Lambda (wuc-iot-to-s3)
    │
    ▼
S3 — wuc-prototype-data-tp (af-south-1)
├── hot/     → forecasts, pipe scores
├── warm/    → features, models, metrics
└── cold/    → raw data, archive
    │
    ├──► Glue Crawler → Glue Catalog (wuc_db)
    │         │
    │         ▼
    │       Athena (SQL on S3)
    │
    ▼
SageMaker (ml.t3.medium · eu-west-2)
    │   Feature engineering + model training
    │
    ├──► Isolation Forest   → leak_detection
    ├──► XGBoost            → quality_failure
    ├──► XGBoost Multiclass → pipe_risk
    └──► Prophet ×8         → demand_forecast
              │
              ▼
         Streamlit Cloud  ←── GitHub (CI/CD)
         Power BI Desktop ←── CSV export from S3
```

---

## ML Models

### 1. Leak Detection — Isolation Forest
- **Data:** 14,615 IoT sensor rows (pressure, flow rate, timestamps)
- **Target:** Anomaly detection — identify pressure/flow signatures of leaks
- **Key features:** `pressure_bar`, `flow_rate_m3hr`, rolling means, lag features
- **Metric:** AUC-ROC > 0.75
- **Output:** `predicted_leak` flag per reading + `anomaly_score`
- **Insight:** Leaks concentrate in the **2–4 AM Minimum Night Flow window** — not morning peaks as raw flow suggested

### 2. Water Quality Failure — XGBoost
- **Data:** 2,856 quality sample rows (chlorine, pH, turbidity, coagulant dose)
- **Target:** Binary — predict quality failure before it reaches consumer taps
- **Metric:** AUC-ROC > 0.85, optimised probability threshold
- **Output:** `fail_prob` + `predicted_fail` per sample
- **Insight:** Top SHAP driver is coagulant dose — treatment process control is the primary lever

### 3. Pipe Risk Scoring — XGBoost Multiclass
- **Data:** 1,885 pipe segments (age, material, GPS, pressure history)
- **Target:** 4-class risk: LOW / MEDIUM / HIGH / CRITICAL
- **Metric:** F1-Macro > 0.80
- **Output:** `predicted_risk` + probability per class per pipe
- **Insight:** 175 pipes classified CRITICAL — concentrated in KNY-01 and LBT-01 zones

### 4. Demand Forecasting — Prophet ×8 Zones
- **Data:** 6,492 billing rows across all zones
- **Target:** 6-month forward demand forecast per zone
- **Metric:** MAPE < 15% across all zones
- **Output:** `yhat`, `yhat_lower`, `yhat_upper` per zone per month to Dec 2026
- **Insight:** Peak demand July–August (dry season). KNY-01 and LBT-01 highest forecast volumes — also highest NRW zones

---

## Dashboard — 4 Tabs

### Tab 1 — 🔴 Leak Detection
- Zone-level leak rate bar chart
- Pressure distribution: leak vs normal (overlapping histograms)
- **Time-of-day leak distribution** — MNF window (2–4 AM) highlighted
- **Per-zone MNF breakdown** — ranked horizontal bar chart with risk table
- CEO insight callout: worst zone by MNF leak share

### Tab 2 — 🧪 Water Quality
- Quality failure rate by zone
- Chlorine vs pH scatter — Pass/Fail coloured
- WHO minimum chlorine reference line (0.2 mg/L)
- SHAP top feature callout

### Tab 3 — 🗺️ Pipe Risk Map
- Interactive Folium map — all pipe segments colour-coded by risk
- MarkerCluster for performance
- Toggle: show CRITICAL only
- Top 10 CRITICAL pipes table

### Tab 4 — 📈 Demand Forecast
- Zone deep-dive: historical + forecast line with 80% confidence band
- All-zone forecast comparison
- Dry season shading (July–August)
- Zone MAPE table

---

## Repository Structure

```
wuc-dashboard/
├── app.py                          # Streamlit dashboard (842 lines)
├── requirements.txt                # Pinned Python dependencies
├── Dockerfile                      # Docker container definition
├── docker-compose.yml              # Local/EC2 deployment
├── .env.example                    # Credential template (never commit .env)
├── .gitignore                      # Excludes credentials, data, PKLs
├── .streamlit/
│   └── config.toml                 # Streamlit server config
├── scripts/
│   └── s3_helpers.py               # Shared S3 data loading functions
├── reports/
│   └── WUC_NRW_Executive_Report.pbix  # Power BI executive report
└── README.md
```

---

## S3 Bucket Structure

```
wuc-prototype-data-tp/
├── hot/
│   ├── forecasts/demand_forecasts_6month.csv
│   └── pipe_scores/pipe_risk_scores.csv
├── warm/
│   ├── features/
│   │   ├── billing_features.csv
│   │   ├── iot_features.csv
│   │   └── quality_features.csv
│   └── models/
│       ├── isolation_forest_leak.pkl
│       ├── isolation_forest_metrics.json
│       ├── xgboost_quality.pkl
│       ├── xgboost_quality_metrics.json
│       ├── logreg_pipe_risk_metrics.json
│       └── prophet_demand_metrics.json
└── cold/
    └── raw/WUC_Clean_Dataset.xlsx
```

---

## Deployment

### Option 1 — Streamlit Cloud (current, free forever)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Add secrets in Streamlit Cloud dashboard:
```toml
aws_default_region = "af-south-1"
aws_access_key_id = "YOUR_KEY"
aws_secret_access_key = "YOUR_SECRET"
```

### Option 2 — Docker (local or EC2 free tier)

```bash
# Clone and configure
git clone https://github.com/pheto-tlotso/wuc-dashboard
cd wuc-dashboard
cp .env.example .env
nano .env  # fill in AWS credentials

# Build and run
docker compose up --build

# Dashboard at http://localhost:8501
```

### Option 3 — EC2 t2.micro (free tier, 750hr/mo)

```bash
# On EC2 Amazon Linux 2
sudo yum install docker git -y
sudo service docker start
sudo usermod -aG docker ec2-user

git clone https://github.com/pheto-tlotso/wuc-dashboard
cd wuc-dashboard
cp .env.example .env && nano .env
docker compose up -d

# Dashboard at http://YOUR-EC2-IP:8501
```

---

## Credential Resolution

The app resolves AWS credentials in this order — **no code changes needed when switching environments:**

```
1. st.secrets[]       → Streamlit Cloud
2. .env / os.environ  → Docker / EC2
3. IAM instance role  → EC2 with attached role (most secure)
```

---

## AWS Cost

This prototype runs at **$0/month** on AWS free tier:

| Service | Free allowance | Usage |
|---------|---------------|-------|
| S3 | 5 GB storage | ~50 MB |
| Lambda | 1M requests/mo | Occasional IoT sim |
| Glue Crawler | 1M objects | Single crawl |
| Athena | 1 TB scanned/mo | MB-scale queries |
| IoT Core | 250K messages/mo | Mock simulation |
| Streamlit Cloud | Unlimited | Dashboard hosting |

> ⚠️ **SageMaker** (ml.t3.medium) has a **12-month free trial**. Stop the notebook instance after training is complete to avoid charges. Use Docker + EC2 t2.micro for ongoing inference.

---

## Local Development

```bash
# Clone
git clone https://github.com/pheto-tlotso/wuc-dashboard
cd wuc-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set credentials
cp .env.example .env
# Edit .env with your AWS credentials

# Run
streamlit run app.py
```

---

## Data

The source dataset is `WUC_Clean_Dataset.xlsx` — 25,848 rows across 4 sheets:

| Sheet | Rows | Description |
|-------|------|-------------|
| IoT sensors | 14,615 | Hourly pressure and flow readings per zone |
| Water quality | 2,856 | Lab sample results per zone |
| Pipe inventory | 1,885 | Pipe segments with age, material, GPS |
| Billing | 6,492 | Monthly consumption and revenue per zone |

Raw data is not included in this repository. Contact Rooteddeck Analytics for access.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML | scikit-learn · XGBoost · Prophet · SHAP |
| Dashboard | Streamlit · Plotly · Folium |
| BI Report | Power BI Desktop |
| Cloud | AWS S3 · Lambda · IoT Core · Glue · Athena · SageMaker |
| DevOps | Docker · GitHub Actions · EC2 |
| Data | pandas · numpy · pyarrow · boto3 |

---

## Sprint Timeline

| Week | Dates | Focus | Status |
|------|-------|-------|--------|
| W1 | 17–23 Mar | Environment setup · Data cleaning · EDA | ✅ Complete |
| W2 | 24–30 Mar | AWS pipeline · S3 data lake · IoT Core | ✅ Complete |
| W3 | 31 Mar–6 Apr | All 4 ML models trained · SHAP evaluation | ✅ Complete |
| W4 | 7–13 Apr | Streamlit dashboard · 4 tabs · Deployment | ✅ Complete |
| W5 | 14–20 Apr | Power BI · Docker · CEO presentation | ✅ Complete |

---

## Author

**Tlotso Tshepang Pheto**  
Operations & Revenue Intelligence Analyst  
Rooteddeck Analytics · Gaborone, Botswana  
GitHub: [@pheto-tlotso](https://github.com/Tlotso)

---

*Built for Water Utilities Corporation Botswana · Sprint prototype · April 2026*
