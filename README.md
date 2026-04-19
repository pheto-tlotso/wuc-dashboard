# WUC Non-Revenue Water (NRW) Intelligence Platform
**Prototype v1.0 — Rooteddeck / Water Utilities Corporation Botswana**

A machine-learning dashboard for detecting, predicting, and monitoring non-revenue water loss across WUC's distribution network.

---

## Overview

This prototype demonstrates four analytical capabilities:
1. **Leak Detection** — XGBoost binary classifier (F1 ≈ 0.84, AUC ≈ 0.97) identifying anomalous flow patterns indicative of leaks
2. **Water Quality Monitoring** — Multi-class XGBoost classifier flagging quality deviations by sensor zone
3. **Pipe Risk Scoring** — Risk stratification model for prioritising pipe inspection and replacement
4. **Demand Forecasting** — Zone-level demand forecasting using Prophet-generated baseline projections

---

## Repository Structure

```
.
├── app.py                  # Streamlit entry point (UI only — imports from src/)
├── src/
│   ├── data_access.py      # S3/local data loading utilities
│   ├── model_loader.py     # Cached model artefact loading
│   ├── feature_utils.py    # Feature alignment and preprocessing helpers
│   └── chart_utils.py      # Reusable Plotly chart builders
├── models/                 # Model artefacts (.pkl/.joblib) — not committed to repo
│   └── .gitkeep
├── data/
│   └── sample/             # Anonymised sample data for local dev/demo
├── docs/
│   ├── architecture.md     # Cloud pipeline and data flow documentation
│   └── model_notes.md      # Model assumptions, feature descriptions, known limitations
├── requirements.txt
└── .env.example            # Environment variable template (no secrets committed)
```

---

## Setup

### Prerequisites
- Python 3.10+
- AWS credentials with read access to the project S3 bucket (contact project lead)
- Streamlit ≥ 1.32

### Local Development

```bash
# 1. Clone and create environment
git clone <repo-url>
cd wuc-nrw-dashboard
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your credentials (never commit .env)

# 4. Run
streamlit run app.py
```

### Environment Variables

All runtime configuration is managed via environment variables. Copy `.env.example` to `.env` and populate:

```
AWS_S3_BUCKET=<your-bucket-name>
AWS_DEFAULT_REGION=<your-region>
```

**Never commit `.env` or any file containing credentials, bucket names, or IAM identifiers.**

---

## Model Artefacts

Model `.pkl` files are excluded from version control (see `.gitignore`). They are stored in S3 and fetched at runtime via `src/model_loader.py`. For local development without S3 access, place artefacts in the `models/` directory and set `USE_LOCAL_MODELS=true` in `.env`.

---

## Known Limitations (Prototype Scope)

- Data is synthetic/simulated — not derived from live WUC sensor feeds
- Demand forecasting relies on precomputed Prophet output files; real-time re-fitting is not implemented at this stage
- The pipe risk model is a gradient-boosted classifier; the label "Random Forest" used in an earlier version was incorrect and has been corrected throughout
- Zone IDs must match the format used during model training (`ZONE_001` etc.) — see `src/feature_utils.py`

---

## Attribution

**IP Owner:** Rooteddeck  
**Implementation:** Tlotso Tshepang Pheto  
**Status:** Internal prototype — not for public distribution

---

## Contact

For questions about the technical implementation, contact the project team via internal channels.
