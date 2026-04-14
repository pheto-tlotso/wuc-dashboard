# WUC NRW Dashboard — Deployment Record

## Live URL
https://wuc-nrw-prototype.streamlit.app

## GitHub Repo
https://github.com/pheto-tlotso/wuc-dashboard

## Streamlit Cloud
Account: tlotso.pheto@rooteddeck.com
App name: wuc-nrw-prototype

## AWS Secrets (stored in Streamlit Cloud — NOT in this file)
Key: aws_access_key_id
Key: aws_secret_access_key
Key: aws_default_region = af-south-1
IAM User: wuc_dashboard_readonly (S3 read-only)

## Data sources (all in wuc-prototype-data-tp)
warm/features/iot_features.csv
warm/features/quality_features.csv
warm/models/isolation_forest_leak.pkl
warm/models/xgboost_quality.pkl
warm/models/logreg_pipe_risk.pkl
warm/models/prophet_demand_metrics.json
hot/pipe_scores/pipe_risk_scores.csv
hot/forecasts/demand_forecasts_6month.csv

## To redeploy after changes
git add app.py && git commit -m "fix: ..." && git push origin main
Streamlit Cloud auto-redeploys within 2 minutes