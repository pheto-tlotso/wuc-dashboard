"""
src/model_loader.py
-------------------
Centralised, cached model artefact loading.

All models are loaded once per Streamlit session using st.cache_resource,
which means they survive reruns without being re-fetched from S3.
Callers receive None if a model cannot be loaded, and are expected to
handle that case gracefully.
"""

import io
import os
import logging
import joblib
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

import streamlit as st

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
_REGION = os.environ.get("AWS_DEFAULT_REGION", "")
_USE_LOCAL = os.environ.get("USE_LOCAL_MODELS", "false").lower() == "true"
_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Canonical model filenames — aligned with artefact names in S3
_MODEL_KEYS = {
    "leak_detection": "models/xgb_leak_detection.joblib",
    "water_quality":  "models/xgb_water_quality.joblib",
    "pipe_risk":       "models/xgb_pipe_risk.joblib",
}


def _load_joblib_from_s3(key: str):
    """Download and deserialise a joblib artefact from S3."""
    try:
        client = boto3.client("s3", region_name=_REGION or None)
        obj = client.get_object(Bucket=_BUCKET, Key=key)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    except (ClientError, NoCredentialsError) as exc:
        logger.error("S3 load failed for %s: %s", key, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error loading model %s: %s", key, exc)
        return None


def _load_joblib_local(filename: str):
    """Load a joblib artefact from the local models/ directory."""
    path = os.path.join(_LOCAL_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Local model not found: %s", path)
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load local model %s: %s", path, exc)
        return None


@st.cache_resource(show_spinner=False)
def get_leak_model():
    """Return the leak detection XGBoost model (cached for the session)."""
    key = _MODEL_KEYS["leak_detection"]
    if _USE_LOCAL:
        return _load_joblib_local(os.path.basename(key))
    return _load_joblib_from_s3(key)


@st.cache_resource(show_spinner=False)
def get_quality_model():
    """Return the water quality XGBoost model (cached for the session)."""
    key = _MODEL_KEYS["water_quality"]
    if _USE_LOCAL:
        return _load_joblib_local(os.path.basename(key))
    return _load_joblib_from_s3(key)


@st.cache_resource(show_spinner=False)
def get_pipe_risk_model():
    """Return the pipe risk XGBoost model (cached for the session)."""
    key = _MODEL_KEYS["pipe_risk"]
    if _USE_LOCAL:
        return _load_joblib_local(os.path.basename(key))
    return _load_joblib_from_s3(key)
