"""
src/data_access.py
------------------
All data loading for the WUC NRW dashboard.
Centralises S3 and local file access so app.py stays free of I/O logic.
"""

import os
import io
import logging
import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Resolved from environment variables only — no literals in source
_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
_REGION = os.environ.get("AWS_DEFAULT_REGION", "")
_USE_LOCAL = os.environ.get("USE_LOCAL_MODELS", "false").lower() == "true"

# S3 key prefixes
_PREFIX_DATA = "data/"
_PREFIX_MODELS = "models/"
_PREFIX_METRICS = "metrics/"


def _s3_client():
    """Return a boto3 S3 client using ambient credentials (IAM role / env vars)."""
    if not _BUCKET:
        raise EnvironmentError(
            "AWS_S3_BUCKET is not set. "
            "Copy .env.example to .env and configure your environment variables."
        )
    return boto3.client("s3", region_name=_REGION or None)


def load_csv_from_s3(key: str) -> pd.DataFrame:
    """
    Load a CSV from S3 and return a DataFrame.

    Parameters
    ----------
    key : str
        Object key relative to the bucket root (e.g. 'data/sensor_readings.csv').

    Returns
    -------
    pd.DataFrame
        Empty DataFrame on failure rather than raising, so callers can guard
        with ``df.empty`` before downstream processing.
    """
    try:
        client = _s3_client()
        obj = client.get_object(Bucket=_BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except (ClientError, NoCredentialsError) as exc:
        logger.error("Failed to load %s from S3: %s", key, exc)
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error loading %s: %s", key, exc)
        return pd.DataFrame()


def load_json_from_s3(key: str) -> dict:
    """
    Load a JSON object from S3 and return a dict.
    Returns an empty dict on failure.
    """
    import json
    try:
        client = _s3_client()
        obj = client.get_object(Bucket=_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load JSON %s from S3: %s", key, exc)
        return {}


def load_sensor_data() -> pd.DataFrame:
    """Load the main sensor readings dataset."""
    return load_csv_from_s3(f"{_PREFIX_DATA}sensor_readings.csv")


def load_pipe_data() -> pd.DataFrame:
    """Load the pipe inventory / inspection dataset."""
    return load_csv_from_s3(f"{_PREFIX_DATA}pipe_inventory.csv")


def load_forecast_data() -> pd.DataFrame:
    """
    Load precomputed Prophet forecast output.
    Returns empty DataFrame if unavailable — callers must check .empty.
    """
    return load_csv_from_s3(f"{_PREFIX_DATA}demand_forecast.csv")


def load_metrics(model_name: str) -> dict:
    """
    Load a model's evaluation metrics JSON.

    Parameters
    ----------
    model_name : str
        One of: 'leak_detection', 'water_quality', 'pipe_risk', 'demand_forecast'
    """
    return load_json_from_s3(f"{_PREFIX_METRICS}{model_name}_metrics.json")
