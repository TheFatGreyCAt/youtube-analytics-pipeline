"""
Configuration for the ML module.
Reads environment variables from .env file at the project root.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project root & .env ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Google Cloud / BigQuery ──────────────────────────────────────────────────
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "project-8fd99edc-9e20-4b82-b43")
CREDENTIALS_PATH: str = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "credentials/project-8fd99edc-9e20-4b82-b43-41fc5f2ccbcd.json",
)

# Resolve relative paths to absolute
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str(PROJECT_ROOT / CREDENTIALS_PATH)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# ── BigQuery dataset names ───────────────────────────────────────────────────
BQ_INTERMEDIATE_DATASET: str = "intermediate"

# Intermediate tables used for ML feature engineering
INT_VIDEOS_ENHANCED_TABLE: str = "int_videos__enhanced"
INT_ENGAGEMENT_METRICS_TABLE: str = "int_engagement_metrics"
INT_CHANNEL_SUMMARY_TABLE: str = "int_channel_summary"

# ── BigQuery location ────────────────────────────────────────────────────────
BQ_LOCATION: str = "asia-southeast1"
