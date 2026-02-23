from prefect import flow
from prefect.client.schemas.schedules import CronSchedule
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrate.flows.youtube_flow import (
    youtube_analytics_pipeline,
    extract_only_pipeline,
    transform_only_pipeline
)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Prefect workflows")
    parser.add_argument(
        "--deployment",
        choices=["all", "daily", "extract", "transform"],
        default="all",
        help="Chọn deployment để triển khai"
    )
    
    args = parser.parse_args()