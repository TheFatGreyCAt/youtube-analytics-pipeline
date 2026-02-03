import os
from datetime import datetime
from dotenv import load_dotenv
import logging

from prefect import flow, task
from prefect.logging import get_run_logger
from prefect_gcp.cloud_storage import GcsBucket
from prefect_gcp.bigquery import BigQueryWarehouse

# Import pipeline modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extract'))
from yt_pipeline import YouTubeETLPipeline

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv('YOUTUBE_API_KEY')
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
DATASET_ID = os.getenv('BQ_DATASET_ID', 'youtube_raw')
CHANNEL_ID = os.getenv('YOUTUBE_CHANNEL_ID')


@task(name="extract_youtube_data", retries=2, retry_delay_seconds=60)
def extract_youtube_data(channel_id: str, api_key: str, project_id: str, dataset_id: str) -> bool:
    """
    Task: Extract YouTube channel data and load to BigQuery
    
    Args:
        channel_id: YouTube channel ID
        api_key: YouTube API key
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
    
    Returns:
        bool: Success status
    """
    logger_prefect = get_run_logger()
    logger_prefect.info(f"Starting YouTube data extraction for channel: {channel_id}")
    
    try:
        pipeline = YouTubeETLPipeline(api_key, project_id, dataset_id)
        success = pipeline.run(channel_id)
        
        if success:
            logger_prefect.info(f"✅ Successfully extracted data for channel {channel_id}")
        else:
            logger_prefect.error(f"❌ Failed to extract data for channel {channel_id}")
        
        return success
    
    except Exception as e:
        logger_prefect.error(f"❌ Error during extraction: {str(e)}")
        raise


@task(name="validate_bigquery_data")
def validate_bigquery_data(project_id: str, dataset_id: str) -> dict:
    """
    Task: Validate data loaded to BigQuery
    
    Returns:
        dict: Validation results
    """
    logger_prefect = get_run_logger()
    
    try:
        from google.cloud import bigquery
        
        client = bigquery.Client(project=project_id)
        results = {}
        
        tables = ['dim_channel', 'dim_video', 'dim_playlist']
        
        for table_name in tables:
            query = f"""
            SELECT COUNT(*) as row_count 
            FROM `{project_id}.{dataset_id}.{table_name}`
            """
            result = client.query(query).result()
            row_count = list(result)[0].row_count
            results[table_name] = row_count
            logger_prefect.info(f"Table {table_name}: {row_count} rows")
        
        return results
    
    except Exception as e:
        logger_prefect.error(f"❌ Validation failed: {str(e)}")
        raise


@flow(name="youtube_analytics_pipeline", description="YouTube data extraction and loading pipeline")
def youtube_analytics_flow(
    channel_id: str = None,
    api_key: str = None,
    project_id: str = None,
    dataset_id: str = None
):
    """
    Main Prefect flow for YouTube analytics pipeline
    
    Steps:
    1. Extract YouTube data
    2. Validate BigQuery data
    """
    
    logger_prefect = get_run_logger()
    
    # Use environment variables if parameters not provided
    channel_id = channel_id or CHANNEL_ID
    api_key = api_key or API_KEY
    project_id = project_id or PROJECT_ID
    dataset_id = dataset_id or DATASET_ID
    
    logger_prefect.info("=" * 60)
    logger_prefect.info("🚀 Starting YouTube Analytics Pipeline Flow")
    logger_prefect.info("=" * 60)
    logger_prefect.info(f"Channel ID: {channel_id}")
    logger_prefect.info(f"Project ID: {project_id}")
    logger_prefect.info(f"Dataset ID: {dataset_id}")
    
    # Step 1: Extract data
    success = extract_youtube_data(channel_id, api_key, project_id, dataset_id)
    
    if success:
        # Step 2: Validate data
        validation_results = validate_bigquery_data(project_id, dataset_id)
        
        logger_prefect.info("\n" + "=" * 60)
        logger_prefect.info("📊 VALIDATION RESULTS")
        logger_prefect.info("=" * 60)
        for table, count in validation_results.items():
            logger_prefect.info(f"✅ {table}: {count} rows")
        logger_prefect.info("=" * 60)
    
    return success


if __name__ == "__main__":
    # Run flow locally
    youtube_analytics_flow()
