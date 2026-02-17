"""
Validation Tasks - Data quality and health check tasks
"""
from prefect import task
from prefect.artifacts import create_table_artifact
from typing import Dict
import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extract.db_manager import PostgresManager, BigQueryManager


@task(name="Health Check - PostgreSQL", tags=["health", "database"], retries=3, retry_delay_seconds=10)
def check_postgres_health_task() -> Dict:
    try:
        pg_manager = PostgresManager()
        channels = pg_manager.list_channels()
        
        return {
            'status': 'healthy',
            'service': 'PostgreSQL',
            'channels_count': len(channels),
            'message': f'Connected successfully. {len(channels)} channels configured.'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'service': 'PostgreSQL',
            'error': str(e),
            'message': f'Connection failed: {str(e)}'
        }


@task(name="Health Check - BigQuery", tags=["health", "database"], retries=3, retry_delay_seconds=10)
def check_bigquery_health_task() -> Dict:
    try:
        bq_manager = BigQueryManager()
        dataset_id = f"{bq_manager.project_id}.{bq_manager.dataset_id}"
        tables = list(bq_manager.client.list_tables(dataset_id))
        
        return {
            'status': 'healthy',
            'service': 'BigQuery',
            'dataset': dataset_id,
            'tables_count': len(tables),
            'message': f'Connected successfully. {len(tables)} tables found.'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'service': 'BigQuery',
            'error': str(e),
            'message': f'Connection failed: {str(e)}'
        }


@task(name="Check API Quota", tags=["monitoring", "api"], retries=2)
def check_api_quota_task() -> Dict:
    pg_manager = PostgresManager()
    history = pg_manager.get_crawl_history(limit=100)
    
    estimated_quota = len(history) * 75
    daily_limit = 10000
    percentage = (estimated_quota / daily_limit * 100)
    
    return {
        'estimated_quota_used': estimated_quota,
        'daily_limit': daily_limit,
        'percentage_used': percentage,
        'status': 'warning' if percentage > 80 else 'ok',
        'message': f'Estimated {percentage:.1f}% of daily quota used'
    }


@task(name="Check Data Freshness", tags=["monitoring", "quality"], retries=2)
def check_data_freshness_task() -> Dict:
    pg_manager = PostgresManager()
    channels = pg_manager.list_channels()
    
    now = datetime.now()
    stale_threshold = timedelta(hours=48)
    
    stale_channels = []
    fresh_channels = []
    
    for ch in channels:
        channel_id, name, status, last_crawl, next_crawl, freq = ch
        
        if last_crawl:
            age = now - last_crawl
            if age > stale_threshold:
                stale_channels.append({
                    'channel_id': channel_id,
                    'name': name,
                    'hours_old': age.total_seconds() / 3600
                })
            else:
                fresh_channels.append(channel_id)
        else:
            stale_channels.append({
                'channel_id': channel_id,
                'name': name,
                'hours_old': None
            })
    
    return {
        'total_channels': len(channels),
        'fresh_channels': len(fresh_channels),
        'stale_channels': len(stale_channels),
        'stale_details': stale_channels,
        'status': 'warning' if len(stale_channels) > 0 else 'ok'
    }


@task(name="Generate Health Report", tags=["reporting", "health"], retries=1)
def generate_health_report_task(pg_health: Dict, bq_health: Dict, api_quota: Dict, data_freshness: Dict) -> Dict:
    all_healthy = (
        pg_health['status'] == 'healthy' and
        bq_health['status'] == 'healthy' and
        api_quota['status'] != 'critical' and
        data_freshness['status'] != 'critical'
    )
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy' if all_healthy else 'degraded',
        'components': {
            'postgresql': pg_health,
            'bigquery': bq_health,
            'api_quota': api_quota,
            'data_freshness': data_freshness
        },
        'summary': {
            'total_channels': data_freshness['total_channels'],
            'stale_channels': data_freshness['stale_channels'],
            'quota_percentage': api_quota['percentage_used']
        }
    }
    
    table_data = [
        ['Component', 'Status', 'Message'],
        ['PostgreSQL', pg_health['status'], pg_health['message']],
        ['BigQuery', bq_health['status'], bq_health['message']],
        ['API Quota', api_quota['status'], api_quota['message']],
        ['Data Freshness', data_freshness['status'], 
         f"{data_freshness['stale_channels']} stale channels"]
    ]
    
    create_table_artifact(
        key="health-report",
        table=table_data,
        description="System Health Check Report"
    )
    
    print(f"\nSystem Health Report")
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"PostgreSQL: {pg_health['status']}")
    print(f"BigQuery: {bq_health['status']}")
    print(f"API Quota: {api_quota['percentage_used']:.1f}% used")
    print(f"Data Freshness: {data_freshness['stale_channels']} stale channels\n")
    
    return report
