from prefect import task
from prefect.artifacts import create_markdown_artifact
from typing import List, Dict, Optional
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from extract.db_manager import PostgresManager, BigQueryManager
from extract.crawlers import YouTubeCrawler
from extract.config import validate_config


@task(name="Validate Configuration", tags=["setup", "validation"], retries=1)
def validate_config_task() -> bool:
    try:
        validate_config()
        return True
    except ValueError as e:
        raise Exception(f"Configuration validation failed: {e}")


@task(name="Setup Databases", tags=["setup", "database"], retries=2, retry_delay_seconds=30)
def setup_databases_task() -> Dict[str, bool]:
    results = {}
    
    try:
        pg_manager = PostgresManager()
        pg_manager.setup_tables()
        results['postgres'] = True
    except Exception as e:
        results['postgres'] = False
        results['postgres_error'] = str(e)
    
    try:
        bq_manager = BigQueryManager()
        bq_manager.setup_tables()
        results['bigquery'] = True
    except Exception as e:
        results['bigquery'] = False
        results['bigquery_error'] = str(e)
    
    return results


@task(name="Get Channels to Crawl", tags=["extract", "scheduling"], retries=3, cache_key_fn=lambda *args, **kwargs: None)
def get_channels_to_crawl_task(limit: int = 10) -> List[str]:
    pg_manager = PostgresManager()
    channels = pg_manager.get_channels_to_crawl(limit=limit)
    
    if not channels:
        print("No channels scheduled for crawling")
        return []
    
    print(f"Found {len(channels)} channels to crawl: {', '.join(channels)}")
    return channels


@task(name="Crawl Channel", tags=["extract", "crawl"], retries=2, retry_delay_seconds=[60, 300])
def crawl_channel_task(channel_id: str, include_comments: bool = False) -> Dict:
    crawler = YouTubeCrawler()
    
    try:
        records_count = crawler.crawl_channel_full(
            channel_id=channel_id,
            include_comments=include_comments
        )
        
        result = {
            'channel_id': channel_id,
            'status': 'success',
            'records_fetched': records_count,
            'error': None
        }
        
        print(f"Channel {channel_id}: {records_count} records")
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"Channel {channel_id} failed: {error_msg}")
        
        return {
            'channel_id': channel_id,
            'status': 'failed',
            'records_fetched': 0,
            'error': error_msg
        }


@task(name="Crawl Multiple Channels", tags=["extract", "batch"], retries=1)
def crawl_channels_batch_task(channel_ids: List[str], include_comments: bool = False, delay_seconds: int = 2) -> Dict:
    import time
    
    results = {
        'total': len(channel_ids),
        'successful': 0,
        'failed': 0,
        'channels': []
    }
    
    for i, channel_id in enumerate(channel_ids, 1):
        print(f"\n[{i}/{len(channel_ids)}] Crawling {channel_id}...")
        
        result = crawl_channel_task(channel_id, include_comments)
        results['channels'].append(result)
        
        if result['status'] == 'success':
            results['successful'] += 1
        else:
            results['failed'] += 1
        
        if i < len(channel_ids):
            time.sleep(delay_seconds)
    
    return results


@task(name="Add Channel Config", tags=["management", "config"], retries=2)
def add_channel_task(channel_id: str, channel_name: str, frequency_hours: int = 24) -> bool:
    pg_manager = PostgresManager()
    pg_manager.add_channel(channel_id, channel_name, frequency_hours)
    print(f"Added/Updated channel: {channel_name} ({channel_id})")
    return True


@task(name="List Channels", tags=["management", "reporting"], retries=2)
def list_channels_task() -> List[Dict]:
    pg_manager = PostgresManager()
    channels = pg_manager.list_channels()
    
    channel_list = []
    for ch in channels:
        channel_list.append({
            'channel_id': ch[0],
            'channel_name': ch[1],
            'status': ch[2],
            'last_crawl': ch[3].isoformat() if ch[3] else None,
            'next_crawl': ch[4].isoformat() if ch[4] else None,
            'frequency_hours': ch[5]
        })
    
    return channel_list


@task(name="Get Crawl History", tags=["monitoring", "reporting"], retries=2)
def get_crawl_history_task(channel_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    pg_manager = PostgresManager()
    logs = pg_manager.get_crawl_history(channel_id, limit)
    
    history = []
    for log in logs:
        history.append({
            'channel_id': log[0],
            'timestamp': log[1].isoformat(),
            'records_fetched': log[2],
            'status': log[3],
            'error_message': log[4]
        })
    
    return history


@task(name="Create Crawl Report", tags=["reporting", "artifact"], retries=1)
def create_crawl_report_task(crawl_results: Dict) -> str:
    total = crawl_results['total']
    successful = crawl_results['successful']
    failed = crawl_results['failed']
    success_rate = (successful / total * 100) if total > 0 else 0
    
    markdown = f"""# YouTube Crawl Report

## Summary
- Total Channels: {total}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {success_rate:.1f}%

## Channel Details

| Channel ID | Status | Records | Error |
|------------|--------|---------|-------|
"""
    
    for result in crawl_results['channels']:
        status = "SUCCESS" if result['status'] == 'success' else "FAILED"
        error = result['error'][:50] + "..." if result['error'] and len(result['error']) > 50 else (result['error'] or "-")
        
        markdown += f"| {result['channel_id']} | {status} | {result['records_fetched']} | {error} |\n"
    
    create_markdown_artifact(
        key="crawl-report",
        markdown=markdown,
        description=f"Crawl results: {successful}/{total} successful"
    )
    
    return markdown


@task(name="Validate Data Quality", tags=["validation", "quality"], retries=1)
def validate_data_quality_task(crawl_results: Dict) -> Dict:
    quality_checks = {
        'passed': True,
        'checks': []
    }
    
    total = crawl_results['total']
    successful = crawl_results['successful']
    success_rate = (successful / total * 100) if total > 0 else 0
    
    if success_rate < 50:
        quality_checks['passed'] = False
        quality_checks['checks'].append({
            'name': 'Success Rate',
            'status': 'FAILED',
            'message': f'Success rate {success_rate:.1f}% is below 50% threshold'
        })
    else:
        quality_checks['checks'].append({
            'name': 'Success Rate',
            'status': 'PASSED',
            'message': f'Success rate {success_rate:.1f}% meets threshold'
        })
    
    if successful == 0:
        quality_checks['passed'] = False
        quality_checks['checks'].append({
            'name': 'Minimum Records',
            'status': 'FAILED',
            'message': 'No successful crawls'
        })
    else:
        quality_checks['checks'].append({
            'name': 'Minimum Records',
            'status': 'PASSED',
            'message': f'{successful} channels crawled successfully'
        })
    
    for result in crawl_results['channels']:
        if result['status'] == 'success' and result['records_fetched'] == 0:
            quality_checks['passed'] = False
            quality_checks['checks'].append({
                'name': f"Channel {result['channel_id']}",
                'status': 'WARNING',
                'message': 'Marked success but 0 records fetched'
            })
    
    if quality_checks['passed']:
        print("All data quality checks passed")
    else:
        print("Some data quality checks failed")
    
    return quality_checks
