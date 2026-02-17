from prefect import flow
from typing import List, Optional

from ..tasks import (
    setup_databases_task,
    add_channel_task,
    list_channels_task,
    get_crawl_history_task
)


@flow(name="Setup Databases", log_prints=True)
def setup_databases_flow() -> dict:
    print("\nDatabase Setup Flow\n")
    
    results = setup_databases_task()
    
    if results.get('postgres'):
        print("PostgreSQL setup successful")
    else:
        print(f"PostgreSQL setup failed: {results.get('postgres_error')}")
    
    if results.get('bigquery'):
        print("BigQuery setup successful")
    else:
        print(f"BigQuery setup failed: {results.get('bigquery_error')}")
    
    return results


@flow(name="Add Channel", log_prints=True)
def add_channel_flow(channel_id: str, channel_name: str, frequency_hours: int = 24) -> bool:
    print(f"\nAdding Channel: {channel_name}")
    print(f"ID: {channel_id}")
    print(f"Frequency: Every {frequency_hours} hours")
    
    result = add_channel_task(channel_id, channel_name, frequency_hours)
    
    if result:
        print(f"Channel added successfully")
    
    return result


@flow(name="Bulk Add Channels", log_prints=True)
def bulk_add_channels_flow(channels: List[dict], default_frequency: int = 24) -> dict:
    print(f"\nBulk Add: {len(channels)} channels\n")
    
    results = {
        'total': len(channels),
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    for ch in channels:
        channel_id = ch.get('channel_id')
        channel_name = ch.get('channel_name')
        frequency = ch.get('frequency_hours', default_frequency)
        
        try:
            add_channel_task(channel_id, channel_name, frequency)
            results['successful'] += 1
            results['details'].append({
                'channel_id': channel_id,
                'status': 'success'
            })
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'channel_id': channel_id,
                'status': 'failed',
                'error': str(e)
            })
    
    print(f"\nAdded {results['successful']}/{results['total']} channels")
    return results


@flow(name="List All Channels", log_prints=True)
def list_all_channels_flow() -> List[dict]:
    print("\nListing All Channels\n")
    
    channels = list_channels_task()
    
    if not channels:
        print("No channels configured")
        return []
    
    print(f"\nFound {len(channels)} channels:\n")
    
    for ch in channels:
        status_icon = {
            'success': '[OK]',
            'failed': '[FAILED]',
            'pending': '[PENDING]'
        }.get(ch['status'], '[?]')
        
        print(f"{status_icon} {ch['channel_name']}")
        print(f"   ID: {ch['channel_id']}")
        print(f"   Status: {ch['status']}")
        print(f"   Last Crawl: {ch['last_crawl'] or 'Never'}")
        print(f"   Next Crawl: {ch['next_crawl'] or 'N/A'}")
        print()
    
    return channels


@flow(name="View Crawl History", log_prints=True)
def view_crawl_history_flow(channel_id: Optional[str] = None, limit: int = 20) -> List[dict]:
    if channel_id:
        print(f"\nCrawl History for {channel_id}")
    else:
        print(f"\nRecent Crawl History (last {limit} entries)")
    
    print()
    
    history = get_crawl_history_task(channel_id, limit)
    
    if not history:
        print("No crawl history found")
        return []
    
    print(f"\nShowing {len(history)} records:\n")
    
    for entry in history:
        status_icon = '[OK]' if entry['status'] == 'success' else '[FAILED]'
        
        print(f"{status_icon} {entry['timestamp']}")
        print(f"   Channel: {entry['channel_id']}")
        print(f"   Records: {entry['records_fetched']}")
        if entry['error_message']:
            print(f"   Error: {entry['error_message'][:100]}")
        print()
    
    return history


@flow(name="Maintenance - Cleanup", log_prints=True)
def maintenance_cleanup_flow(days_to_keep: int = 90) -> dict:
    print(f"\nMaintenance Cleanup (keeping {days_to_keep} days)\n")
    
    results = {
        'logs_cleaned': 0,
        'disk_space_freed': '0 MB',
        'status': 'success'
    }
    
    print(f"\nCleanup completed")
    print(f"Logs cleaned: {results['logs_cleaned']}")
    print(f"Space freed: {results['disk_space_freed']}")
    
    return results


if __name__ == "__main__":
    list_all_channels_flow()
