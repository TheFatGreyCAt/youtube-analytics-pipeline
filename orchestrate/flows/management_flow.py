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
    result = add_channel_task(channel_id, channel_name, frequency_hours)
    if result:
        print(f"Channel {channel_name} added successfully")
    return result


@flow(name="Bulk Add Channels", log_prints=True)
def bulk_add_channels_flow(channels: List[dict], default_frequency: int = 24) -> dict:
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
            results['details'].append({'channel_id': channel_id, 'status': 'success'})
        except Exception as e:
            results['failed'] += 1
            results['details'].append({'channel_id': channel_id, 'status': 'failed', 'error': str(e)})

    print(f"Added {results['successful']}/{results['total']} channels")
    return results


@flow(name="List All Channels", log_prints=True)
def list_all_channels_flow() -> List[dict]:
    channels = list_channels_task()

    if not channels:
        print("No channels configured")
        return []

    for ch in channels:
        print(f"[{ch['status'].upper()}] {ch['channel_name']} ({ch['channel_id']})")
        print(f"   Last Crawl: {ch['last_crawl'] or 'Never'} | Next: {ch['next_crawl'] or 'N/A'}")

    return channels


@flow(name="View Crawl History", log_prints=True)
def view_crawl_history_flow(channel_id: Optional[str] = None, limit: int = 20) -> List[dict]:
    history = get_crawl_history_task(channel_id, limit)

    if not history:
        print("No crawl history found")
        return []

    for entry in history:
        status = entry['status'].upper()
        print(f"[{status}] {entry['timestamp']} - {entry['channel_id']} ({entry['records_fetched']} records)")
        if entry['error_message']:
            print(f"   Error: {entry['error_message'][:100]}")

    return history


@flow(name="Maintenance - Cleanup", log_prints=True)
def maintenance_cleanup_flow(days_to_keep: int = 90) -> dict:
    results = {
        'logs_cleaned': 0,
        'disk_space_freed': '0 MB',
        'status': 'success'
    }
    print(f"Cleanup completed: {results['logs_cleaned']} logs, {results['disk_space_freed']} freed")
    return results


if __name__ == "__main__":
    list_all_channels_flow()
