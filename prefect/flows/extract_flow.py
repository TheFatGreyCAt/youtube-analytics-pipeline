from prefect import flow

from ..tasks import (
    validate_config_task,
    get_channels_to_crawl_task,
    crawl_channels_batch_task,
    create_crawl_report_task,
    validate_data_quality_task,
    alert_on_failure_task,
    send_success_summary_task
)


@flow(name="YouTube Extract Flow", log_prints=True)
def youtube_extract_flow(limit: int = 10, include_comments: bool = False, skip_validation: bool = False) -> dict:
    print("\nYouTube Extract Flow - Starting\n")
    
    if not skip_validation:
        print("Step 1: Validating configuration...")
        validate_config_task()
        print("Configuration validated\n")
    
    print(f"Step 2: Fetching channels to crawl (limit: {limit})...")
    channels = get_channels_to_crawl_task(limit=limit)
    
    if not channels:
        print("No channels to crawl. Flow completed.")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'channels': [],
            'message': 'No channels scheduled for crawling'
        }
    
    print(f"\nStep 3: Crawling {len(channels)} channels...")
    crawl_results = crawl_channels_batch_task(
        channel_ids=channels,
        include_comments=include_comments,
        delay_seconds=2
    )
    
    print("\nStep 4: Generating crawl report...")
    create_crawl_report_task(crawl_results)
    
    print("\nStep 5: Validating data quality...")
    quality_results = validate_data_quality_task(crawl_results)
    
    print("\nStep 6: Sending notifications...")
    alert_on_failure_task(crawl_results)
    send_success_summary_task(crawl_results)
    
    print(f"\nYouTube Extract Flow - Completed")
    print(f"Total: {crawl_results['total']} | Success: {crawl_results['successful']} | Failed: {crawl_results['failed']}\n")
    
    return {
        **crawl_results,
        'quality_checks': quality_results
    }


@flow(name="Single Channel Extract", log_prints=True)
def single_channel_extract_flow(channel_id: str, include_comments: bool = False) -> dict:
    from ..tasks import crawl_channel_task
    
    print(f"\nSingle Channel Extract: {channel_id}\n")
    
    validate_config_task()
    result = crawl_channel_task(channel_id, include_comments)
    
    if result['status'] == 'success':
        print(f"\nSuccess: {result['records_fetched']} records extracted")
    else:
        print(f"\nFailed: {result['error']}")
    
    return result


if __name__ == "__main__":
    youtube_extract_flow(limit=5, include_comments=False)
