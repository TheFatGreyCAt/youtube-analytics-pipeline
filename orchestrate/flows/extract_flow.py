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
    if not skip_validation:
        validate_config_task()

    channels = get_channels_to_crawl_task(limit=limit)

    if not channels:
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'channels': [],
            'message': 'No channels scheduled for crawling'
        }

    crawl_results = crawl_channels_batch_task(
        channel_ids=channels,
        include_comments=include_comments,
        delay_seconds=2
    )

    create_crawl_report_task(crawl_results)
    quality_results = validate_data_quality_task(crawl_results)
    alert_on_failure_task(crawl_results)
    send_success_summary_task(crawl_results)

    return {
        **crawl_results,
        'quality_checks': quality_results
    }


@flow(name="Single Channel Extract", log_prints=True)
def single_channel_extract_flow(channel_id: str, include_comments: bool = False) -> dict:
    from ..tasks import crawl_channel_task

    validate_config_task()
    result = crawl_channel_task(channel_id, include_comments)

    return result


if __name__ == "__main__":
    youtube_extract_flow(limit=5, include_comments=False)
