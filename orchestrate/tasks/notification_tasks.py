from prefect import task
from typing import Dict
import logging

logger = logging.getLogger(__name__)


@task(name="Send Email Notification", tags=["notification", "email"], retries=2)
def send_email_notification_task(subject: str, body: str, recipients: list = None) -> bool:
    logger.info(f"Email notification: {subject}")
    return True


@task(name="Send Slack Notification", tags=["notification", "slack"], retries=2)
def send_slack_notification_task(message: str, channel: str = "#alerts", webhook_url: str = None) -> bool:
    logger.info(f"Slack notification to {channel}: {message}")
    return True


@task(name="Alert on Failure", tags=["notification", "alert"], retries=1)
def alert_on_failure_task(crawl_results: Dict) -> bool:
    total = crawl_results['total']
    failed = crawl_results['failed']

    if total == 0:
        return False

    failure_rate = (failed / total * 100)

    if failure_rate > 50:
        failed_channels = [r['channel_id'] for r in crawl_results['channels'] if r['status'] == 'failed']

        subject = f"High Failure Rate: {failure_rate:.1f}%"
        body = f"""YouTube Crawl Alert - High Failure Rate

Summary:
- Total Channels: {total}
- Failed: {failed}
- Failure Rate: {failure_rate:.1f}%

Failed Channels:
{chr(10).join(f"- {ch}" for ch in failed_channels)}
        """
        send_email_notification_task(subject, body)
        send_slack_notification_task(f"YouTube Crawl Alert: {failed}/{total} channels failed ({failure_rate:.1f}%)")
        return True

    return False


@task(name="Success Summary", tags=["notification", "summary"], retries=1)
def send_success_summary_task(crawl_results: Dict) -> bool:
    total = crawl_results['total']
    successful = crawl_results['successful']
    failed = crawl_results['failed']

    if total == 0:
        return False

    success_rate = (successful / total * 100) if total > 0 else 0
    logger.info(f"Crawl summary: {successful}/{total} successful ({success_rate:.1f}%)")

    return True
