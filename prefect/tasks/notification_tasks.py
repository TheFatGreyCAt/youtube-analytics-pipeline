"""
Notification Tasks - Send alerts and notifications
"""
from prefect import task
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@task(name="Send Email Notification", tags=["notification", "email"], retries=2)
def send_email_notification_task(subject: str, body: str, recipients: Optional[list] = None) -> bool:
    print(f"\nEMAIL NOTIFICATION")
    print(f"To: {recipients or ['admin@example.com']}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}\n")
    
    logger.info(f"Email notification logged: {subject}")
    return True


@task(name="Send Slack Notification", tags=["notification", "slack"], retries=2)
def send_slack_notification_task(message: str, channel: str = "#alerts", webhook_url: Optional[str] = None) -> bool:
    print(f"\nSLACK NOTIFICATION")
    print(f"Channel: {channel}")
    print(f"Message: {message}\n")
    
    logger.info(f"Slack notification logged to {channel}")
    return True


@task(name="Alert on Failure", tags=["notification", "alert"], retries=1)
def alert_on_failure_task(crawl_results: Dict) -> bool:
    total = crawl_results['total']
    failed = crawl_results['failed']
    
    if total == 0:
        return False
    
    failure_rate = (failed / total * 100)
    
    if failure_rate > 50:
        subject = f"High Failure Rate: {failure_rate:.1f}%"
        
        failed_channels = [
            r['channel_id'] for r in crawl_results['channels'] 
            if r['status'] == 'failed'
        ]
        
        body = f"""
YouTube Crawl Alert - High Failure Rate

Summary:
- Total Channels: {total}
- Failed: {failed}
- Failure Rate: {failure_rate:.1f}%

Failed Channels:
{chr(10).join(f"- {ch}" for ch in failed_channels)}

Please investigate the issues.
        """
        
        send_email_notification_task(subject, body)
        
        slack_msg = f"YouTube Crawl Alert: {failed}/{total} channels failed ({failure_rate:.1f}%)"
        send_slack_notification_task(slack_msg)
        
        return True
    
    return False


@task(name="Success Summary", tags=["notification", "summary"], retries=1)
def send_success_summary_task(crawl_results: Dict) -> bool:
    total = crawl_results['total']
    successful = crawl_results['successful']
    failed = crawl_results['failed']
    success_rate = (successful / total * 100) if total > 0 else 0
    
    if total == 0:
        return False
    
    message = f"""
YouTube Crawl Completed

Summary:
- Total Channels: {total}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {success_rate:.1f}%
"""
    
    if failed > 0:
        failed_channels = [
            r['channel_id'] for r in crawl_results['channels'] 
            if r['status'] == 'failed'
        ]
        message += f"\nFailed Channels:\n"
        message += "\n".join(f"- {ch}" for ch in failed_channels)
    
    print(message)
    logger.info(f"Crawl summary: {successful}/{total} successful")
    
    return True
