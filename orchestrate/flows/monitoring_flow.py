from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

from ..tasks import (
    check_postgres_health_task,
    check_bigquery_health_task,
    check_api_quota_task,
    check_data_freshness_task,
    generate_health_report_task,
    send_email_notification_task
)


@flow(name="System Health Check", task_runner=ConcurrentTaskRunner(), log_prints=True)
def system_health_check_flow() -> dict:
    pg_health = check_postgres_health_task()
    bq_health = check_bigquery_health_task()
    api_quota = check_api_quota_task()
    data_freshness = check_data_freshness_task()

    report = generate_health_report_task(
        pg_health=pg_health,
        bq_health=bq_health,
        api_quota=api_quota,
        data_freshness=data_freshness
    )

    if report['overall_status'] != 'healthy':
        subject = "System Health Alert"
        body = f"""System Health Check Alert

Overall Status: {report['overall_status'].upper()}

Component Status:
- PostgreSQL: {pg_health['status']}
- BigQuery: {bq_health['status']}
- API Quota: {api_quota['status']} ({api_quota['percentage_used']:.1f}% used)
- Data Freshness: {data_freshness['stale_channels']} stale channels
        """
        send_email_notification_task(subject, body)

    return report


@flow(name="Data Freshness Monitor", log_prints=True)
def data_freshness_monitor_flow(alert_threshold_hours: int = 48) -> dict:
    freshness = check_data_freshness_task()

    if freshness['stale_channels'] > 0:
        stale_list = "\n".join([
            f"- {ch['channel_id']} ({ch['name']}): {ch['hours_old']:.1f}h old" if ch['hours_old']
            else f"- {ch['channel_id']} ({ch['name']}): Never crawled"
            for ch in freshness['stale_details'][:10]
        ])

        subject = f"Stale Data Alert: {freshness['stale_channels']} channels"
        body = f"""Data Freshness Alert

Stale Channels: {freshness['stale_channels']} / {freshness['total_channels']}
Threshold: {alert_threshold_hours} hours

Top Stale Channels:
{stale_list}
        """
        send_email_notification_task(subject, body)

    return freshness


@flow(name="API Quota Monitor", log_prints=True)
def api_quota_monitor_flow(warning_threshold: float = 80.0) -> dict:
    quota = check_api_quota_task()

    if quota['percentage_used'] >= warning_threshold:
        subject = f"API Quota Warning: {quota['percentage_used']:.1f}%"
        body = f"""YouTube API Quota Warning

Current Usage: {quota['estimated_quota_used']:,} / {quota['daily_limit']:,}
Percentage: {quota['percentage_used']:.1f}%
Status: {quota['status'].upper()}
        """
        send_email_notification_task(subject, body)

    return quota


if __name__ == "__main__":
    system_health_check_flow()
