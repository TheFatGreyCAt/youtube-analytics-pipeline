from prefect import flow

from .extract_flow import youtube_extract_flow
from .dbt_flow import dbt_transformation_flow
from .monitoring_flow import system_health_check_flow


@flow(name="YouTube Analytics Pipeline", log_prints=True)
def youtube_analytics_pipeline(
    crawl_limit: int = 10,
    include_comments: bool = False,
    run_dbt_tests: bool = True,
    health_check_first: bool = True
):
    results = {
        'health_check': None,
        'extract': None,
        'transform': None,
        'overall_status': 'success'
    }

    if health_check_first:
        try:
            health_report = system_health_check_flow()
            results['health_check'] = health_report
            if health_report['overall_status'] != 'healthy':
                print("Warning: System health issues detected, but continuing...")
        except Exception as e:
            results['health_check'] = {'status': 'failed', 'error': str(e)}

    try:
        extract_result = youtube_extract_flow(limit=crawl_limit, include_comments=include_comments)
        results['extract'] = extract_result

        if extract_result.get('successful', 0) == 0:
            results['overall_status'] = 'partial_failure'
            return results

    except Exception as e:
        results['extract'] = {'status': 'failed', 'error': str(e)}
        results['overall_status'] = 'failed'
        return results

    if extract_result.get('successful', 0) > 0:
        try:
            transform_result = dbt_transformation_flow(run_tests=run_dbt_tests)
            results['transform'] = transform_result
        except Exception as e:
            results['transform'] = {'status': 'failed', 'error': str(e)}
            results['overall_status'] = 'partial_failure'

    return results


@flow(name="YouTube Analytics - Extract Only", log_prints=True)
def extract_only_pipeline(crawl_limit: int = 10, include_comments: bool = False):
    return youtube_extract_flow(limit=crawl_limit, include_comments=include_comments)


@flow(name="YouTube Analytics - Transform Only", log_prints=True)
def transform_only_pipeline(run_tests: bool = True):
    return dbt_transformation_flow(run_tests=run_tests)


if __name__ == "__main__":
    youtube_analytics_pipeline(
        crawl_limit=10,
        include_comments=False,
        run_dbt_tests=True,
        health_check_first=True
    )
