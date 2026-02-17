from prefect import flow

from .flows.extract_flow import youtube_extract_flow
from .flows.dbt_flow import dbt_transform_flow
from .flows.monitoring_flow import system_health_check_flow


@flow(name="YouTube Analytics Pipeline", log_prints=True)
def youtube_analytics_pipeline(
    crawl_limit: int = 10,
    include_comments: bool = False,
    run_dbt_tests: bool = True,
    health_check_first: bool = True
):
    print("\nYouTube Analytics Pipeline - Starting\n")
    
    results = {
        'health_check': None,
        'extract': None,
        'transform': None,
        'overall_status': 'success'
    }
    
    if health_check_first:
        print("Step 0: System Health Check...")
        try:
            health_report = system_health_check_flow()
            results['health_check'] = health_report
            
            if health_report['overall_status'] != 'healthy':
                print("Warning: System health issues detected, but continuing...")
        except Exception as e:
            print(f"Health check failed: {e}, but continuing pipeline...")
            results['health_check'] = {'status': 'failed', 'error': str(e)}
    
    print("\nStep 1: Extracting YouTube data...")
    try:
        extract_result = youtube_extract_flow(
            limit=crawl_limit,
            include_comments=include_comments
        )
        results['extract'] = extract_result
        
        if extract_result.get('successful', 0) == 0:
            print("No data extracted successfully. Skipping transformation.")
            results['overall_status'] = 'partial_failure'
            return results
            
    except Exception as e:
        print(f"Extract step failed: {e}")
        results['extract'] = {'status': 'failed', 'error': str(e)}
        results['overall_status'] = 'failed'
        return results
    
    if extract_result.get('successful', 0) > 0:
        print("\nStep 2: Transforming data with dbt...")
        try:
            transform_result = dbt_transform_flow(run_tests=run_dbt_tests)
            results['transform'] = transform_result
        except Exception as e:
            print(f"Transform step failed: {e}")
            results['transform'] = {'status': 'failed', 'error': str(e)}
            results['overall_status'] = 'partial_failure'
    else:
        print("Skipping transformation - no successful extracts")
    
    print("\nYouTube Analytics Pipeline - Completed")
    print(f"Overall Status: {results['overall_status'].upper()}")
    if results['extract']:
        print(f"Channels Crawled: {results['extract'].get('successful', 0)}/{results['extract'].get('total', 0)}")
    print()
    
    return results


@flow(name="YouTube Analytics - Extract Only", log_prints=True)
def extract_only_pipeline(crawl_limit: int = 10, include_comments: bool = False):
    print("\nExtract Only Pipeline\n")
    
    result = youtube_extract_flow(
        limit=crawl_limit,
        include_comments=include_comments
    )
    
    return result


@flow(name="YouTube Analytics - Transform Only", log_prints=True)
def transform_only_pipeline(run_tests: bool = True):
    print("\nTransform Only Pipeline\n")
    
    result = dbt_transform_flow(run_tests=run_tests)
    
    return result


if __name__ == "__main__":
    youtube_analytics_pipeline(
        crawl_limit=10,
        include_comments=False,
        run_dbt_tests=True,
        health_check_first=True
    )
