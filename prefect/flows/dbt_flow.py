from prefect import flow, task
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
dbt_project_path = project_root / "dbt_project"


@task(name="Run dbt", tags=["dbt", "transform"], retries=2, retry_delay_seconds=30)
def run_dbt_task(command: str = "run") -> dict:
    try:
        result = subprocess.run(
            ["dbt", command, "--project-dir", str(dbt_project_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        return {
            'status': 'success',
            'command': command,
            'output': result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        print(f"dbt {command} failed:")
        print(e.stderr)
        return {
            'status': 'failed',
            'command': command,
            'error': e.stderr
        }


@task(name="Run dbt Test", tags=["dbt", "test"], retries=1)
def run_dbt_test_task() -> dict:
    return run_dbt_task("test")


@flow(name="dbt Transform Flow", log_prints=True)
def dbt_transform_flow(run_tests: bool = True) -> dict:
    print("\nRunning dbt transformation...")
    
    results = {
        'models': None,
        'tests': None,
        'overall_status': 'success'
    }
    
    print("\nExecuting dbt models...")
    models_result = run_dbt_task("run")
    results['models'] = models_result
    
    if models_result['status'] == 'failed':
        print("dbt models failed")
        results['overall_status'] = 'failed'
        return results
    
    if run_tests:
        print("\nExecuting dbt tests...")
        tests_result = run_dbt_test_task()
        results['tests'] = tests_result
        
        if tests_result['status'] == 'failed':
            print("dbt tests failed")
            results['overall_status'] = 'partial_failure'
    
    print("\ndbt transformation completed")
    return results


if __name__ == "__main__":
    dbt_transform_flow(run_tests=True)
