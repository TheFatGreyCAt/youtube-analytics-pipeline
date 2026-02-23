from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
import subprocess
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent
dbt_project_path = project_root / "dbt_project"

@task(name="dbt-run-layer", retries=2, retry_delay_seconds=60)
def dbt_run_layer(layer: str):
    logger.info(f"Running dbt layer: {layer}")
    result = subprocess.run(
        ["dbt", "run", "--select", layer],
        cwd=str(dbt_project_path),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(f"dbt {layer} failed: {result.stderr}")
    logger.info(f"{layer} layer completed")
    return result.stdout

@task(name="dbt-test", retries=1)
def dbt_test():
    logger.info("Running dbt tests")
    result = subprocess.run(
        ["dbt", "test"],
        cwd=str(dbt_project_path),
        capture_output=True,
        text=True
    )
    if "ERROR" in result.stdout or result.returncode != 0:
        create_markdown_artifact(
            key="dbt-test-failures",
            markdown=f"# dbt Tests Failed\n\n```\n{result.stdout}\n```",
            description=f"Failed tests at {datetime.now()}"
        )
        raise Exception(f"Some dbt tests failed: {result.stderr}")
    create_markdown_artifact(
        key="dbt-test-success",
        markdown=f"# All dbt Tests Passed\n\nRun at: {datetime.now()}",
        description="Successful dbt test run"
    )
    logger.info("All tests passed")
    return result.stdout

@task(name="dbt-source-freshness")
def check_source_freshness():
    logger.info("Checking source freshness")
    result = subprocess.run(
        ["dbt", "source", "freshness"],
        cwd=str(dbt_project_path),
        capture_output=True,
        text=True
    )
    if "ERROR" in result.stdout:
        logger.warning("Some sources are stale")
    return result.stdout

@task(name="dbt-docs-generate")
def generate_docs():
    logger.info("Generating dbt documentation")
    result = subprocess.run(
        ["dbt", "docs", "generate"],
        cwd=str(dbt_project_path),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.warning(f"Docs generation warning: {result.stderr}")
    logger.info("Documentation generated")
    return result.stdout

@flow(name="dbt-transformation-flow", log_prints=True)
def dbt_transformation_flow(run_tests: bool = True):
    check_source_freshness()
    dbt_run_layer("staging")
    dbt_run_layer("intermediate")
    dbt_run_layer("mart")
    
    if run_tests:
        dbt_test()
    
    generate_docs()

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "layers_executed": ["staging", "intermediate", "mart"],
        "tests_run": run_tests
    }

if __name__ == "__main__":
    dbt_transformation_flow()
