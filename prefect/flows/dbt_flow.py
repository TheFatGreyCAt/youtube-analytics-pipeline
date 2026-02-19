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
    """Run specific dbt layer (staging/intermediate/mart)"""
    logger.info(f"Running dbt layer: {layer}")
    
    result = subprocess.run(
        ["dbt", "run", "--select", layer],
        cwd="dbt_project",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"dbt {layer} failed: {result.stderr}")
    
    logger.info(f"✓ {layer} layer completed")
    return result.stdout

@task(name="dbt-test", retries=1)
def dbt_test():
    """Run all dbt tests"""
    logger.info("Running dbt tests")
    
    result = subprocess.run(
        ["dbt", "test"],
        cwd="dbt_project",
        capture_output=True,
        text=True
    )
    
    if "ERROR" in result.stdout or result.returncode != 0:
        create_markdown_artifact(
            key="dbt-test-failures",
            markdown=f"# ❌ dbt Tests Failed\n\n```\n{result.stdout}\n```",
            description=f"Failed tests at {datetime.now()}"
        )
        raise Exception(f"Some dbt tests failed: {result.stderr}")
    
    create_markdown_artifact(
        key="dbt-test-success",
        markdown=f"# ✅ All dbt Tests Passed\n\nRun at: {datetime.now()}",
        description="Successful dbt test run"
    )
    
    logger.info("✓ All tests passed")
    return result.stdout

@task(name="dbt-source-freshness")
def check_source_freshness():
    """Check data freshness from sources"""
    logger.info("Checking source freshness")
    
    result = subprocess.run(
        ["dbt", "source", "freshness"],
        cwd="dbt_project",
        capture_output=True,
        text=True
    )
    
    if "ERROR" in result.stdout:
        logger.warning("⚠️ Some sources are stale")
    
    return result.stdout

@task(name="dbt-docs-generate")
def generate_docs():
    """Generate dbt documentation"""
    logger.info("Generating dbt documentation")
    
    result = subprocess.run(
        ["dbt", "docs", "generate"],
        cwd="dbt_project",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.warning(f"Docs generation warning: {result.stderr}")
    
    logger.info("✓ Documentation generated")
    return result.stdout

@flow(name="dbt-transformation-flow", log_prints=True)
def dbt_transformation_flow():
    """
    Complete dbt transformation pipeline
    
    Layers:
    1. Staging: Flatten JSON to relational
    2. Intermediate: Business logic & enrichment
    3. Mart: Analytics-ready tables
    """
    
    print("=" * 60)
    print("DBT TRANSFORMATION PIPELINE STARTED")
    print("=" * 60)
    
    print("\n🔍 Step 1: Check source freshness")
    freshness = check_source_freshness()
    
    print("\n📊 Step 2: Run staging layer")
    dbt_run_layer("staging")
    
    print("\n🔄 Step 3: Run intermediate layer")
    dbt_run_layer("intermediate")
    
    print("\n🎯 Step 4: Run mart layer")
    dbt_run_layer("mart")
    
    print("\n✅ Step 5: Run dbt tests")
    dbt_test()
    
    print("\n📚 Step 6: Generate documentation")
    generate_docs()
    
    print("\n" + "=" * 60)
    print("✨ DBT TRANSFORMATION PIPELINE COMPLETED")
    print("=" * 60)
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "layers_executed": ["staging", "intermediate", "mart"]
    }

if __name__ == "__main__":
    dbt_transformation_flow()
