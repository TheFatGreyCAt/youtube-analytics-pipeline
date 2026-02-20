#!/usr/bin/env python3
"""
DBT CLI Wrapper for YouTube Analytics Pipeline
Loads .env from root and executes dbt commands in dbt_project directory
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv


# Project paths
ROOT_DIR = Path(__file__).parent
DBT_PROJECT_DIR = ROOT_DIR / "dbt_project"
ENV_FILE = ROOT_DIR / ".env"


def load_environment():
    """Load environment variables from .env file"""
    if not ENV_FILE.exists():
        print(f"❌ Error: .env file not found at {ENV_FILE}")
        sys.exit(1)
    
    load_dotenv(ENV_FILE)
    print(f"✅ Loaded environment from {ENV_FILE}")
    
    # Validate required variables
    required_vars = ["GCP_PROJECT_ID", "BQ_DATASET_ID", "GOOGLE_APPLICATION_CREDENTIALS"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    print(f"✅ GCP Project: {os.getenv('GCP_PROJECT_ID')}")
    print(f"✅ Dataset: {os.getenv('BQ_DATASET_ID')}")
    print()


def run_dbt_command(command: list, cwd: Path = DBT_PROJECT_DIR):
    """Execute dbt command in the dbt_project directory"""
    print(f"📂 Working directory: {cwd}")
    print(f"🚀 Running: dbt {' '.join(command)}\n")
    
    try:
        result = subprocess.run(
            ["dbt"] + command,
            cwd=cwd,
            check=False,
            env=os.environ.copy()
        )
        return result.returncode
    except FileNotFoundError:
        print("❌ Error: dbt command not found. Please install dbt-bigquery.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="DBT CLI Wrapper for YouTube Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s debug                    # Check connection
  %(prog)s run                      # Run all models
  %(prog)s run --select staging.*   # Run only staging models
  %(prog)s test                     # Run all tests
  %(prog)s build                    # Run + test in one command
  %(prog)s full-refresh             # Full refresh pipeline
        """
    )
    
    parser.add_argument(
        "command",
        choices=["debug", "run", "test", "build", "deps", "compile", "parse", "clean", "full-refresh", "pipeline"],
        help="DBT command to execute"
    )
    
    parser.add_argument(
        "--select", "-s",
        help="Select specific models (e.g., staging.*, mart.*)"
    )
    
    parser.add_argument(
        "--target", "-t",
        default="dev",
        choices=["dev", "prod"],
        help="Target environment (default: dev)"
    )
    
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Full refresh for incremental models"
    )
    
    parser.add_argument(
        "--vars",
        help="Override dbt variables (e.g., '{\"key\": \"value\"}')"
    )
    
    args, unknown = parser.parse_known_args()
    
    # Load environment
    load_environment()
    
    # Build dbt command
    if args.command == "pipeline":
        # Run full pipeline: deps -> run -> test
        print("=" * 60)
        print("🚀 RUNNING FULL DBT PIPELINE")
        print("=" * 60)
        
        commands = [
            (["deps"], "Installing dependencies"),
            (["run", "--target", args.target], "Running models"),
            (["test", "--target", args.target], "Running tests"),
        ]
        
        for cmd, desc in commands:
            print(f"\n{'=' * 60}")
            print(f"📌 {desc}")
            print(f"{'=' * 60}\n")
            
            returncode = run_dbt_command(cmd)
            if returncode != 0:
                print(f"\n❌ Pipeline failed at: {desc}")
                sys.exit(returncode)
        
        print(f"\n{'=' * 60}")
        print("✅ DBT PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'=' * 60}")
        sys.exit(0)
    
    elif args.command == "full-refresh":
        # Run full refresh
        cmd = ["run", "--full-refresh", "--target", args.target]
    else:
        # Single command
        cmd = [args.command, "--target", args.target]
    
    # Add select option
    if args.select:
        cmd.extend(["--select", args.select])
    
    # Add full-refresh flag
    if args.full_refresh and args.command == "run":
        cmd.append("--full-refresh")
    
    # Add vars
    if args.vars:
        cmd.extend(["--vars", args.vars])
    
    # Add any unknown arguments (pass-through)
    cmd.extend(unknown)
    
    # Execute
    returncode = run_dbt_command(cmd)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
