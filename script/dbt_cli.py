#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent.parent
DBT_PROJECT_DIR = ROOT_DIR / "dbt_project"
ENV_FILE = ROOT_DIR / ".env"


def load_environment():
    if not ENV_FILE.exists():
        print(f"Error: .env file not found at {ENV_FILE}")
        sys.exit(1)

    load_dotenv(ENV_FILE)

    required_vars = ["GCP_PROJECT_ID", "BQ_DATASET_ID", "GOOGLE_APPLICATION_CREDENTIALS"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)


def run_dbt_command(command: list, cwd: Path = DBT_PROJECT_DIR):
    try:
        result = subprocess.run(
            ["dbt"] + command,
            cwd=cwd,
            check=False,
            env=os.environ.copy()
        )
        return result.returncode
    except FileNotFoundError:
        print("Error: dbt command not found. Please install dbt-bigquery.")
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
    parser.add_argument("--select", "-s", help="Select specific models (e.g., staging.*, mart.*)")
    parser.add_argument("--target", "-t", default="dev", choices=["dev", "prod"], help="Target environment (default: dev)")
    parser.add_argument("--full-refresh", action="store_true", help="Full refresh for incremental models")
    parser.add_argument("--vars", help="Override dbt variables (e.g., '{\"key\": \"value\"}')")

    args, unknown = parser.parse_known_args()

    load_environment()

    if args.command == "pipeline":
        commands = [
            (["deps"], "Installing dependencies"),
            (["run", "--target", args.target], "Running models"),
            (["test", "--target", args.target], "Running tests"),
        ]

        for cmd, desc in commands:
            print(desc)
            returncode = run_dbt_command(cmd)
            if returncode != 0:
                print(f"Pipeline failed at: {desc}")
                sys.exit(returncode)

        sys.exit(0)

    elif args.command == "full-refresh":
        cmd = ["run", "--full-refresh", "--target", args.target]
    else:
        cmd = [args.command, "--target", args.target]

    if args.select:
        cmd.extend(["--select", args.select])

    if args.full_refresh and args.command == "run":
        cmd.append("--full-refresh")

    if args.vars:
        cmd.extend(["--vars", args.vars])

    cmd.extend(unknown)

    returncode = run_dbt_command(cmd)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
