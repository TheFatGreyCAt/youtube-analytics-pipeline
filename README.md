# YouTube Analytics Pipeline

A complete data pipeline for analyzing YouTube channels using Google Cloud Platform, dbt, Prefect, and Streamlit. This project helps you automatically collect, transform, and visualize YouTube channel performance data.

## Quick Start for New Users

If you're setting up this project for the first time, you can use the automated setup script that handles everything in one go.

### Prerequisites

Before you begin, make sure you have:
- Python 3.12 or higher installed
- Docker Desktop installed and running
- A YouTube Data API key from Google Cloud Console
- A Google Cloud service account with BigQuery permissions

### One-Click Setup

For Windows users:
```bash
setup.bat
```

For Linux or Mac users:
```bash
chmod +x setup.sh
./setup.sh
```

You can also run the Python script directly:
```bash
python script/setup_all.py
```

Or use Make if you have it installed:
```bash
make setup-all
```

The setup script will automatically:
1. Check if Python and Docker are properly installed
2. Create a .env configuration file from the template
3. Verify your Google Cloud credentials
4. Install all required Python packages
5. Start Docker services (Postgres, Redis, Prefect)
6. Set up database tables and schemas
7. Deploy Prefect workflows with automatic scheduling
8. Optionally help you add sample channels and run a test crawl

After the script completes, you'll need to:
1. Update your API credentials in the .env file
2. Add YouTube channels you want to track
3. Access the Prefect UI at http://localhost:4200

## Manual Setup Guide

If you prefer to set things up step by step, or if the automated setup doesn't work for your environment, follow these instructions.

### Step 1: Environment Configuration

Copy the example environment file and edit it with your credentials:

```bash
cp .env.example .env
```

Open the .env file and update these important values:
- YOUTUBE_API_KEY: Your YouTube Data API key
- GCP_PROJECT_ID: Your Google Cloud project ID
- GOOGLE_APPLICATION_CREDENTIALS: Path to your service account key file
- PG_PASSWORD: A secure password for PostgreSQL

### Step 2: Install Python Dependencies

Create a virtual environment and install packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[all]"
```

The [all] option installs everything you need: extraction tools, orchestration, transformation, and dashboard components.

### Step 3: Start Docker Services

Use Docker Compose to start the required services:

```bash
docker compose up -d
```

This starts PostgreSQL, Redis, and Prefect server. Wait about a minute for all services to be ready.

### Step 4: Initialize Database

Create the necessary database tables:

```bash
python -m extract.cli setup
```

Or using Make:

```bash
make setup
```

### Step 5: Add YouTube Channels

You have several options for adding channels to track:

Option 1 - Using the command line:
```bash
python -m extract.cli add UCXuqSBlHAE6Xw-yeJA0Tunw "Linus Tech Tips"
```

Option 2 - Bulk import from CSV:
```bash
python script/bulk_add_channels.py config/channels_template.csv
```

Option 3 - Edit the config file directly:

Open config/channels.yml and add your channels:

```yaml
channels:
  - id: UCXuqSBlHAE6Xw-yeJA0Tunw
    name: Linus Tech Tips
    frequency_hours: 24
    priority: 1
    active: true
    include_comments: false
```

### Step 6: Deploy Prefect Workflows

Set up automated scheduling with Prefect:

```bash
python script/deploy_prefect.py
```

This deploys three workflows with different schedules:
- daily-youtube-analytics: Runs at 2:00 AM every day (full pipeline)
- extract-3times-daily: Runs at 8:00 AM, 2:00 PM, and 8:00 PM (data collection only)
- dbt-transform-daily: Runs at 8:30 AM, 2:30 PM, and 8:30 PM (data transformation only)

## Understanding YouTube API Quotas

The YouTube Data API has daily quota limits that you need to be aware of:

Daily quota limit: 10,000 units per day

Typical costs per channel:
- Basic channel info: 1 unit
- Recent videos list: 1-2 units per 50 videos
- Playlists: 1 unit (optional)
- Comments: 1 unit per video (only for recent 10 videos)

Total per channel: Usually 5-15 units without comments, up to 25 units with comments enabled

Recommended daily limits:
- Without comments: 15 channels safely
- With comments: 10 channels to stay under quota

Before crawling, always check your current quota usage:

```bash
python script/monitor_quota.py
```

To estimate cost before running:

```bash
python script/monitor_quota.py --estimate 15
```

## Collecting Data from YouTube

### Monitor System Status First

Always check your quota status before crawling:

```bash
python script/monitor_quota.py
```

This shows:
- Current API quota usage
- Available remaining quota
- List of channels and their last crawl time
- System health status

### Running a Crawl

Crawl channels from your config file (recommended method):

```bash
python -m extract.cli crawl-file --limit 10
```

This reads from config/channels.yml and processes up to 10 channels. The system automatically:
- Respects API rate limits
- Retries failed requests
- Logs all activities
- Stops if quota reaches 90 percent

For a specific channel:

```bash
python -m extract.cli crawl --channel UCXuqSBlHAE6Xw-yeJA0Tunw
```

To include comments (uses more quota):

```bash
python -m extract.cli crawl --channel UCXuqSBlHAE6Xw-yeJA0Tunw --with-comments
```

### Managing Channels

List all tracked channels:

```bash
python -m extract.cli channels
```

View crawl history:

```bash
python -m extract.cli history --limit 20
```

## Transforming Data with dbt

After collecting data, use dbt to transform and clean it.

### Check dbt Connection

Verify dbt can connect to BigQuery:

```bash
python script/dbt_cli.py debug
```

### Run Complete Pipeline

This runs all transformation steps including tests:

```bash
python script/dbt_cli.py pipeline
```

Or using Make:

```bash
make dbt-pipeline
```

The pipeline does:
1. Install dbt package dependencies
2. Run all dbt models (staging, intermediate, mart layers)
3. Execute data quality tests
4. Generate documentation

### Run Specific Layers

Run only staging models:

```bash
python script/dbt_cli.py run --select staging.*
```

Run only mart models:

```bash
python script/dbt_cli.py run --select mart.*
```

### Run Tests Only

Execute data quality tests without running models:

```bash
python script/dbt_cli.py test
```

## Automated Scheduling with Docker

The project includes Docker-based Prefect deployment for hands-off operation.

### Deploy Workflows

Start everything with one command:

```bash
python script/deploy_prefect.py
```

Or on Windows:

```bash
script\deploy_prefect.bat
```

On Linux/Mac:

```bash
./script/deploy_prefect.sh
```

Using Make:

```bash
make prefect-deploy
```

### Managing Docker Services

View logs from the Prefect worker:

```bash
docker compose logs -f prefect-worker
```

View logs from the Prefect server:

```bash
docker compose logs -f prefect-server
```

Check service status:

```bash
docker compose ps
```

Restart the worker (after code changes):

```bash
docker compose restart prefect-worker
```

Rebuild the worker image (after dependency changes):

```bash
docker compose up -d --build prefect-worker
```

Stop all services:

```bash
docker compose down
```

Remove all data including volumes:

```bash
docker compose down -v
```

### Customizing Schedules

Edit config/prefect.yaml to change when workflows run:

Run at 4 AM instead of 2 AM:
```yaml
schedule:
  cron: "0 4 * * *"
  timezone: Asia/Ho_Chi_Minh
```

Run every 6 hours:
```yaml
schedule:
  cron: "0 */6 * * *"
  timezone: Asia/Ho_Chi_Minh
```

Run only weekdays:
```yaml
schedule:
  cron: "0 2 * * 1-5"
  timezone: Asia/Ho_Chi_Minh
```

Change crawl limits:
```yaml
parameters:
  crawl_limit: 20
  include_comments: true
  run_dbt_tests: true
```

After editing, redeploy:

```bash
python script/deploy_prefect.py
```

### Checking Deployments

List all deployments:

```bash
docker compose exec prefect-worker prefect deployment ls
```

Run a deployment immediately (don't wait for schedule):

```bash
docker compose exec prefect-worker prefect deployment run youtube-analytics-pipeline/daily-youtube-analytics
```

View recent flow runs:

```bash
docker compose exec prefect-worker prefect flow-run ls --limit 10
```

## Daily Workflow Example

Here's a typical daily workflow for managing 10-15 channels:

Morning - Check quota status:
```bash
python script/monitor_quota.py
```

If quota looks good (under 80 percent used), run a crawl:
```bash
python -m extract.cli crawl-file --limit 15
```

Transform the new data:
```bash
python script/dbt_cli.py pipeline
```

Check the results:
```bash
python script/monitor_quota.py
```

With automated scheduling through Prefect, these steps happen automatically at the times you configure.

## Accessing Services

After starting Docker services, you can access:

Prefect UI: http://localhost:4200
Use this to monitor workflow runs, view logs, and manage schedules.

Streamlit Dashboard: http://localhost:8501
View analytics and visualizations of your YouTube data.

PgAdmin: http://localhost:5050 (if enabled in compose.yml)
Database management interface for PostgreSQL.

## Configuration Files

Understanding the key configuration files:

.env file:
Contains all environment variables including API keys, database credentials, and service settings. This file is gitignored for security.

config/channels.yml:
Lists all YouTube channels to track with their settings like crawl frequency and priority.

config/prefect.yaml:
Defines Prefect workflows and their schedules for automated execution.

compose.yml:
Docker Compose configuration for all services (Postgres, Redis, Prefect server and worker).

pyproject.toml:
Python package dependencies organized by component (extract, orchestrate, serve, dev).

## Useful Commands Reference

Monitoring and status:
```bash
python script/monitor_quota.py                    # Full system status
python script/monitor_quota.py --quota-only       # Just quota info
python script/monitor_quota.py --channels-only    # Just channel list
python script/monitor_quota.py --estimate 15      # Estimate cost
```

Channel management:
```bash
python -m extract.cli channels                    # List all channels
python -m extract.cli history --limit 20          # View history
python -m extract.cli add ID "Name"               # Add channel
```

Data extraction:
```bash
python -m extract.cli crawl-file --limit 10       # Crawl from config
python -m extract.cli crawl --channel ID          # Crawl specific
python -m extract.cli crawl --limit 10            # Crawl scheduled
```

dbt operations:
```bash
python script/dbt_cli.py debug                    # Check connection
python script/dbt_cli.py pipeline                 # Full pipeline
python script/dbt_cli.py run                      # Run models
python script/dbt_cli.py test                     # Run tests
python script/dbt_cli.py run --select staging.*   # Specific layer
```

Docker services:
```bash
make up              # Start all services
make down            # Stop services  
make logs            # View all logs
make clean           # Remove everything
make prefect-deploy  # Deploy workflows
make prefect-logs    # View worker logs
```

## Project Structure

The project is organized into several main directories:

extract/
Contains code for collecting data from YouTube API. Includes the CLI tool, crawler logic, database manager, and schema definitions.

orchestrate/
Prefect workflows and task definitions for automating the pipeline. Contains flows for extraction, transformation, and monitoring.

dbt_project/
All dbt transformation code including models, tests, macros, and seeds. Organized into staging, intermediate, and mart layers.

serve/
Streamlit dashboard application for visualizing the data.

script/
Utility scripts for common tasks like monitoring quotas, deploying Prefect, running dbt, and bulk operations.

config/
Configuration files for channels, Prefect workflows, and application settings.

docker/
Dockerfiles for different components of the system.

## Troubleshooting Common Issues

API quota limit reached:
Check your current usage with the monitor script. If you're near the limit, wait until the quota resets (midnight Pacific Time) or reduce the number of channels you're tracking.

Channel not found error:
Verify the channel ID is correct. YouTube channel IDs start with "UC" and are 24 characters long. Make sure the channel is public and hasn't been deleted.

Database connection error:
Ensure Docker services are running with "docker compose ps". Check that your .env file has correct database credentials matching those in compose.yml.

BigQuery permission denied:
Verify your service account has the BigQuery Admin role in Google Cloud Console. Check that the GOOGLE_APPLICATION_CREDENTIALS path in .env points to a valid key file.

Prefect worker not picking up jobs:
Make sure the worker is running and connected to the right server. Check logs with "docker compose logs prefect-worker". Verify the work pool name matches in both config/prefect.yaml and the worker startup command.

## Performance Tips

For optimal performance with limited quota:
- Crawl during off-peak hours when you're less likely to need quota for testing
- Use a small delay between API calls (0.5 seconds is safe)
- Only enable comments for your most important channels
- Process channels in batches of 10-15 rather than all at once
- Monitor quota regularly to avoid unexpected limits

For better data quality:
- Run dbt tests after every transformation
- Check dbt logs for any warnings or errors
- Review the assert tests in dbt_project/tests directory
- Use the Streamlit dashboard to spot anomalies

## Important Notes

About API quotas:
The system has built-in protection and will stop automatically at 90 percent quota usage. However, you should still monitor usage regularly, especially when adding new channels or enabling comments.

About error handling:
Failed API calls are automatically retried twice. If a channel still fails, it's logged and you can try again later. Check the logs directory for detailed error information.

About data freshness:
Raw data is stored in PostgreSQL and BigQuery. The dbt models transform this into analytics tables. For the freshest data, run the full pipeline after each crawl.

About Docker volumes:
Database data persists in Docker volumes. Use "docker compose down -v" to completely reset the database, but be aware this deletes all collected data.

About credentials:
Never commit your .env file or service account keys to version control. The .gitignore file is set up to exclude these, but always double-check before pushing code.

## Getting Help

If you run into issues:

Check the logs in the logs/ directory for detailed error messages.

Use the help flag on any command to see available options:
```bash
python -m extract.cli --help
python script/dbt_cli.py --help
python script/monitor_quota.py --help
```

Review the documentation in the docs/ directory for more detailed information about specific components.

Check Docker service logs if something isn't working:
```bash
docker compose logs service-name
```

## License

This project is licensed under the MIT License.