# YouTube Analytics Pipeline

Data pipeline for YouTube channel analytics using GCP, dbt, Prefect, and Streamlit.

## Project Structure

```
youtube-analytics-pipeline/
├── extract/              # Data extraction from YouTube API
├── prefect/              # Workflow orchestration
├── dbt_project/          # Data transformation
├── streamlit/            # Dashboard
├── docker/               # Container configs
├── ml/                   # ML models
├── docker-compose.yml    # Full stack
├── Makefile              # Commands
└── .env                  # Config
```

## Quick Start

### 1. Setup

```bash
# Create .env file
cp .env.example .env

# Install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r extract/requirements.txt

# Setup databases
python -m extract.cli setup
```

### 2. Add Channels

```bash
python -m extract.cli add <CHANNEL_ID> "Channel Name"
python -m extract.cli list
```

### 3. Run Pipeline

```bash
# Manual crawl
python -m extract.cli crawl

# Or start full stack
make up
```

Access:
- Prefect UI: http://localhost:4200
- Dashboard: http://localhost:8501

## Commands

```bash
make setup     # Setup databases
make up        # Start services
make down      # Stop services
make crawl     # Run crawl
make list      # List channels
make clean     # Remove containers
```

## CLI Reference

```bash
# Database setup
python -m extract.cli setup

# Channel management
python -m extract.cli add <ID> "<NAME>" [--frequency 24]
python -m extract.cli list
python -m extract.cli remove <ID>

# Crawling
python -m extract.cli crawl [--limit 10] [--channel <ID>] [--with-comments]

# History
python -m extract.cli history [--channel <ID>] [--limit 10]
```

## Environment Variables

```bash
YOUTUBE_API_KEY=your_api_key
GCP_PROJECT_ID=your_project_id
BQ_DATASET_ID=raw_yt
GOOGLE_APPLICATION_CREDENTIALS=credentials/key.json
PG_CONN_STR=postgresql://user:pass@localhost:5432/prefect
```

## Architecture

```
YouTube API → Extract (Python) → BigQuery (Raw)
                ↓                      ↓
          PostgreSQL              dbt Transform
           (Schedule)                  ↓
                              BigQuery (Mart) → Streamlit
```

## Development

```bash
# dbt
cd dbt_project
dbt run

# Tests
pytest extract/tests/
dbt test
```

## License

MIT