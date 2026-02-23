# YouTube Analytics Pipeline

Data pipeline for YouTube channel analytics using GCP, dbt, Prefect, and Streamlit.

## 🚀 Quick Start Guide (10-15 Channels)

### 1. Setup Environment

```bash
# Copy .env.example to .env and update values
cp .env.example .env

# Edit .env with your credentials
# - YOUTUBE_API_KEY
# - GCP_PROJECT_ID  
# - GOOGLE_APPLICATION_CREDENTIALS
# - PostgreSQL credentials
```

### 2. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[all]"  # Install all dependencies
```

### 3. Setup Databases

```bash
# Start Docker services
make up

# Setup database schemas
make setup
# Or: python -m extract.cli setup
```

### 4. Add Channels

**Option 1: Edit config/channels.yml**
```yaml
channels:
  - id: UCXuqSBlHAE6Xw-yeJA0Tunw
    name: Linus Tech Tips
    frequency_hours: 24
    priority: 1
    active: true
    include_comments: false
```

**Option 2: Use Bulk Add Script**
```bash
# Edit config/channels_template.csv
python script/bulk_add_channels.py config/channels_template.csv
```

**Option 3: Add via CLI**
```bash
python -m extract.cli add <CHANNEL_ID> "Channel Name"
```

### 5. Check System Status

```bash
# Monitor API quota and channel status
python script/monitor_quota.py

# Estimate quota cost before crawling
python script/monitor_quota.py --estimate 15

# With comments enabled
python script/monitor_quota.py --estimate 15 --with-comments
```

### 6. Run Crawl

```bash
# Crawl 10 channels (recommended first time)
python -m extract.cli crawl-file --limit 10

# Or use Makefile
make crawl
```

### 7. Run dbt Transformation

```bash
# Run full dbt pipeline
python script/dbt_cli.py pipeline

# Or use Makefile
make dbt-pipeline
```

## 📊 Important Limits & Quotas

### YouTube API Quota
- **Daily Limit**: 10,000 units/day
- **Per Channel**: ~5-15 units (without comments)
- **With Comments**: +10 units per channel
- **Safe Daily Limit**: 15 channels without comments, 10 with comments

### Recommended Settings (config/channels.yml)
```yaml
settings:
  max_videos_per_channel: 50      # Don't exceed 100
  max_comments_per_video: 100     # Only for important channels
  batch_size: 10                  # Process 10 channels at a time
  api_delay_seconds: 0.5          # Rate limiting
```

## 🛠️ Useful Commands

### Monitoring
```bash
# Check system status
python script/monitor_quota.py

# View quota only
python script/monitor_quota.py --quota-only

# View channels only  
python script/monitor_quota.py --channels-only

# Estimate cost for N channels
python script/monitor_quota.py --estimate 15
```

### Channel Management
```bash
# List all channels
python -m extract.cli channels

# View crawl history
python -m extract.cli history --limit 20

# Add single channel
python -m extract.cli add <ID> "Name" --frequency 24

# Bulk add from file
python script/bulk_add_channels.py channels.csv
```

### Data Extraction
```bash
# Crawl from config file (RECOMMENDED)
python -m extract.cli crawl-file --limit 10

# Crawl specific channel
python -m extract.cli crawl --channel <ID> --with-comments

# Crawl scheduled channels from DB
python -m extract.cli crawl --limit 10
```

### dbt Operations
```bash
# Check connection
python script/dbt_cli.py debug

# Run full pipeline (deps -> run -> test)
python script/dbt_cli.py pipeline

# Run specific layer
python script/dbt_cli.py run --select staging.*
python script/dbt_cli.py run --select mart.*

# Run tests only
python script/dbt_cli.py test

# Full refresh
python script/dbt_cli.py full-refresh
```

### Docker Services
```bash
# Start all services
make up

# Stop services
make down

# View logs
make logs

# Clean up everything
make clean
```

## 📋 Daily Workflow (10-15 Channels)

```bash
# Morning: Check quota status
python script/monitor_quota.py

# If quota < 20%, crawl channels
python -m extract.cli crawl-file --limit 15

# Transform data with dbt
python script/dbt_cli.py pipeline

# Check results
python script/monitor_quota.py
```

## ⚠️ Important Notes

1. **API Quota Management**
   - Check quota before each crawl
   - Stop at 90% usage automatically
   - Monitor with `python script/monitor_quota.py`

2. **Error Handling**
   - System will retry failed API calls (2 times)
   - Failed channels are logged and can be re-crawled
   - Check logs in `logs/` directory

3. **Performance Tips**
   - Crawl during off-peak hours
   - Use `api_delay_seconds: 0.5` to avoid rate limits
   - Enable comments only for important channels
   - Batch process 10-15 channels at a time

4. **Data Quality**
   - dbt runs data quality tests automatically
   - Check `dbt_project/tests/` for test definitions
   - View test results in dbt logs

## 🎯 Quota Cost Estimation

| Operation | Cost | Notes |
|-----------|------|-------|
| Channel info | 1 unit | Basic metadata |
| Videos list | 1-2 units | Per 50 videos |
| Playlists | 1 unit | Optional |
| Comments | 1 unit/video | Only recent 10 videos |
| **Total per channel** | **5-15 units** | Depends on settings |

**Examples:**
- 10 channels, no comments: ~50-75 units
- 15 channels, no comments: ~75-100 units  
- 10 channels, with comments: ~150 units
- 15 channels, with comments: ~225 units

## 📁 Project Structure

```
youtube-analytics-pipeline/
├── extract/              # Data extraction from YouTube API
├── orchestrate/          # Workflow orchestration (Prefect)
├── dbt_project/          # Data transformation (dbt)
├── serve/                # Dashboard (Streamlit)
├── script/               # Utility scripts
│   ├── dbt_cli.py       # dbt wrapper
│   ├── monitor_quota.py # System monitoring ⭐
│   └── bulk_add_channels.py # Bulk channel import ⭐
├── config/               # Configuration files
│   ├── channels.yml     # Active channels
│   └── channels_template.csv # Template for bulk add
├── compose.yml           # Docker services
├── Makefile              # Command shortcuts
├── pyproject.toml        # Python dependencies
└── .env                  # Environment config
```

## 🔧 Troubleshooting

### "API quota limit reached"
```bash
# Check current quota
python script/monitor_quota.py --quota-only

# Wait until tomorrow or reduce channels
```

### "Channel not found"
```bash
# Verify channel ID is correct (should be UC...)
# Check if channel is public
```

### "Database connection error"
```bash
# Check if PostgreSQL is running
make up

# Verify credentials in .env
```

### "BigQuery permission denied"
```bash
# Check service account has BigQuery Admin role
# Verify GOOGLE_APPLICATION_CREDENTIALS path
```

## 📚 Environment Variables

```bash
# Required
YOUTUBE_API_KEY=your_api_key           # Get from Google Cloud Console
GCP_PROJECT_ID=your_project_id         # GCP project
BQ_DATASET_ID=raw_yt                   # BigQuery dataset
GOOGLE_APPLICATION_CREDENTIALS=path     # Service account key

# PostgreSQL (from Docker Compose)
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=prefect
PG_USER=prefect
PG_PASSWORD=your_password

# Optional
MAX_VIDEOS_PER_CHANNEL=50              # Override channels.yml
MAX_COMMENTS_PER_VIDEO=100
API_DELAY_SECONDS=0.5
```

## 📈 Access Services

After running `make up`:
- **Prefect UI**: http://localhost:4200
- **Streamlit Dashboard**: http://localhost:8501
- **PgAdmin**: http://localhost:5050 (if enabled)

## 🆘 Getting Help

```bash
# CLI help
python -m extract.cli --help

# dbt help
python script/dbt_cli.py --help

# Monitor help
python script/monitor_quota.py --help
```

## License

MIT