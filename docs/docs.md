# BÁO CÁO DỰ ÁN: YOUTUBE ANALYTICS PIPELINE

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Giới thiệu
**YouTube Analytics Pipeline** là một hệ thống data pipeline hoàn chỉnh để thu thập, xử lý và phân tích dữ liệu từ YouTube Data API v3. Dự án được xây dựng dựa trên nền tảng Modern Data Stack với các công nghệ:

- **Data Extraction**: Python + YouTube Data API v3
- **Data Storage**: PostgreSQL (staging) + Google BigQuery (warehouse)
- **Data Transformation**: dbt (data build tool)
- **Orchestration**: Prefect 3.0
- **Visualization**: Streamlit
- **Infrastructure**: Docker Compose

### 1.2. Mục tiêu dự án
- Tự động thu thập dữ liệu từ nhiều kênh YouTube theo lịch trình
- Xử lý và làm sạch dữ liệu thô thành các bảng phân tích
- Theo dõi hiệu suất video và kênh theo thời gian
- Tối ưu hóa việc sử dụng YouTube API quota (10,000 units/day)
- Cung cấp dashboard trực quan cho việc ra quyết định

### 1.3. Đối tượng sử dụng
- Content creators cần phân tích nhiều kênh YouTube
- Marketing teams theo dõi performance của video campaigns
- Data analysts nghiên cứu xu hướng nội dung YouTube
- Developers muốn tìm hiểu về modern data pipeline

---

## 2. NGUỒN GỐC VÀ KẾ THỪA TỪ FIVETRAN

### 2.1. Fivetran YouTube Analytics Connector
Dự án này lấy cảm hứng và kế thừa một số thành phần từ **Fivetran's dbt YouTube Analytics package**, một open-source package dbt để transform dữ liệu YouTube từ Fivetran connector.

**Fivetran là gì?**
- Fivetran là một ELT (Extract, Load, Transform) platform
- Cung cấp 200+ pre-built connectors để sync data từ các nguồn khác nhau
- YouTube Analytics connector của Fivetran tự động sync data từ YouTube Analytics API
- Dữ liệu được load trực tiếp vào data warehouse (BigQuery, Snowflake, Redshift...)
- Chi phí: $1-2 per monthly active row (MAR)

### 2.2. Những gì được kế thừa từ Fivetran dbt package

#### a) Cấu trúc dbt models theo medallion architecture:
```
staging/       → Clean & standardize raw data
intermediate/  → Business logic transformations  
mart/          → Final analytics tables
```

#### b) Package dependencies trong `packages.yml`:
```yaml
- package: fivetran/fivetran_utils
  version: [">=0.4.0", "<0.5.0"]
- package: dbt-labs/dbt_utils
  version: [">=1.0.0", "<2.0.0"]
```

**fivetran_utils** cung cấp các macros hữu ích:
- `add_pass_through_columns()`: Cho phép thêm custom columns
- `source_relation`: Hỗ trợ multi-source scenarios
- Testing utilities và helper functions

#### c) Naming conventions:
- Staging models: `stg_youtube__<entity>`
- Mart models: `fct_` (fact) và `dim_` (dimension)
- Intermediate models: `int_youtube__<purpose>`

#### d) Integration tests structure:
Thư mục `integration_tests/` với:
- Integrity tests: Kiểm tra data consistency
- Consistency tests: So sánh với expected outputs
- Seeds data: Sample data để test

#### e) Metadata columns:
- `_fivetran_synced`: Timestamp của lần sync cuối
- `_fivetran_id`: Unique identifier cho mỗi record

### 2.3. Tại sao không dùng Fivetran trực tiếp?

**Ưu điểm của Fivetran:**
- ✅ Setup nhanh, không cần code
- ✅ Tự động handle errors, retries
- ✅ Managed infrastructure
- ✅ Support nhiều data sources

**Nhược điểm và lý do build custom solution:**
- ❌ **Chi phí cao**: $1-2/MAR, với 15 channels x 50 videos = $750-1500/month
- ❌ **Giới hạn kiểm soát**: Không thể customize crawl logic
- ❌ **API Quota**: Không control được cách sử dụng YouTube quota
- ❌ **Vendor lock-in**: Phụ thuộc vào Fivetran platform
- ❌ **Learning opportunity**: Mất cơ hội học về data engineering
- ❌ **Flexibility**: Khó thêm custom features (ví dụ: crawl comments có điều kiện)

**Lợi ích của custom solution:**
- ✅ **Miễn phí**: Chỉ tốn chi phí GCP/BigQuery
- ✅ **Full control**: Tùy chỉnh mọi khía cạnh của pipeline
- ✅ **Quota optimization**: Implement smart quota management
- ✅ **Custom features**: Thêm bất kỳ logic nào cần thiết
- ✅ **Educational**: Học về data engineering từ đầu đến cuối

---

## 3. KIẾN TRÚC HỆ THỐNG

### 3.1. Sơ đồ tổng quan

```
┌─────────────────┐
│  YouTube API    │
│   (Source)      │
└────────┬────────┘
         │
         │ Python Crawler
         │ (extract/)
         ↓
┌─────────────────┐      ┌─────────────────┐
│   PostgreSQL    │      │   BigQuery      │
│  (Staging DB)   │─────→│  (Data Warehouse)│
└─────────────────┘      └────────┬────────┘
         ↑                         │
         │                         │ dbt transformation
         │                         │ (dbt_project/)
    ┌────┴─────┐                  ↓
    │  Prefect │           ┌─────────────────┐
    │(Scheduler)│           │  Analytics      │
    └──────────┘           │  Tables (Mart)  │
                           └────────┬────────┘
                                    │
                                    ↓
                           ┌─────────────────┐
                           │   Streamlit     │
                           │   Dashboard     │
                           └─────────────────┘
```

### 3.2. Data Flow chi tiết

#### Phase 1: Data Extraction (Python)
```python
# extract/crawlers.py
YouTube API → Python Crawler → PostgreSQL (raw JSON)
                              → BigQuery (raw tables)

Tables created:
- raw_videos
- raw_channels  
- raw_playlists
- raw_comments (optional)
```

**Features:**
- Rate limiting: 0.5s delay giữa các requests
- Retry logic: 2 retries với exponential backoff
- Quota monitoring: Dừng tự động khi đạt 90% quota
- Error handling: Log chi tiết, không crash toàn bộ pipeline
- Incremental updates: Chỉ crawl videos mới hoặc cần update

#### Phase 2: Data Transformation (dbt)
```sql
-- dbt_project/models/

Staging Layer (stg_youtube__*):
- Parse JSON thành columns
- Type casting và validation
- Deduplicate records
- Add metadata columns

Intermediate Layer (int_youtube__*):
- Join các entities
- Calculate derived metrics
- Apply business logic

Mart Layer (fct_*, dim_*):
- fct_video_performance: Video metrics over time
- dim_channel_summary: Channel aggregations
- agg_daily_metrics: Daily rollups
```

#### Phase 3: Orchestration (Prefect)
```python
# orchestrate/flows/

daily-youtube-analytics:
  Schedule: 2:00 AM daily
  Steps:
    1. Check quota
    2. Crawl channels (limit=15)
    3. Run dbt models
    4. Run dbt tests
    5. Send notifications

extract-3times-daily:
  Schedule: 8:00 AM, 2:00 PM, 8:00 PM
  Steps: Chỉ crawl data

dbt-transform-daily:
  Schedule: 30 mins sau extract
  Steps: Transform data mới
```

### 3.3. Technology Stack

#### Backend Services:
- **PostgreSQL 14**: Temporary staging storage, metadata
- **Redis 7**: Prefect message queue và caching
- **Prefect Server 3.0**: Workflow orchestration UI
- **Prefect Worker**: Execute scheduled flows

#### Data Processing:
- **Python 3.12**: Extraction logic
- **dbt-core 1.7.0**: SQL transformations
- **dbt-bigquery**: BigQuery adapter
- **Google BigQuery**: Cloud data warehouse

#### APIs & SDKs:
- **google-api-python-client**: YouTube Data API v3
- **google-cloud-bigquery**: BigQuery Python client
- **psycopg2**: PostgreSQL adapter

#### Development Tools:
- **Docker Compose**: Local development environment
- **Poetry/pip**: Python dependency management
- **Pytest**: Unit testing
- **Ruff**: Python linting

---

## 4. CÁC THÀNH PHẦN MỚI (SO VỚI FIVETRAN)

### 4.1. Custom Python Crawler (`extract/`)

#### Cấu trúc module:
```
extract/
├── __init__.py
├── cli.py           # Command-line interface
├── config.py        # Configuration management
├── crawlers.py      # Core crawling logic
├── db_manager.py    # Database operations
└── schemas/
    └── schema_postgres.sql
```

#### Tính năng chính:

**a) Multi-channel management:**
```python
# config/channels.yml
channels:
  - id: UCXuqSBlHAE6Xw-yeJA0Tunw
    name: Linus Tech Tips
    frequency_hours: 24
    priority: 1
    active: true
    include_comments: false
```

**b) Smart quota management:**
```python
# script/monitor_quota.py
- Real-time quota tracking
- Estimate cost before crawl
- Auto-stop at 90% usage
- Daily quota reset detection
```

**c) CLI tool:**
```bash
# Add channels
python -m extract.cli add <channel_id> "<name>"

# Crawl specific channel
python -m extract.cli crawl --channel <id>

# Crawl from config file
python -m extract.cli crawl-file --limit 10

# View history
python -m extract.cli history

# List all channels
python -m extract.cli channels
```

**d) Incremental crawling:**
- Chỉ crawl videos uploaded sau lần crawl cuối
- Update statistics cho videos hiện có
- Skip videos đã có đầy đủ data

**e) Error resilience:**
```python
@retry(max_attempts=3, backoff=exponential)
def fetch_video_data(video_id):
    try:
        # API call
    except HttpError as e:
        if e.resp.status == 403:
            # Quota exceeded
        elif e.resp.status == 404:
            # Video deleted
```

### 4.2. Prefect Orchestration (`orchestrate/`)

#### Cấu trúc flows:
```
orchestrate/
├── flows/
│   ├── complete_pipeline.py      # Full ETL pipeline
│   ├── extract_youtube_data.py   # Data extraction only
│   └── transform_with_dbt.py     # dbt transformation only
├── tasks/
│   ├── youtube_tasks.py          # YouTube API tasks
│   ├── dbt_tasks.py             # dbt operations
│   └── notification_tasks.py     # Alerts & monitoring
└── deployments/
    └── deploy_daily_schedule.py
```

#### Prefect advantages vs cron:
- **Visual monitoring**: Web UI để track runs
- **Retry logic**: Automatic retry với configurable strategy
- **Alerting**: Email/Slack notifications on failure
- **Parameterization**: Easy to change parameters
- **Concurrency control**: Prevent overlapping runs
- **Logging**: Centralized log storage

### 4.3. Custom dbt Macros

#### a) `parse_iso8601_duration.sql`
```sql
-- Convert PT1H23M45S → 5025 seconds
{{ parse_iso8601_duration('PT1H23M45S') }}
```

YouTube API trả về duration theo ISO 8601 format. Macro này convert sang seconds để dễ tính toán.

#### b) `deduplicate_by_latest.sql`
```sql
-- Keep only latest version of each record
{{ deduplicate_by_latest(
    'raw_videos',
    'id',
    'ingestion_time'
) }}
```

Xử lý trường hợp crawl video nhiều lần trong ngày, chỉ giữ version mới nhất.

#### c) `get_layer_schema.sql`
```sql
-- Dynamic schema naming
-- staging → stg_yt
-- intermediate → int_yt
-- mart → mart_yt
```

Tổ chức schemas theo layer cho dễ quản lý.

#### d) `get_passthrough_columns.sql`
```sql
-- Allow users to add custom columns
-- without modifying source code
{{ get_passthrough_columns(['custom_tag', 'internal_notes']) }}
```

### 4.4. Data Quality Framework

#### Built-in tests:
```sql
-- tests/assert_positive_statistics.sql
SELECT * FROM {{ ref('fct_video_performance') }}
WHERE view_count < 0
   OR like_count < 0
   OR comment_count < 0

-- tests/assert_valid_engagement_rates.sql  
SELECT * FROM {{ ref('fct_video_performance') }}
WHERE engagement_rate > 1.0
   OR engagement_rate < 0
```

#### dbt tests in models:
```yaml
# models/staging/_stg_youtube__models.yml
columns:
  - name: video_id
    tests:
      - unique
      - not_null
  - name: view_count
    tests:
      - not_null
      - dbt_utils.accepted_range:
          min_value: 0
```

### 4.5. Streamlit Dashboard (`serve/`)

#### Features:
- **Overview metrics**: Total views, subscribers, videos
- **Channel comparison**: Side-by-side performance
- **Video analytics**: Individual video deep-dive
- **Trend analysis**: Time-series charts
- **Category insights**: Performance by video category
- **Export functionality**: Download data as CSV

#### Charts implemented:
- Line charts: Views/likes over time
- Bar charts: Top performing videos
- Scatter plots: Engagement vs views
- Heatmaps: Upload patterns by day/hour
- Pie charts: Video distribution by category

### 4.6. Infrastructure as Code

#### Docker Compose setup:
```yaml
services:
  postgres:       # Metadata & staging
  redis:          # Prefect message queue
  prefect-server: # Orchestration UI
  prefect-worker: # Task execution
  streamlit:      # Dashboard (optional)
```

#### Benefits:
- **Reproducible**: Bất kỳ ai cũng có thể setup giống hệt
- **Isolated**: Không ảnh hưởng system packages
- **Scalable**: Dễ dàng scale services riêng lẻ
- **Version controlled**: Infrastructure changes tracked in Git

### 4.7. Development Tools

#### a) Makefile shortcuts:
```makefile
make setup          # Initialize database
make up            # Start services
make down          # Stop services
make crawl         # Run extraction
make dbt-run       # Run transformations
make prefect-deploy # Deploy workflows
```

#### b) Scripts automation:
```
script/
├── setup_all.py           # One-click setup
├── monitor_quota.py       # Quota monitoring
├── bulk_add_channels.py   # Import channels from CSV
├── deploy_prefect.py      # Automated deployment
└── dbt_cli.py            # dbt wrapper with logging
```

#### c) Configuration management:
```
config/
├── channels.yml           # Channel definitions
├── channels_template.csv  # Bulk import template
└── prefect.yaml          # Workflow schedules
```

---

## 5. PHÂN TÍCH SO SÁNH

### 5.1. Fivetran Solution vs Custom Solution

| Aspect | Fivetran | Custom (This Project) |
|--------|----------|----------------------|
| **Setup Time** | 15 minutes | 2-3 hours |
| **Monthly Cost** | $750-1500 | $10-50 (GCP only) |
| **Code Required** | Minimal (dbt only) | Extensive (Python + dbt) |
| **Customization** | Limited | Unlimited |
| **Maintenance** | Low (managed) | Medium (self-managed) |
| **Learning Curve** | Low | High |
| **Quota Control** | No | Yes |
| **Feature Addition** | Depends on Fivetran | Immediate |
| **Scalability** | Automatic | Manual (but flexible) |
| **Data Freshness** | Fixed schedule | Custom schedule |

### 5.2. Khi nào nên dùng Fivetran?

✅ **Nên dùng Fivetran khi:**
- Budget không là vấn đề
- Cần setup nhanh (production ASAP)
- Team không có Python developers
- Cần sync nhiều data sources (>10)
- Ưu tiên stability over customization

### 5.3. Khi nào nên dùng Custom Solution?

✅ **Nên dùng Custom Solution khi:**
- Budget giới hạn hoặc startup stage
- Cần control chi tiết crawling logic
- Muốn optimize API quota usage
- Có yêu cầu custom features
- Team có Python & data engineering skills
- Muốn học về modern data engineering

---

## 6. ĐIỂM MẠNH CỦA DỰ ÁN

### 6.1. Technical Excellence

**1. Idempotency:**
```python
# Chạy lại nhiều lần → cùng kết quả
# Không tạo duplicate records
# Safe để retry failed runs
```

**2. Incremental Loading:**
```python
# Chỉ process data mới
# Tiết kiệm API quota
# Faster execution
```

**3. Type Safety:**
```python
# Pydantic models cho validation
# Type hints everywhere
# Catch errors at parse time
```

**4. Observability:**
- Detailed logging at every step
- Metrics tracking (quota, rows processed)
- Error notifications
- Prefect UI monitoring

### 6.2. Best Practices

**1. Separation of Concerns:**
```
extract/     → Data collection
dbt_project/ → Data transformation
orchestrate/ → Workflow management
serve/       → Data presentation
```

**2. Configuration Management:**
- Environment variables cho secrets
- YAML files cho declarative config
- Template files cho easy setup

**3. Testing:**
- Unit tests cho Python code
- dbt tests cho data quality
- Integration tests với sample data

**4. Documentation:**
- Inline comments
- README với examples
- Schema documentation trong dbt

### 6.3. Production-Ready Features

- ✅ **Error handling**: Comprehensive try-catch
- ✅ **Retry logic**: Exponential backoff
- ✅ **Rate limiting**: Respect API limits
- ✅ **Quota monitoring**: Prevent overages
- ✅ **Data validation**: Schema enforcement
- ✅ **Logging**: Structured logs
- ✅ **Alerts**: Failure notifications
- ✅ **Rollback**: Version control với dbt
- ✅ **Monitoring**: Prefect dashboard
- ✅ **Scalability**: Container-based

---

## 7. HẠNG CHẾ VÀ CÁCH KHẮC PHỤC

### 7.1. Limitations hiện tại

**1. API Quota Constraints:**
- ❌ Giới hạn 10,000 units/day
- ❌ Không thể crawl real-time
- ✅ **Solution**: Smart scheduling, prioritize channels

**2. No Historical Data:**
- ❌ Chỉ có data từ khi bắt đầu crawl
- ❌ Không access được YouTube Analytics API
- ✅ **Solution**: Start crawling ASAP, backfill where possible

**3. Single-threaded Crawling:**
- ❌ Sequential API calls
- ❌ Slow khi có nhiều channels
- ✅ **Solution**: Có thể implement parallel crawling với semaphore

**4. Local Deployment:**
- ❌ Requires machine running 24/7
- ❌ No cloud deployment yet
- ✅ **Solution**: Deploy lên GCP Cloud Run or AWS ECS

### 7.2. Future Enhancements

**Phase 2 - Cloud Native:**
```
- Deploy Prefect to Cloud Run
- Use Cloud Scheduler instead of local
- Cloud SQL instead of local PostgreSQL
- Terraform for infrastructure
```

**Phase 3 - Advanced Analytics:**
```
- Sentiment analysis on comments
- Thumbnail analysis với Vision API
- Competitor analysis
- Trend prediction với ML
```

**Phase 4 - Multi-platform:**
```
- Add TikTok data
- Add Instagram data
- Cross-platform analytics
```

---

## 8. HƯỚNG DẪN SỬ DỤNG NÂNG CAO

### 8.1. Customize Crawling Logic

```python
# extract/crawlers.py

# Thêm custom fields
def enrich_video_data(video):
    video['custom_score'] = calculate_score(video)
    video['trending'] = is_trending(video)
    return video

# Conditional crawling
def should_crawl_comments(video):
    return video['view_count'] > 10000
```

### 8.2. Extend dbt Models

```sql
-- models/mart/custom_metrics.sql

{{ config(
    materialized='incremental',
    unique_key='video_id'
) }}

SELECT
    video_id,
    {{ calculate_virality_score() }} as virality,
    {{ predict_future_views() }} as predicted_views
FROM {{ ref('fct_video_performance') }}
```

### 8.3. Add Custom Dashboards

```python
# serve/pages/custom_analysis.py

import streamlit as st

def show_custom_analysis():
    st.title("Custom Analysis")
    # Your custom logic
```

---

## 9. KẾT LUẬN

### 9.1. Thành tựu đạt được

Dự án **YouTube Analytics Pipeline** đã thành công trong việc:

1. ✅ Xây dựng một **end-to-end data pipeline** hoàn chỉnh
2. ✅ Tiết kiệm **$750-1500/month** so với Fivetran
3. ✅ Có **full control** over data collection và processing
4. ✅ Implement **production-grade features** (retry, monitoring, testing)
5. ✅ Tạo **learning resource** cho data engineering community
6. ✅ Áp dụng **modern data stack** best practices
7. ✅ Kế thừa **proven patterns** từ Fivetran dbt package

### 9.2. Bài học kinh nghiệm

**Technical Lessons:**
- dbt là công cụ mạnh mẽ cho data transformation
- Prefect giúp orchestration dễ dàng hơn nhiều so với Airflow
- API quota management là critical cho YouTube projects
- Docker Compose makes development reproducible

**Business Lessons:**
- Build vs Buy decision phụ thuộc vào context
- Open-source packages tiết kiệm rất nhiều effort
- Documentation là investment, not cost
- Automation saves time in the long run

### 9.3. Recommendations

**Cho beginners:**
- Start với Fivetran để hiểu data flow
- Sau đó build custom để hiểu deeper
- Focus vào data modeling (dbt) trước
- Orchestration có thể dùng cron trước rồi migrate sang Prefect

**Cho advanced users:**
- Extend project với ML models
- Implement real-time streaming với Pub/Sub
- Add more data sources
- Build recommendation engine

### 9.4. Đánh giá cuối cùng

Dự án này là một **excellent learning project** và **viable production solution** cho:
- Small to medium YouTube creators (5-20 channels)
- Marketing agencies managing client channels
- Data analysts muốn portfolio project
- Startups cần cost-effective analytics

**ROI (Return on Investment):**
- Development time: 40-60 hours
- Monthly savings: $750-1500
- Payback period: < 1 month
- Learning value: Priceless 🚀

---

## PHỤ LỤC

### A. Tech Stack Chi Tiết

#### A.1. Python Dependencies
```toml
# Core
python = "^3.12"
google-api-python-client = "^2.100.0"
google-cloud-bigquery = "^3.23.1"
psycopg2-binary = "^2.9.9"

# Orchestration
prefect = "^3.0.0"
prefect-gcp = "^0.4.0"

# Transformation
dbt-core = "^1.7.0"
dbt-bigquery = "^1.7.0"

# Analytics
pandas = "^2.2.0"
numpy = "^1.26.0"

# Dashboard
streamlit = "^1.32.0"
plotly = "^5.20.0"
```

#### A.2. dbt Packages
```yaml
packages:
  - package: fivetran/fivetran_utils
    version: [">=0.4.0", "<0.5.0"]
  - package: dbt-labs/dbt_utils
    version: [">=1.0.0", "<2.0.0"]
  - package: dbt-labs/spark_utils
    version: [">=0.3.0", "<0.4.0"]
```

### B. Database Schemas

#### B.1. PostgreSQL (Staging)
```sql
-- Channels metadata
CREATE TABLE channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    frequency_hours INTEGER DEFAULT 24,
    priority INTEGER DEFAULT 1,
    active BOOLEAN DEFAULT TRUE,
    include_comments BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Crawl history
CREATE TABLE crawl_history (
    id SERIAL PRIMARY KEY,
    channel_id TEXT REFERENCES channels(id),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,
    videos_crawled INTEGER,
    quota_used INTEGER,
    error_message TEXT
);
```

#### B.2. BigQuery (Warehouse)
```sql
-- Raw tables
raw_yt.raw_videos
raw_yt.raw_channels
raw_yt.raw_playlists
raw_yt.raw_comments

-- Staging tables
stg_yt.stg_youtube__videos
stg_yt.stg_youtube__channels
stg_yt.stg_youtube__playlists

-- Mart tables
mart_yt.fct_video_performance
mart_yt.dim_channel_summary
mart_yt.agg_daily_metrics
```

### C. API Quota Breakdown

| Operation | Cost (units) | Notes |
|-----------|--------------|-------|
| channels.list | 1 | Basic channel info |
| playlistItems.list | 1 | Per 50 videos |
| videos.list | 1 | Per 50 videos |
| commentThreads.list | 1 | Per 100 comments |
| search.list | 100 | ❌ Very expensive! |

**Example calculation for 1 channel:**
```
1 channel info          = 1 unit
1 playlist fetch        = 1 unit
50 videos details       = 1 unit
10 videos x comments    = 10 units
Total per channel       = 13 units

Max channels per day    = 10,000 / 13 ≈ 769 channels
Realistic (with buffer) = 15-20 channels
```

### D. Useful Resources

#### Documentation:
- YouTube Data API: https://developers.google.com/youtube/v3
- dbt Docs: https://docs.getdbt.com
- Prefect Docs: https://docs.prefect.io
- BigQuery Docs: https://cloud.google.com/bigquery/docs

#### Related Projects:
- Fivetran dbt YouTube: https://github.com/fivetran/dbt_youtube_analytics
- Meltano (Open-source ELT): https://meltano.com
- Singer Taps: https://www.singer.io

---

**Document Version**: 1.0  
**Last Updated**: February 23, 2026  
**Author**: YouTube Analytics Pipeline Team  
**License**: MIT
