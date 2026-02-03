CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.fact_videos` (
  video_id STRING NOT NULL,           -- PK: YouTube video ID
  channel_id STRING NOT NULL,         -- FK: Channel reference
  published_at TIMESTAMP,             -- Feature: Publication time (time series)
  view_count INT64,                   -- Target KPI #1: View performance
  like_count INT64,                   -- Target: Engagement metric
  comment_count INT64,                -- Target: Community engagement
  duration_seconds INT64,             -- Feature: Video length for ML
  category_id INT64,                  -- Feature: One-hot encoding for ML
  title_length INT64,                 -- Feature: Content characteristics
  engagement_rate FLOAT64,            -- Feature: KPI #2 (likes+comments)/views
  title STRING,                       -- Reference: Video title
  ingestion_time TIMESTAMP            -- Audit: When data was loaded
)
PARTITION BY DATE(published_at)
CLUSTER BY channel_id, category_id;

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.dim_channels` (
  channel_id STRING NOT NULL,         -- PK: YouTube channel ID
  title STRING,                       -- Branding: Channel name (embedding input)
  subscriber_count INT64,             -- KPI #1: Growth metric
  country STRING,                     -- Feature: Geo-mapping, one-hot encoding
  published_at TIMESTAMP,             -- Feature: Channel age
  ingestion_time TIMESTAMP            -- Audit: When data was loaded
)
CLUSTER BY country;

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_videos` (
  id STRING NOT NULL,
  raw STRING NOT NULL,                -- Full API response as JSON
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time);

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_channels` (
  id STRING NOT NULL,
  raw STRING NOT NULL,                -- Full API response as JSON
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time);

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_playlists` (
  id STRING NOT NULL,
  channel_id STRING NOT NULL,         -- FK: Owner channel
  raw STRING NOT NULL,                -- Full API response as JSON
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY channel_id;

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_comments` (
  id STRING NOT NULL,
  video_id STRING NOT NULL,           -- FK: Parent video
  channel_id STRING NOT NULL,         -- FK: Video owner channel
  raw STRING NOT NULL,                -- Full API response as JSON
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY video_id, channel_id;

-- View for ML Training: videos with channel context
CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.v_videos_ml` AS
SELECT
  f.video_id,
  f.channel_id,
  f.published_at,
  f.view_count,
  f.like_count,
  f.comment_count,
  f.duration_seconds,
  f.category_id,
  f.title_length,
  f.engagement_rate,
  c.subscriber_count,
  c.country,
  DATE_DIFF(CURRENT_DATE(), DATE(f.published_at), DAY) as days_since_publish
FROM `{PROJECT_ID}.{DATASET_ID}.fact_videos` f
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.dim_channels` c USING(channel_id)
WHERE f.view_count IS NOT NULL;

-- View for KPI Dashboard
CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.v_channel_performance` AS
SELECT
  c.channel_id,
  c.title,
  c.subscriber_count,
  c.country,
  COUNT(f.video_id) as total_videos,
  SUM(f.view_count) as total_views,
  SUM(f.like_count) as total_likes,
  SUM(f.comment_count) as total_comments,
  AVG(f.engagement_rate) as avg_engagement_rate,
  AVG(f.duration_seconds) as avg_duration,
  MIN(f.published_at) as first_video_date,
  MAX(f.published_at) as latest_video_date
FROM `{PROJECT_ID}.{DATASET_ID}.dim_channels` c
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.fact_videos` f USING(channel_id)
GROUP BY 1, 2, 3, 4;

-- View for Category Analysis
CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.v_category_insights` AS
SELECT
  category_id,
  COUNT(*) as video_count,
  AVG(view_count) as avg_views,
  AVG(like_count) as avg_likes,
  AVG(engagement_rate) as avg_engagement,
  AVG(duration_seconds) as avg_duration,
  STDDEV(view_count) as view_stddev
FROM `{PROJECT_ID}.{DATASET_ID}.fact_videos`
WHERE category_id IS NOT NULL
GROUP BY category_id;
