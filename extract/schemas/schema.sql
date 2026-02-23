CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.fact_videos` (
  video_id STRING NOT NULL,
  channel_id STRING NOT NULL,
  published_at TIMESTAMP,
  view_count INT64,
  like_count INT64,
  comment_count INT64,
  duration_seconds INT64,
  category_id INT64,
  title_length INT64,
  engagement_rate FLOAT64,
  title STRING,
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(published_at)
CLUSTER BY channel_id, category_id;

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.dim_channels` (
  channel_id STRING NOT NULL,
  title STRING,
  subscriber_count INT64,
  country STRING,
  published_at TIMESTAMP,
  ingestion_time TIMESTAMP
)
CLUSTER BY country;

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_videos` (
  id STRING NOT NULL,
  raw STRING NOT NULL,
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time);

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_channels` (
  id STRING NOT NULL,
  raw STRING NOT NULL,
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time);

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_playlists` (
  id STRING NOT NULL,
  channel_id STRING NOT NULL,
  raw STRING NOT NULL,
  ingestion_time TIMESTAMP
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY channel_id;

CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.raw_comments` (
  id STRING NOT NULL,
  video_id STRING NOT NULL,
  channel_id STRING NOT NULL,
  raw STRING NOT NULL,
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
