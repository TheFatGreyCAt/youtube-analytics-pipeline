-- BIGQUERY SCHEMA - Raw Data Storage Only
-- Purpose: Store raw API responses from YouTube
-- Processing: Use dbt for transformation

CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}.raw_channels` (
  id STRING NOT NULL,
  raw JSON NOT NULL,
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY id;

CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}.raw_videos` (
  id STRING NOT NULL,
  raw JSON NOT NULL,
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY id;

CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}.raw_playlists` (
  id STRING NOT NULL,
  channel_id STRING NOT NULL,
  raw JSON NOT NULL,
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY channel_id, id;

CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}.raw_comments` (
  id STRING NOT NULL,
  video_id STRING NOT NULL,
  channel_id STRING NOT NULL,
  raw JSON NOT NULL,
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(ingestion_time)
CLUSTER BY video_id, channel_id;
