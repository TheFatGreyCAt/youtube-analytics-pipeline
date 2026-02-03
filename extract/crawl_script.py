import os
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.cloud import bigquery
import psycopg2
from typing import List, Dict

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Config
PROJECT_ID = os.getenv('PROJECT_ID', 'your-project')
DATASET_ID = os.getenv('DATASET_ID', 'yt_analytics')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
PG_CONN_STR = os.getenv('PG_CONN_STR')  # postgresql://user:pass@host/db
bq_client = bigquery.Client(project=PROJECT_ID)
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def get_channels_to_crawl() -> List[str]:
    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    cur.execute("""
        SELECT channel_id FROM channels_config 
        WHERE next_crawl_ts <= NOW() AND crawl_status != 'failed'
        ORDER BY last_crawl_ts ASC LIMIT 10
    """)
    channels = [row[0] for row in cur.fetchall()]
    conn.close()
    logging.info(f"Found {len(channels)} channels to crawl")
    return channels

def crawl_channel_raw(channel_id: str, api_response: Dict) -> Dict:
    row = {
        'id': channel_id,
        'raw': json.dumps(api_response),
        'ingestion_time': datetime.utcnow()
    }
    table_id = f"{PROJECT_ID}.{DATASET_ID}.raw_channels"
    bq_client.insert_rows_json(table_id, [row])
    logging.info(f"Saved raw channel {channel_id}")

def crawl_videos_raw(channel_id: str, api_response: Dict) -> List[str]:
    video_ids = [item['id']['videoId'] for item in api_response.get('items', [])]
    rows = [{
        'id': channel_id,
        'raw': json.dumps(api_response),
        'ingestion_time': datetime.utcnow()
    }]
    table_id = f"{PROJECT_ID}.{DATASET_ID}.raw_videos"
    bq_client.insert_rows_json(table_id, rows)
    logging.info(f"Saved {len(video_ids)} raw videos for {channel_id}")
    return video_ids

def crawl_playlists_raw(channel_id: str, api_response: Dict):
    rows = [{
        'id': channel_id,
        'channid': channel_id,  # owner
        'raw': json.dumps(api_response),
        'ingestion_time': datetime.utcnow()
    }]
    table_id = f"{PROJECT_ID}.{DATASET_ID}.raw_playlists"
    bq_client.insert_rows_json(table_id, rows)

def crawl_comments_raw(video_id: str, channel_id: str, api_response: Dict):
    rows = [{
        'id': video_id,
        'video_id': video_id,
        'channel_id': channel_id,
        'raw': json.dumps(api_response),
        'ingestion_time': datetime.utcnow()
    }]
    table_id = f"{PROJECT_ID}.{DATASET_ID}.raw_comments"
    bq_client.insert_rows_json(table_id, rows)

def crawl_channel_full(channel_id: str):
    try:
        # 1. Raw channels
        ch_req = youtube.channels().list(part='snippet,statistics,contentDetails,status', id=channel_id)
        ch_resp = ch_req.execute()
        crawl_channel_raw(channel_id, ch_resp)
        
        # 2. Raw videos (uploads playlist)
        uploads_id = ch_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        video_ids = []
        next_page = ''
        while True:
            vid_req = youtube.playlistItems().list(part='snippet', playlistId=uploads_id, maxResults=50, pageToken=next_page)
            vid_resp = vid_req.execute()
            video_ids.extend([item['snippet']['resourceId']['videoId'] for item in vid_resp['items']])
            next_page = vid_resp.get('nextPageToken')
            if not next_page: break
            time.sleep(1)  # Rate limit
        
        # Batch videos details (50/video)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            vstats_req = youtube.videos().list(part='snippet,statistics,contentDetails', id=','.join(batch))
            vstats_resp = vstats_req.execute()
            crawl_videos_raw(channel_id, vstats_resp)
            time.sleep(1)
        
        # 3. Raw playlists
        pl_req = youtube.playlists().list(part='snippet,contentDetails,status', channelId=channel_id, maxResults=50)
        pl_resp = pl_req.execute()
        crawl_playlists_raw(channel_id, pl_resp)
        
        # 4. Raw comments (top 5/video)
        for vid_id in video_ids[:10]:  # Limit 10 videos demo
            cm_req = youtube.commentThreads().list(part='snippet', videoId=vid_id, maxResults=5, order='relevance')
            cm_resp = cm_req.execute()
            crawl_comments_raw(vid_id, channel_id, cm_resp)
            time.sleep(0.5)
        
        # Update PostgreSQL
        update_pg_success(channel_id, len(video_ids))
        
    except Exception as e:
        logging.error(f"Error crawling {channel_id}: {e}")
        update_pg_failed(channel_id, str(e))

def update_pg_success(channel_id: str, video_count: int):
    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    cur.execute("""
        UPDATE channels_config 
        SET last_crawl_ts = NOW(), next_crawl_ts = NOW() + INTERVAL '24 hours', crawl_status = 'success'
        WHERE channel_id = %s
    """, (channel_id,))
    # Log
    cur.execute("""
        INSERT INTO crawl_log (channel_id, records_fetched, status) 
        VALUES (%s, %s, 'success')
    """, (channel_id, video_count))
    conn.commit()
    conn.close()

def update_pg_failed(channel_id: str, error: str):
    conn = psycopg2.connect(PG_CONN_STR)
    cur = conn.cursor()
    cur.execute("UPDATE channels_config SET crawl_status = 'failed', next_crawl_ts = NOW() + INTERVAL '1 hour' WHERE channel_id = %s", (channel_id,))
    cur.execute("INSERT INTO crawl_log (channel_id, status, error_msg) VALUES (%s, 'failed', %s)", (channel_id, error))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    channels = get_channels_to_crawl()
    for ch_id in channels:
        crawl_channel_full(ch_id)
        time.sleep(2)  # Channel cooldown
    logging.info("Daily crawl completed")
