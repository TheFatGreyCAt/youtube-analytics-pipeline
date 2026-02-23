import logging
import psycopg2
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import tempfile

from .config import (
    PG_CONN_STR, 
    GCP_PROJECT_ID, 
    BQ_DATASET_ID, 
    CREDENTIALS_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PostgresManager:
    
    def __init__(self, conn_str: Optional[str] = None):
        self.conn_str = conn_str or PG_CONN_STR
    
    def setup_tables(self):
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        schema_path = Path(__file__).parent / 'schemas' / 'schema_postgres.sql'
        
        if schema_path.exists():
            logging.info("Loading PostgreSQL schema from schema_postgres.sql...")
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            try:
                cur.execute(schema_sql)
                conn.commit()
                logging.info("PostgreSQL metadata tables created successfully")
            except Exception as e:
                logging.error(f"Error creating schema: {e}")
                conn.rollback()
                raise
        else:
            logging.error("schema_postgres.sql not found")
            raise FileNotFoundError("schema_postgres.sql is required")
        
        cur.close()
        conn.close()
        logging.info("PostgreSQL setup completed")
    
    def check_channel_exists(self, channel_id: str) -> bool:
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM channels_config WHERE channel_id = %s", (channel_id,))
        exists = cur.fetchone() is not None
        
        cur.close()
        conn.close()
        return exists
    
    def get_existing_channels(self, channel_ids: List[str]) -> set:
        if not channel_ids:
            return set()
        
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT channel_id FROM channels_config 
            WHERE channel_id = ANY(%s)
        """, (channel_ids,))
        
        existing = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
        
        return existing
    
    def add_channel(self, channel_id: str, channel_name: str, frequency_hours: int = 24):
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO channels_config (channel_id, channel_name, crawl_frequency_hours, next_crawl_ts)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (channel_id) 
            DO UPDATE SET channel_name = EXCLUDED.channel_name, 
                          crawl_frequency_hours = EXCLUDED.crawl_frequency_hours,
                          updated_at = NOW()
        """, (channel_id, channel_name, frequency_hours))
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"Added/Updated channel: {channel_name} ({channel_id})")
    
    def add_channels_batch(self, channels: List[Tuple[str, str, int]]):
        if not channels:
            return
        
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        from psycopg2.extras import execute_values
        
        execute_values(cur, """
            INSERT INTO channels_config (channel_id, channel_name, crawl_frequency_hours, next_crawl_ts)
            VALUES %s
            ON CONFLICT (channel_id) 
            DO UPDATE SET channel_name = EXCLUDED.channel_name, 
                          crawl_frequency_hours = EXCLUDED.crawl_frequency_hours,
                          updated_at = NOW()
        """, channels, template="(%s, %s, %s, NOW())")
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"Batch added/updated {len(channels)} channels")
    
    def get_channels_to_crawl(self, limit: int = 10) -> List[str]:
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT channel_id FROM channels_config 
            WHERE next_crawl_ts <= NOW() 
              AND crawl_status != 'failed'
              AND is_active = TRUE
            ORDER BY priority DESC, last_crawl_ts ASC NULLS FIRST
            LIMIT %s
        """, (limit,))
        
        channels = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        
        logging.info(f"Found {len(channels)} channels to crawl")
        return channels
    
    def list_channels(self) -> List[Tuple]:
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT channel_id, channel_name, crawl_status, 
                   last_crawl_ts, next_crawl_ts, crawl_frequency_hours, is_active
            FROM channels_config
            ORDER BY priority DESC, next_crawl_ts
        """)
        
        channels = cur.fetchall()
        cur.close()
        conn.close()
        
        return channels
    
    def update_crawl_success(self, channel_id: str, records_count: int, execution_time: float = 0):
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE channels_config 
            SET last_crawl_ts = NOW(), 
                next_crawl_ts = NOW() + (crawl_frequency_hours || ' hours')::INTERVAL,
                crawl_status = 'success',
                updated_at = NOW()
            WHERE channel_id = %s
        """, (channel_id,))
        
        cur.execute("""
            INSERT INTO crawl_log (channel_id, records_fetched, status, execution_time_seconds) 
            VALUES (%s, %s, 'success', %s)
        """, (channel_id, records_count, execution_time))
        
        conn.commit()
        cur.close()
        conn.close()
    
    def update_crawl_failed(self, channel_id: str, error_msg: str):
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE channels_config 
            SET crawl_status = 'failed', 
                next_crawl_ts = NOW() + INTERVAL '1 hour',
                updated_at = NOW()
            WHERE channel_id = %s
        """, (channel_id,))
        
        cur.execute("""
            INSERT INTO crawl_log (channel_id, status, error_msg) 
            VALUES (%s, 'failed', %s)
        """, (channel_id, error_msg))
        
        conn.commit()
        cur.close()
        conn.close()
    
    def get_crawl_history(self, channel_id: Optional[str] = None, limit: int = 10) -> List[Tuple]:
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        if channel_id:
            cur.execute("""
                SELECT channel_id, crawl_ts, records_fetched, status, error_msg
                FROM crawl_log
                WHERE channel_id = %s
                ORDER BY crawl_ts DESC
                LIMIT %s
            """, (channel_id, limit))
        else:
            cur.execute("""
                SELECT channel_id, crawl_ts, records_fetched, status, error_msg
                FROM crawl_log
                ORDER BY crawl_ts DESC
                LIMIT %s
            """, (limit,))
        
        logs = cur.fetchall()
        cur.close()
        conn.close()
        
        return logs
    
    def remove_channel(self, channel_id: str):
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("DELETE FROM channels_config WHERE channel_id = %s", (channel_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"Removed channel: {channel_id}")
    
    def update_api_quota(self, quota_used: int):
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO api_quota_usage (date, quota_used, updated_at)
            VALUES (CURRENT_DATE, %s, NOW())
            ON CONFLICT (date)
            DO UPDATE SET quota_used = api_quota_usage.quota_used + EXCLUDED.quota_used,
                          updated_at = NOW()
        """, (quota_used,))
        
        conn.commit()
        cur.close()
        conn.close()
    
    def get_api_quota_status(self) -> Dict:
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT quota_used, daily_limit
            FROM api_quota_usage
            WHERE date = CURRENT_DATE
        """)
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            return {
                'quota_used': row[0],
                'daily_limit': row[1],
                'percentage_used': (row[0] / row[1] * 100) if row[1] > 0 else 0
            }
        else:
            return {'quota_used': 0, 'daily_limit': 10000, 'percentage_used': 0}


class BigQueryManager:
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None):
        self.project_id = project_id or GCP_PROJECT_ID
        self.dataset_id = dataset_id or BQ_DATASET_ID
        
        if CREDENTIALS_PATH:
            credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
            self.client = bigquery.Client(project=self.project_id, credentials=credentials)
        else:
            self.client = bigquery.Client(project=self.project_id)
    
    def setup_tables(self):
        dataset_full_id = f"{self.project_id}.{self.dataset_id}"
        dataset = bigquery.Dataset(dataset_full_id)
        dataset.location = "asia-southeast1"
        
        try:
            self.client.create_dataset(dataset, exists_ok=True)
            logging.info(f"Dataset {dataset_full_id} ready")
        except Exception as e:
            logging.warning(f"Dataset creation: {e}")
        
        schema_path = Path(__file__).parent / 'schemas' / 'schema.sql'
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        schema_sql = schema_sql.replace('{PROJECT_ID}', self.project_id or '')
        schema_sql = schema_sql.replace('{DATASET_ID}', self.dataset_id or '')
        
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        
        for i, statement in enumerate(statements, 1):
            try:
                query_job = self.client.query(statement)
                query_job.result()
                
                table_type = 'VIEW' if 'VIEW' in statement else 'TABLE'
                logging.info(f"[{i}/{len(statements)}] {table_type} created")
            except Exception as e:
                logging.error(f"[{i}/{len(statements)}] Error: {e}")
        
        logging.info("BigQuery setup completed")
    
    def get_existing_records(self, table_name: str, ids: List[str]) -> set:
        if not ids:
            return set()
        
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        query = f"""
            SELECT id FROM `{table_id}`
            WHERE id IN UNNEST(@ids)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ids", "STRING", ids)
            ]
        )
        
        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            existing = {row.id for row in results}
            
            logging.info(f"Found {len(existing)}/{len(ids)} existing records in {table_name}")
            return existing
        except Exception as e:
            logging.warning(f"Error checking existing records in {table_name}: {e}")
            return set()
    
    def filter_new_records(self, table_name: str, rows: List[Dict]) -> List[Dict]:
        if not rows:
            return []
        
        ids = [str(row.get('id')) for row in rows if row.get('id')]
        
        existing_ids = self.get_existing_records(table_name, ids)
        
        new_rows = [row for row in rows if str(row.get('id')) not in existing_ids]
        
        skipped = len(rows) - len(new_rows)
        if skipped > 0:
            logging.info(f"Skipped {skipped} duplicate records, inserting {len(new_rows)} new records")
        
        return new_rows
    
    def insert_raw_data(self, table_name: str, rows: List[Dict], check_duplicates: bool = True):
        if not rows:
            logging.warning(f"No rows to insert into {table_name}")
            return
        
        if check_duplicates:
            rows = self.filter_new_records(table_name, rows)
            
            if not rows:
                logging.info(f"All records already exist in {table_name}, nothing to insert")
                return
        
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            for row in rows:
                tmp_file.write(json.dumps(row) + '\n')
            tmp_file_path = tmp_file.name
        
        try:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                autodetect=False,
                create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            )
            
            with open(tmp_file_path, 'rb') as source_file:
                load_job = self.client.load_table_from_file(
                    source_file,
                    table_id,
                    job_config=job_config
                )
            
            result = load_job.result()
            
            if load_job.errors:
                logging.error(f"Load job errors: {load_job.errors}")
                raise Exception(f"Load job failed with errors: {load_job.errors}")
            
            logging.info(f"Loaded {len(rows)} rows into {table_name}")
            
        except Exception as e:
            logging.error(f"Error loading into {table_name}: {e}")
            raise Exception(f"BigQuery batch load failed: {e}")
        
        finally:
            try:
                Path(tmp_file_path).unlink()
            except:
                pass
    
    def insert_channel_raw(self, channel_id: str, api_response: Dict, check_duplicates: bool = True):
        row = {
            'id': channel_id,
            'raw': json.dumps(api_response),
            'ingestion_time': datetime.utcnow().isoformat()
        }
        self.insert_raw_data('raw_channels', [row], check_duplicates=check_duplicates)
    
    def insert_videos_raw(self, videos: List[Dict], check_duplicates: bool = True):
        rows = []
        for video in videos:
            video_id = video.get('id', {}).get('videoId') if isinstance(video.get('id'), dict) else video.get('id', '')
            if video_id:
                rows.append({
                    'id': video_id,
                    'raw': json.dumps(video),
                    'ingestion_time': datetime.utcnow().isoformat()
                })
        
        if rows:
            self.insert_raw_data('raw_videos', rows, check_duplicates=check_duplicates)
    
    def insert_playlists_raw(self, channel_id: str, playlists: List[Dict], check_duplicates: bool = True):
        rows = []
        for playlist in playlists:
            playlist_id = playlist.get('id', '')
            if playlist_id:
                rows.append({
                    'id': playlist_id,
                    'channel_id': channel_id,
                    'raw': json.dumps(playlist),
                    'ingestion_time': datetime.utcnow().isoformat()
                })
        
        if rows:
            self.insert_raw_data('raw_playlists', rows, check_duplicates=check_duplicates)
    
    def insert_comments_raw(self, video_id: str, channel_id: str, comments: List[Dict], check_duplicates: bool = True):
        rows = []
        for comment in comments:
            comment_id = comment.get('id', '')
            if comment_id:
                rows.append({
                    'id': comment_id,
                    'video_id': video_id,
                    'channel_id': channel_id,
                    'raw': json.dumps(comment),
                    'ingestion_time': datetime.utcnow().isoformat()
                })
        
        if rows:
            self.insert_raw_data('raw_comments', rows, check_duplicates=check_duplicates)
