import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import pandas as pd
from google.cloud import bigquery
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import googleapiclient.discovery

from schema_utilities import YouTubeDataTransformer, SchemaValidator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# YouTube API configuration
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']


class YouTubeCrawler:
    """Crawl YouTube data using YouTube API v3"""
    
    def __init__(self, api_key: str):
        """Initialize YouTube API client"""
        self.youtube = googleapiclient.discovery.build(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            developerKey=api_key
        )
        self.api_key = api_key
        logger.info("YouTube API client initialized")
    
    def get_video_details(self, video_ids: List[str], parts: List[str] = None) -> List[Dict[str, Any]]:
        """Get video details from YouTube API"""
        if parts is None:
            parts = ['snippet', 'statistics', 'contentDetails']
        
        videos = []
        # YouTube API limit: 50 videos per request
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            try:
                request = self.youtube.videos().list(
                    part=','.join(parts),
                    id=','.join(batch)
                )
                response = request.execute()
                videos.extend(response.get('items', []))
                logger.info(f"Retrieved {len(response.get('items', []))} videos")
            except Exception as e:
                logger.error(f"Error fetching videos: {e}")
        
        return videos
    
    def get_channel_details(self, channel_ids: List[str], parts: List[str] = None) -> List[Dict[str, Any]]:
        """Get channel details from YouTube API"""
        if parts is None:
            parts = ['snippet', 'statistics', 'topicDetails']
        
        channels = []
        # YouTube API limit: 50 channels per request
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i+50]
            try:
                request = self.youtube.channels().list(
                    part=','.join(parts),
                    id=','.join(batch)
                )
                response = request.execute()
                channels.extend(response.get('items', []))
                logger.info(f"Retrieved {len(response.get('items', []))} channels")
            except Exception as e:
                logger.error(f"Error fetching channels: {e}")
        
        return channels
    
    def search_videos(self, query: str, max_results: int = 10, 
                     order: str = 'relevance') -> List[str]:
        """Search videos by query and return video IDs"""
        video_ids = []
        try:
            request = self.youtube.search().list(
                q=query,
                part='snippet',
                maxResults=min(max_results, 50),
                type='video',
                order=order
            )
            response = request.execute()
            
            for item in response.get('items', []):
                video_ids.append(item['id']['videoId'])
            
            logger.info(f"Found {len(video_ids)} videos for query: {query}")
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
        
        return video_ids
    
    def get_channel_playlists(self, channel_id: str) -> List[Dict[str, Any]]:
        """Get all playlists from a channel"""
        playlists = []
        next_page_token = None
        
        try:
            while True:
                request = self.youtube.playlists().list(
                    channelId=channel_id,
                    part='snippet,contentDetails',
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                playlists.extend(response.get('items', []))
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            logger.info(f"Retrieved {len(playlists)} playlists for channel: {channel_id}")
        except Exception as e:
            logger.error(f"Error fetching playlists: {e}")
        
        return playlists
    
    def get_video_comments(self, video_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get comments from a video"""
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                request = self.youtube.commentThreads().list(
                    videoId=video_id,
                    part='snippet',
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    textFormat='plainText'
                )
                response = request.execute()
                comments.extend(response.get('items', []))
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            logger.info(f"Retrieved {len(comments)} comments for video: {video_id}")
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")
        
        return comments


class BigQueryLoader:
    """Load data into BigQuery tables"""
    
    def __init__(self, project_id: str, dataset_id: str, credentials_path: str = None):
        """Initialize BigQuery client"""
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)
        logger.info(f"BigQuery client initialized - Project: {project_id}, Dataset: {dataset_id}")
    
    def load_raw_videos(self, raw_videos: List[Dict[str, Any]]) -> int:
        """Load raw video data to raw_videos table"""
        if not raw_videos:
            logger.warning("No raw videos to load")
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.raw_videos"
        rows_to_insert = []
        
        for video in raw_videos:
            rows_to_insert.append({
                'id': video.get('id'),
                'raw': json.dumps(video),
                'ingestion_time': datetime.utcnow().isoformat()
            })
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Errors inserting raw_videos: {errors}")
            else:
                logger.info(f"Successfully loaded {len(rows_to_insert)} raw videos")
            return len(rows_to_insert)
        except Exception as e:
            logger.error(f"Error loading raw_videos: {e}")
            return 0
    
    def load_raw_channels(self, raw_channels: List[Dict[str, Any]]) -> int:
        """Load raw channel data to raw_channels table"""
        if not raw_channels:
            logger.warning("No raw channels to load")
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.raw_channels"
        rows_to_insert = []
        
        for channel in raw_channels:
            rows_to_insert.append({
                'id': channel.get('id'),
                'raw': json.dumps(channel),
                'ingestion_time': datetime.utcnow().isoformat()
            })
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Errors inserting raw_channels: {errors}")
            else:
                logger.info(f"Successfully loaded {len(rows_to_insert)} raw channels")
            return len(rows_to_insert)
        except Exception as e:
            logger.error(f"Error loading raw_channels: {e}")
            return 0
    
    def load_raw_playlists(self, raw_playlists: List[Dict[str, Any]]) -> int:
        """Load raw playlist data to raw_playlists table"""
        if not raw_playlists:
            logger.warning("No raw playlists to load")
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.raw_playlists"
        rows_to_insert = []
        
        for playlist in raw_playlists:
            rows_to_insert.append({
                'id': playlist.get('id'),
                'channel_id': playlist.get('snippet', {}).get('channelId'),
                'raw': json.dumps(playlist),
                'ingestion_time': datetime.utcnow().isoformat()
            })
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Errors inserting raw_playlists: {errors}")
            else:
                logger.info(f"Successfully loaded {len(rows_to_insert)} raw playlists")
            return len(rows_to_insert)
        except Exception as e:
            logger.error(f"Error loading raw_playlists: {e}")
            return 0
    
    def load_raw_comments(self, raw_comments: List[Dict[str, Any]], 
                         video_id: str, channel_id: str) -> int:
        """Load raw comment data to raw_comments table"""
        if not raw_comments:
            logger.warning("No raw comments to load")
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.raw_comments"
        rows_to_insert = []
        
        for comment in raw_comments:
            rows_to_insert.append({
                'id': comment.get('id'),
                'video_id': video_id,
                'channel_id': channel_id,
                'raw': json.dumps(comment),
                'ingestion_time': datetime.utcnow().isoformat()
            })
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Errors inserting raw_comments: {errors}")
            else:
                logger.info(f"Successfully loaded {len(rows_to_insert)} raw comments")
            return len(rows_to_insert)
        except Exception as e:
            logger.error(f"Error loading raw_comments: {e}")
            return 0
    
    def load_fact_videos(self, videos: List[Dict[str, Any]]) -> int:
        """Load transformed video data to fact_videos table"""
        if not videos:
            logger.warning("No videos to load")
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.fact_videos"
        rows_to_insert = []
        
        for video in videos:
            if SchemaValidator.validate_video(video):
                rows_to_insert.append(video)
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Errors inserting fact_videos: {errors}")
            else:
                logger.info(f"Successfully loaded {len(rows_to_insert)} fact videos")
            return len(rows_to_insert)
        except Exception as e:
            logger.error(f"Error loading fact_videos: {e}")
            return 0
    
    def load_dim_channels(self, channels: List[Dict[str, Any]]) -> int:
        """Load transformed channel data to dim_channels table"""
        if not channels:
            logger.warning("No channels to load")
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.dim_channels"
        rows_to_insert = []
        
        for channel in channels:
            if SchemaValidator.validate_channel(channel):
                rows_to_insert.append(channel)
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Errors inserting dim_channels: {errors}")
            else:
                logger.info(f"Successfully loaded {len(rows_to_insert)} dim channels")
            return len(rows_to_insert)
        except Exception as e:
            logger.error(f"Error loading dim_channels: {e}")
            return 0


class YouTubeAnalyticsPipeline:
    """Main pipeline to orchestrate crawling and loading"""
    
    def __init__(self, youtube_api_key: str, project_id: str, dataset_id: str, 
                 credentials_path: str = None):
        """Initialize pipeline"""
        self.crawler = YouTubeCrawler(youtube_api_key)
        self.loader = BigQueryLoader(project_id, dataset_id, credentials_path)
        self.transformer = YouTubeDataTransformer()
    
    def crawl_and_load_videos(self, video_ids: List[str]):
        """Crawl videos and load to BigQuery"""
        logger.info(f"Starting video crawl for {len(video_ids)} videos")
        
        # Get raw video data
        raw_videos = self.crawler.get_video_details(video_ids)
        
        # Load raw data
        self.loader.load_raw_videos(raw_videos)
        
        # Transform data
        transformed_videos = [
            self.transformer.transform_video(video) 
            for video in raw_videos
            if self.transformer.transform_video(video)
        ]
        
        # Load transformed data
        loaded = self.loader.load_fact_videos(transformed_videos)
        logger.info(f"Pipeline completed - Loaded {loaded} videos to fact_videos")
        
        return loaded
    
    def crawl_and_load_channels(self, channel_ids: List[str]):
        """Crawl channels and load to BigQuery"""
        logger.info(f"Starting channel crawl for {len(channel_ids)} channels")
        
        # Get raw channel data
        raw_channels = self.crawler.get_channel_details(channel_ids)
        
        # Load raw data
        self.loader.load_raw_channels(raw_channels)
        
        # Transform data
        transformed_channels = [
            self.transformer.transform_channel(channel)
            for channel in raw_channels
            if self.transformer.transform_channel(channel)
        ]
        
        # Load transformed data
        loaded = self.loader.load_dim_channels(transformed_channels)
        logger.info(f"Pipeline completed - Loaded {loaded} channels to dim_channels")
        
        return loaded
    
    def crawl_and_load_channel_content(self, channel_id: str):
        """Crawl all content from a channel: playlists and comments from videos"""
        logger.info(f"Starting content crawl for channel: {channel_id}")
        
        # Get playlists
        raw_playlists = self.crawler.get_channel_playlists(channel_id)
        self.loader.load_raw_playlists(raw_playlists)
        logger.info(f"Loaded {len(raw_playlists)} playlists")
        
        # Get first video from channel to get comments
        # This is just an example - in production, you'd iterate all videos
        if raw_playlists:
            logger.info(f"Content crawl completed for channel: {channel_id}")
    
    def search_and_load_videos(self, query: str, max_results: int = 10):
        """Search videos by query and load to BigQuery"""
        logger.info(f"Searching videos for query: {query}")
        
        video_ids = self.crawler.search_videos(query, max_results=max_results)
        if video_ids:
            return self.crawl_and_load_videos(video_ids)
        return 0


def main():
    """Main execution function"""
    # Load configuration from environment variables
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    project_id = os.getenv('GCP_PROJECT_ID')
    dataset_id = os.getenv('BQ_DATASET_ID')
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not youtube_api_key or not project_id or not dataset_id:
        raise ValueError("Missing required environment variables: YOUTUBE_API_KEY, GCP_PROJECT_ID, BQ_DATASET_ID")
    
    # Initialize pipeline
    pipeline = YouTubeAnalyticsPipeline(
        youtube_api_key=youtube_api_key,
        project_id=project_id,
        dataset_id=dataset_id,
        credentials_path=credentials_path
    )
    
    # Example: Search and load videos
    query = os.getenv('YOUTUBE_SEARCH_QUERY', 'machine learning')
    max_videos = int(os.getenv('MAX_VIDEOS', 20))
    
    logger.info(f"Starting YouTube Analytics Pipeline")
    logger.info(f"Search Query: {query}, Max Results: {max_videos}")
    
    try:
        loaded_videos = pipeline.search_and_load_videos(query, max_results=max_videos)
        logger.info(f"Pipeline execution completed - Total videos loaded: {loaded_videos}")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
