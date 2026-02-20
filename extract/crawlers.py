"""
YouTube Crawlers - Main crawling logic
Extracts data from YouTube API and saves to BigQuery
PostgreSQL only used for scheduling metadata
"""
import time
import logging
from typing import List, Dict
from googleapiclient.discovery import build

from .config import YOUTUBE_API_KEY, get_active_channels, get_crawl_settings
from .db_manager import PostgresManager, BigQueryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class YouTubeCrawler:
    
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.pg_manager = PostgresManager()
        self.bq_manager = BigQueryManager()
        self.settings = get_crawl_settings()
    
    def crawl_channel(self, channel_id: str) -> Dict:
        try:
            request = self.youtube.channels().list(
                part='snippet,statistics,contentDetails,status',
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                raise ValueError(f"Channel {channel_id} not found")
            
            self.bq_manager.insert_channel_raw(channel_id, response)
            logging.info(f"Crawled channel: {channel_id}")
            
            return response
        except Exception as e:
            logging.error(f"Error crawling channel {channel_id}: {e}")
            raise
    
    def crawl_videos(self, channel_id: str, max_results: int = None) -> List[Dict]:
        if max_results is None:
            max_results = self.settings.get('max_videos_per_channel', 50)
        
        try:
            channel_resp = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            uploads_playlist_id = channel_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            video_ids = []
            next_page_token = None
            
            while True:
                playlist_request = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                playlist_response = playlist_request.execute()
                
                for item in playlist_response.get('items', []):
                    video_id = item['snippet']['resourceId']['videoId']
                    video_ids.append(video_id)
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or len(video_ids) >= max_results:
                    break
                
                time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            all_videos = []
            for i in range(0, len(video_ids), 50):
                batch = video_ids[i:i+50]
                videos_request = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch)
                )
                videos_response = videos_request.execute()
                
                videos_list = videos_response.get('items', [])
                all_videos.extend(videos_list)
                
                self.bq_manager.insert_videos_raw(videos_list)
                
                time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            logging.info(f"Crawled {len(all_videos)} videos for channel {channel_id}")
            return all_videos
            
        except Exception as e:
            logging.error(f"Error crawling videos for {channel_id}: {e}")
            raise
    
    def crawl_playlists(self, channel_id: str, max_results: int = 50):
        try:
            request = self.youtube.playlists().list(
                part='snippet,contentDetails,status',
                channelId=channel_id,
                maxResults=max_results
            )
            response = request.execute()
            
            playlists_list = response.get('items', [])
            self.bq_manager.insert_playlists_raw(channel_id, playlists_list)
            
            playlist_count = len(playlists_list)
            logging.info(f"Crawled {playlist_count} playlists for channel {channel_id}")
            
        except Exception as e:
            logging.error(f"Error crawling playlists for {channel_id}: {e}")
            raise
    
    def crawl_comments(self, video_id: str, channel_id: str, max_results: int = None):
        if max_results is None:
            max_results = self.settings.get('max_comments_per_video', 100)
        
        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_results,
                order='relevance'
            )
            response = request.execute()
            
            comments_list = response.get('items', [])
            self.bq_manager.insert_comments_raw(video_id, channel_id, comments_list)
            
            comment_count = len(comments_list)
            logging.info(f"Crawled {comment_count} comments for video {video_id}")
            
        except Exception as e:
            if 'commentsDisabled' in str(e):
                logging.warning(f"Comments disabled for video {video_id}")
            else:
                logging.error(f"Error crawling comments for {video_id}: {e}")
    
    def crawl_channel_full(self, channel_id: str, include_comments: bool = False) -> int:

        start_time = time.time()
        
        try:
            logging.info(f"Starting full crawl for channel {channel_id}")
            
            self.crawl_channel(channel_id)
            
            videos = self.crawl_videos(channel_id)
            video_count = len(videos)
            
            self.crawl_playlists(channel_id)
            
            if include_comments and videos:
                recent_videos = videos[:10]
                for video in recent_videos:
                    video_id = video.get('id', '')
                    if video_id:
                        self.crawl_comments(video_id, channel_id)
                        time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            execution_time = time.time() - start_time
            
            self.pg_manager.update_crawl_success(channel_id, video_count, execution_time)
            
            logging.info(f"Completed crawl for channel {channel_id}: {video_count} videos in {execution_time:.2f}s")
            return video_count
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to crawl channel {channel_id}: {error_msg}")
            
            self.pg_manager.update_crawl_failed(channel_id, error_msg)
            
            raise


def crawl_from_config_file(limit: int = None):

    crawler = YouTubeCrawler()
    channels = get_active_channels()
    
    if not channels:
        logging.info("No active channels found in channels.yml")
        logging.info("Please edit channels.yml to add channels")
        return
    
    if limit:
        channels = channels[:limit]
    
    logging.info(f"Found {len(channels)} active channel(s) to crawl")
    
    for i, channel_config in enumerate(channels, 1):
        channel_id = channel_config.get('id')
        channel_name = channel_config.get('name', channel_id)
        include_comments = channel_config.get('include_comments', False)
        
        try:
            logging.info(f"\n[{i}/{len(channels)}] Crawling: {channel_name}")
            crawler.crawl_channel_full(channel_id, include_comments)
        except Exception as e:
            logging.error(f"Failed to crawl {channel_name}: {e}")
            continue


def crawl_scheduled_channels(limit: int = 10, include_comments: bool = False):
    crawler = YouTubeCrawler()
    
    channels = crawler.pg_manager.get_channels_to_crawl(limit)
    
    if not channels:
        logging.info("No channels scheduled for crawling in database")
        logging.info("Tip: Use channels.yml file instead with: python -m extract.cli crawl-file")
        return
    
    logging.info(f"Found {len(channels)} channels to crawl from database")
    
    for channel_id in channels:
        try:
            crawler.crawl_channel_full(channel_id, include_comments)
        except Exception as e:
            logging.error(f"Failed to crawl {channel_id}: {e}")
            continue


if __name__ == "__main__":
    crawl_from_config_file(limit=10)
