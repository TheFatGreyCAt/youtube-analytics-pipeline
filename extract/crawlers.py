import time
import logging
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import socket

from .config import YOUTUBE_API_KEY, get_active_channels, get_crawl_settings
from .db_manager import PostgresManager, BigQueryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

QUOTA_COSTS = {
    'channels.list': 1,
    'videos.list': 1,
    'playlistItems.list': 1,
    'playlists.list': 1,
    'commentThreads.list': 1,
}


class YouTubeCrawler:
    
    def __init__(self):
        socket.setdefaulttimeout(30)
        
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.pg_manager = PostgresManager()
        self.bq_manager = BigQueryManager()
        self.settings = get_crawl_settings()
        self.quota_used = 0
    
    def _check_quota_and_track(self, operation: str) -> bool:
        quota_status = self.pg_manager.get_api_quota_status()
        quota_used = quota_status['quota_used']
        daily_limit = quota_status['daily_limit']
        
        if quota_used >= daily_limit * 0.8:
            logging.warning(f"API Quota at {quota_status['percentage_used']:.1f}% ({quota_used}/{daily_limit})")
        
        if quota_used >= daily_limit * 0.9:
            logging.error(f"API Quota limit reached! {quota_used}/{daily_limit} units used")
            return False
        
        cost = QUOTA_COSTS.get(operation, 1)
        self.quota_used += cost
        
        return True
    
    def _update_quota_usage(self, cost: int = 1):
        try:
            self.pg_manager.update_api_quota(cost)
        except Exception as e:
            logging.warning(f"Failed to update quota tracking: {e}")
    
    def crawl_channel(self, channel_id: str) -> Dict:
        try:
            if not self._check_quota_and_track('channels.list'):
                raise Exception("API quota limit reached")
            
            request = self.youtube.channels().list(
                part='snippet,statistics,contentDetails,status',
                id=channel_id
            )
            response = request.execute()
            
            self._update_quota_usage(QUOTA_COSTS['channels.list'])
            
            if not response.get('items'):
                raise ValueError(f"Channel {channel_id} not found")
            
            self.bq_manager.insert_channel_raw(channel_id, response)
            logging.info(f"Crawled channel: {channel_id}")
            
            return response
        except HttpError as e:
            logging.error(f"YouTube API error for channel {channel_id}: {e}")
            raise
        except socket.timeout:
            logging.error(f"Timeout while crawling channel {channel_id}")
            raise
        except Exception as e:
            logging.error(f"Error crawling channel {channel_id}: {e}")
            raise
    
    def crawl_videos(self, channel_id: str, max_results: int | None = None) -> List[Dict]:
        if max_results is None:
            max_results = self.settings.get('max_videos_per_channel', 50)
        
        try:
            if not self._check_quota_and_track('channels.list'):
                raise Exception("API quota limit reached")
            
            channel_resp = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            self._update_quota_usage(QUOTA_COSTS['channels.list'])
            
            uploads_playlist_id = channel_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            video_ids = []
            next_page_token = None
            max_results = int(max_results) if max_results else 50
            
            while True:
                if not self._check_quota_and_track('playlistItems.list'):
                    logging.warning(f"Quota limit reached, stopping at {len(video_ids)} videos")
                    break
                
                playlist_request = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                playlist_response = playlist_request.execute()
                
                self._update_quota_usage(QUOTA_COSTS['playlistItems.list'])
                
                for item in playlist_response.get('items', []):
                    video_id = item['snippet']['resourceId']['videoId']
                    video_ids.append(video_id)
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or len(video_ids) >= max_results:
                    break
                
                time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            all_videos = []
            for i in range(0, len(video_ids), 50):
                if not self._check_quota_and_track('videos.list'):
                    logging.warning(f"Quota limit reached, stopping at {len(all_videos)} videos")
                    break
                
                batch = video_ids[i:i+50]
                videos_request = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails,status',
                    id=','.join(batch)
                )
                videos_response = videos_request.execute()
                
                self._update_quota_usage(QUOTA_COSTS['videos.list'])
                
                videos_list = videos_response.get('items', [])
                all_videos.extend(videos_list)
                
                self.bq_manager.insert_videos_raw(videos_list)
                
                time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            logging.info(f"Crawled {len(all_videos)} videos for channel {channel_id}")
            return all_videos
            
        except HttpError as e:
            logging.error(f"YouTube API error for videos {channel_id}: {e}")
            raise
        except socket.timeout:
            logging.error(f"Timeout while crawling videos for {channel_id}")
            raise
        except Exception as e:
            logging.error(f"Error crawling videos for {channel_id}: {e}")
            raise
    
    def crawl_playlists(self, channel_id: str, max_results: int = 50):
        try:
            if not self._check_quota_and_track('playlists.list'):
                logging.warning("Skipping playlists due to quota limit")
                return
            
            request = self.youtube.playlists().list(
                part='snippet,contentDetails,status',
                channelId=channel_id,
                maxResults=max_results
            )
            response = request.execute()
            
            self._update_quota_usage(QUOTA_COSTS['playlists.list'])
            
            playlists_list = response.get('items', [])
            self.bq_manager.insert_playlists_raw(channel_id, playlists_list)
            
            playlist_count = len(playlists_list)
            logging.info(f"Crawled {playlist_count} playlists for channel {channel_id}")
            
        except HttpError as e:
            logging.error(f"YouTube API error for playlists {channel_id}: {e}")
            raise
        except socket.timeout:
            logging.error(f"Timeout while crawling playlists for {channel_id}")
            raise
        except Exception as e:
            logging.error(f"Error crawling playlists for {channel_id}: {e}")
            raise
    
    def crawl_comments(self, video_id: str, channel_id: str, max_results: int | None = None):
        if max_results is None:
            max_results = self.settings.get('max_comments_per_video', 100)
        
        try:
            if not self._check_quota_and_track('commentThreads.list'):
                logging.warning("Skipping comments due to quota limit")
                return
            
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_results,
                order='relevance'
            )
            response = request.execute()
            
            self._update_quota_usage(QUOTA_COSTS['commentThreads.list'])
            
            comments_list = response.get('items', [])
            self.bq_manager.insert_comments_raw(video_id, channel_id, comments_list)
            
            comment_count = len(comments_list)
            logging.info(f"Crawled {comment_count} comments for video {video_id}")
            
        except HttpError as e:
            if 'commentsDisabled' in str(e):
                logging.warning(f"Comments disabled for video {video_id}")
            else:
                logging.error(f"YouTube API error for comments {video_id}: {e}")
        except socket.timeout:
            logging.error(f"Timeout while crawling comments for {video_id}")
        except Exception as e:
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
                logging.info(f"Crawling comments for {len(recent_videos)} recent videos")
                for video in recent_videos:
                    video_id = video.get('id', '')
                    if video_id:
                        self.crawl_comments(video_id, channel_id)
                        time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            execution_time = time.time() - start_time
            self.pg_manager.update_crawl_success(channel_id, video_count, execution_time)
            
            logging.info(f"Completed crawl: {video_count} videos in {execution_time:.2f}s, quota used: ~{self.quota_used} units")
            
            return video_count
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to crawl channel {channel_id}: {error_msg}")
            self.pg_manager.update_crawl_failed(channel_id, error_msg)
            raise


def crawl_from_config_file(limit: int | None = None):
    crawler = YouTubeCrawler()
    channels = get_active_channels()
    
    if not channels:
        logging.info("No active channels found in channels.yml")
        logging.info("Please edit channels.yml to add channels")
        return
    
    if limit:
        channels = channels[:limit]
    
    logging.info(f"Found {len(channels)} active channel(s) to crawl")
    
    quota_status = crawler.pg_manager.get_api_quota_status()
    logging.info(f"Current API Quota: {quota_status['quota_used']}/{quota_status['daily_limit']} units ({quota_status['percentage_used']:.1f}%)")
    
    if quota_status['percentage_used'] > 90:
        logging.error("API quota usage is too high! Please try again tomorrow.")
        return
    
    success_count = 0
    failed_count = 0
    
    for i, channel_config in enumerate(channels, 1):
        channel_id = channel_config.get('id')
        channel_name = channel_config.get('name', channel_id)
        include_comments = channel_config.get('include_comments', False)
        
        try:
            logging.info(f"[{i}/{len(channels)}] Crawling channel: {channel_name} ({channel_id})")
            crawler.crawl_channel_full(channel_id, include_comments)
            success_count += 1
            
        except Exception as e:
            logging.error(f"Failed to crawl {channel_name}: {e}")
            failed_count += 1
            continue
    
    final_quota = crawler.pg_manager.get_api_quota_status()
    logging.info(f"Crawl session completed - Success: {success_count}/{len(channels)}, Failed: {failed_count}/{len(channels)}")
    logging.info(f"Total quota used: ~{crawler.quota_used} units")
    logging.info(f"Final API Quota: {final_quota['quota_used']}/{final_quota['daily_limit']} units ({final_quota['percentage_used']:.1f}%)")


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
