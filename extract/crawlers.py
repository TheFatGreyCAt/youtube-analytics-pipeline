import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
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
    
    def crawl_videos_incremental(
        self, 
        channel_id: str, 
        published_after: Optional[datetime] = None,
        max_results: int = 50
    ) -> List[Dict]:
        try:
            if published_after is None:
                published_after = self.pg_manager.get_last_crawled_video_date(channel_id)
            
            if published_after is None:
                logging.info(f"No previous crawl date found for {channel_id}, doing full crawl")
                return self.crawl_videos(channel_id, max_results)
            
            published_after_str = published_after.strftime('%Y-%m-%dT%H:%M:%SZ')
            logging.info(f"Incremental crawl: Getting videos published after {published_after_str}")
            
            if not self._check_quota_and_track('channels.list'):
                raise Exception("API quota limit reached")
            
            channel_resp = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            self._update_quota_usage(QUOTA_COSTS['channels.list'])
            
            if not channel_resp.get('items'):
                raise ValueError(f"Channel {channel_id} not found")
            
            uploads_playlist_id = channel_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            video_ids = []
            next_page_token = None
            videos_checked = 0
            
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
                    videos_checked += 1
                    published_at_str = item['snippet'].get('publishedAt')
                    
                    if published_at_str:
                        video_published = datetime.strptime(published_at_str, '%Y-%m-%dT%H:%M:%SZ')
                        
                        if video_published > published_after:
                            video_id = item['snippet']['resourceId']['videoId']
                            video_ids.append(video_id)
                        else:
                            logging.info(f"Reached videos older than cutoff date, stopping")
                            next_page_token = None
                            break
                
                if not next_page_token or len(video_ids) >= max_results:
                    break
                
                next_page_token = playlist_response.get('nextPageToken')
                time.sleep(self.settings.get('api_delay_seconds', 0.5))
            
            if not video_ids:
                logging.info(f"No new videos found after {published_after_str}")
                return []
            
            logging.info(f"Found {len(video_ids)} new videos (checked {videos_checked} total)")
            
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
            
            if all_videos:
                latest_video = max(
                    all_videos,
                    key=lambda v: v['snippet'].get('publishedAt', '')
                )
                latest_published = datetime.strptime(
                    latest_video['snippet']['publishedAt'],
                    '%Y-%m-%dT%H:%M:%SZ'
                )
                self.pg_manager.update_last_video_date(channel_id, latest_published)
                logging.info(f"Updated last video date to: {latest_published}")
            
            logging.info(f"Incremental crawl completed: {len(all_videos)} new videos")
            return all_videos
            
        except HttpError as e:
            logging.error(f"YouTube API error for incremental videos {channel_id}: {e}")
            raise
        except socket.timeout:
            logging.error(f"Timeout while crawling incremental videos for {channel_id}")
            raise
        except Exception as e:
            logging.error(f"Error in incremental crawl for {channel_id}: {e}")
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
            
            logging.info(f"Completed crawl: {video_count} videos in {execution_time:.2f}s, quota used: {self.quota_used} units")
            return video_count
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to crawl channel {channel_id}: {error_msg}")
            self.pg_manager.update_crawl_failed(channel_id, error_msg)
            raise
    
    def crawl_channel_smart(
        self, 
        channel_id: str, 
        include_comments: bool = False,
        force_full: bool = False
    ) -> int:
        start_time = time.time()
        
        try:
            should_do_full = force_full or self.pg_manager.should_full_crawl(channel_id)
            
            if should_do_full:
                logging.info(f"FULL CRAWL for channel {channel_id}")
                return self.crawl_channel_full(channel_id, include_comments)
            else:
                logging.info(f"INCREMENTAL CRAWL for channel {channel_id}")
                
                self.crawl_channel(channel_id)
                videos = self.crawl_videos_incremental(channel_id)
                video_count = len(videos)
                
                logging.info(f"Skipping playlists (incremental mode)")
                
                if include_comments and videos:
                    recent_videos = videos[:5]
                    logging.info(f"Crawling comments for {len(recent_videos)} new videos")
                    for video in recent_videos:
                        video_id = video.get('id', '')
                        if video_id:
                            self.crawl_comments(video_id, channel_id)
                            time.sleep(self.settings.get('api_delay_seconds', 0.5))
                
                execution_time = time.time() - start_time
                self.pg_manager.update_crawl_success(channel_id, video_count, execution_time)
                
                logging.info(f"Incremental crawl completed: {video_count} videos in {execution_time:.2f}s, quota used: {self.quota_used} units")
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
        return
    
    if limit:
        channels = channels[:limit]
    
    logging.info(f"Found {len(channels)} active channel(s) to crawl")
    
    quota_status = crawler.pg_manager.get_api_quota_status()
    logging.info(f"Current API Quota: {quota_status['quota_used']}/{quota_status['daily_limit']} units ({quota_status['percentage_used']:.1f}%)")
    
    if quota_status['percentage_used'] > 90:
        logging.error("API quota usage is too high")
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
    
    final_quota = crawler.pg_manager.get_api_quota_status()
    logging.info(f"Crawl session completed - Success: {success_count}/{len(channels)}, Failed: {failed_count}/{len(channels)}")
    logging.info(f"Total quota used: {crawler.quota_used} units")
    logging.info(f"Final API Quota: {final_quota['quota_used']}/{final_quota['daily_limit']} units ({final_quota['percentage_used']:.1f}%)")


def crawl_scheduled_channels(limit: int = 10, include_comments: bool = False):
    crawler = YouTubeCrawler()
    channels = crawler.pg_manager.get_channels_to_crawl(limit)
    
    if not channels:
        logging.info("No channels scheduled for crawling")
        return
    
    logging.info(f"Found {len(channels)} channels to crawl")
    
    for channel_id in channels:
        try:
            crawler.crawl_channel_full(channel_id, include_comments)
        except Exception as e:
            logging.error(f"Failed to crawl {channel_id}: {e}")


def crawl_scheduled_channels_smart(limit: int = 10, include_comments: bool = False):
    crawler = YouTubeCrawler()
    channels = crawler.pg_manager.get_channels_to_crawl(limit)
    
    if not channels:
        logging.info("No channels scheduled for crawling")
        return
    
    logging.info(f"Found {len(channels)} channels to crawl")
    
    success_count = 0
    full_count = 0
    incremental_count = 0
    
    for channel_id in channels:
        try:
            is_full = crawler.pg_manager.should_full_crawl(channel_id)
            crawler.crawl_channel_smart(channel_id, include_comments)
            success_count += 1
            
            if is_full:
                full_count += 1
            else:
                incremental_count += 1
        except Exception as e:
            logging.error(f"Failed to crawl {channel_id}: {e}")
    
    logging.info(f"Crawl completed: {success_count}/{len(channels)} successful")
    logging.info(f"Full crawls: {full_count}, Incremental crawls: {incremental_count}")
    logging.info(f"Total quota used: {crawler.quota_used} units")


if __name__ == "__main__":
    crawl_from_config_file(limit=10)
