import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import isodate
import logging

logger = logging.getLogger(__name__)


class YouTubeDataTransformer:
    """Transform YouTube API data to fact_videos and dim_channels tables"""
    
    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert to int, return None if fails"""
        try:
            return int(value) if value else None
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert to float, return None if fails"""
        try:
            return float(value) if value else None
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_duration(iso_duration: str) -> int:
        """Convert ISO8601 duration to seconds"""
        try:
            return int(isodate.parse_duration(iso_duration).total_seconds())
        except Exception:
            return 0
    
    @staticmethod
    def _calculate_engagement_rate(likes: int, comments: int, views: int) -> float:
        """Calculate engagement rate: (likes + comments) / views"""
        if not views or views == 0:
            return 0.0
        return round((likes + comments) / views, 6)
    
    @classmethod
    def transform_video(cls, api_video: Dict[str, Any]) -> Dict[str, Any]:

        try:
            snippet = api_video.get('snippet', {})
            stats = api_video.get('statistics', {})
            content = api_video.get('contentDetails', {})
            
            # Extract core values
            video_id = api_video.get('id')
            channel_id = snippet.get('channelId')
            published_at = snippet.get('publishedAt')
            title = snippet.get('title', '')
            
            # Extract statistics
            view_count = cls._safe_int(stats.get('viewCount'))
            like_count = cls._safe_int(stats.get('likeCount'))
            comment_count = cls._safe_int(stats.get('commentCount'))
            
            # Parse duration
            duration_seconds = cls._parse_duration(content.get('duration', 'PT0S'))
            
            # Category and title length
            category_id = cls._safe_int(snippet.get('categoryId'))
            title_length = len(title)
            
            # Calculate engagement rate
            engagement_rate = cls._calculate_engagement_rate(
                like_count or 0,
                comment_count or 0,
                view_count or 1
            )
            
            return {
                'video_id': video_id,
                'channel_id': channel_id,
                'published_at': published_at,
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'duration_seconds': duration_seconds,
                'category_id': category_id,
                'title_length': title_length,
                'engagement_rate': engagement_rate,
                'title': title,  # Extra for reference
                'ingestion_time': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error transforming video {api_video.get('id')}: {e}")
            return {}
    
    @classmethod
    def transform_channel(cls, api_channel: Dict[str, Any]) -> Dict[str, Any]:

        try:
            snippet = api_channel.get('snippet', {})
            stats = api_channel.get('statistics', {})
            
            channel_id = api_channel.get('id')
            title = snippet.get('title', '')
            subscriber_count = cls._safe_int(stats.get('subscriberCount'))
            country = snippet.get('country')
            published_at = snippet.get('publishedAt')
            
            return {
                'channel_id': channel_id,
                'title': title,
                'subscriber_count': subscriber_count,
                'country': country,
                'published_at': published_at,
                'ingestion_time': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error transforming channel {api_channel.get('id')}: {e}")
            return {}
    
    @classmethod
    def transform_playlist(cls, api_playlist: Dict[str, Any]) -> Dict[str, Any]:
        """Transform playlist API response"""
        try:
            snippet = api_playlist.get('snippet', {})
            content = api_playlist.get('contentDetails', {})
            
            return {
                'playlist_id': api_playlist.get('id'),
                'channel_id': snippet.get('channelId'),
                'title': snippet.get('title'),
                'item_count': cls._safe_int(content.get('itemCount')),
                'published_at': snippet.get('publishedAt'),
                'ingestion_time': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error transforming playlist {api_playlist.get('id')}: {e}")
            return {}


class SchemaValidator:
    
    FACT_VIDEO_SCHEMA = {
        'video_id': str,
        'channel_id': str,
        'published_at': str,
        'view_count': (int, type(None)),
        'like_count': (int, type(None)),
        'comment_count': (int, type(None)),
        'duration_seconds': int,
        'category_id': (int, type(None)),
        'title_length': int,
        'engagement_rate': float,
    }
    
    DIM_CHANNEL_SCHEMA = {
        'channel_id': str,
        'title': str,
        'subscriber_count': (int, type(None)),
        'country': (str, type(None)),
        'published_at': str,
    }
    
    @classmethod
    def validate_video(cls, record: Dict[str, Any]) -> bool:
        """Validate fact_video record"""
        for field, expected_type in cls.FACT_VIDEO_SCHEMA.items():
            if field not in record:
                logger.warning(f"Missing field: {field}")
                return False
            if not isinstance(record[field], expected_type):
                logger.warning(f"Invalid type for {field}: expected {expected_type}")
                return False
        return True
    
    @classmethod
    def validate_channel(cls, record: Dict[str, Any]) -> bool:
        """Validate dim_channel record"""
        for field, expected_type in cls.DIM_CHANNEL_SCHEMA.items():
            if field not in record:
                logger.warning(f"Missing field: {field}")
                return False
            if not isinstance(record[field], expected_type):
                logger.warning(f"Invalid type for {field}: expected {expected_type}")
                return False
        return True