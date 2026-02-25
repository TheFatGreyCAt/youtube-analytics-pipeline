"""
Module to search for YouTube channel IDs by channel name
Works with CSV files and integrates with PostgreSQL database
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import time
import logging
from .config import YOUTUBE_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChannelFinder:
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError("YouTube API key is required")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.quota_used = 0
    
    def calculate_priority(self, subscriber_count: int, video_count: int) -> int:
        try:
            subs = int(subscriber_count)
        except (ValueError, TypeError):
            subs = 0
        
        if subs >= 5_000_000:
            return 5
        elif subs >= 1_000_000:
            return 4
        elif subs >= 100_000:
            return 3
        elif subs >= 10_000:
            return 2
        else:
            return 1
    
    def calculate_crawl_frequency(self, subscriber_count: int, video_count: int) -> int:
        try:
            subs = int(subscriber_count)
            videos = int(video_count)
        except (ValueError, TypeError):
            return 24
        
        upload_frequency = videos / 365 if videos > 0 else 0
        
        if subs >= 1_000_000 or upload_frequency > 1:
            return 12
        elif subs >= 100_000:
            return 24
        else:
            return 48
    
    def get_channel_details(self, channel_id: str) -> Optional[Dict]:
        try:
            request = self.youtube.channels().list(
                part='snippet,statistics,contentDetails',
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
            
            item = response['items'][0]
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            
            return {
                'channel_id': channel_id,
                'channel_name': snippet.get('title', ''),
                'description': snippet.get('description', '')[:200],
                'custom_url': snippet.get('customUrl', ''),
                'published_at': snippet.get('publishedAt', ''),
                'country': snippet.get('country', ''),
                'subscriber_count': statistics.get('subscriberCount', '0'),
                'video_count': statistics.get('videoCount', '0'),
                'view_count': statistics.get('viewCount', '0'),
                'thumbnail_url': snippet.get('thumbnails', {}).get('default', {}).get('url', '')
            }
        except Exception as e:
            logging.error(f"Error getting channel details for '{channel_id}': {e}")
            return None
    
    def search_channel(self, channel_name: str, max_results: int = 5) -> Optional[Tuple[str, str]]:
        try:
            request = self.youtube.search().list(
                part='snippet',
                q=channel_name,
                type='channel',
                maxResults=max_results
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
            
            for item in response['items']:
                found_name = item['snippet']['title']
                channel_id = item['snippet']['channelId']
                
                if channel_name.lower() == found_name.lower():
                    return channel_id, found_name
                
                if channel_name.lower() in found_name.lower():
                    return channel_id, found_name
            
            first_result = response['items'][0]
            return first_result['snippet']['channelId'], first_result['snippet']['title']
        except HttpError as e:
            logging.error(f"HTTP Error searching for '{channel_name}': {e}")
            return None
        except Exception as e:
            logging.error(f"Error searching for '{channel_name}': {e}")
            return None
    
    def verify_channel_id(self, channel_id: str) -> Optional[str]:
        try:
            request = self.youtube.channels().list(
                part='snippet',
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return None
            
            return response['items'][0]['snippet']['title']
        except Exception as e:
            logging.error(f"Error verifying channel ID '{channel_id}': {e}")
            return None
    
    def search_from_csv(self, input_csv: str, output_csv: Optional[str] = None, 
                       delay: float = 0.3) -> List[Dict]:
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_csv}")
        
        results = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            channels = list(reader)
        
        logging.info(f"Found {len(channels)} channels to search")
        
        for idx, row in enumerate(channels, 1):
            num_id = row.get('num_id', idx)
            channel_name = row.get('channel_name', '').strip()
            
            if not channel_name:
                results.append({
                    'num_id': num_id,
                    'status': 'SKIPPED',
                    'search_name': '',
                    'channel_id': '',
                    'channel_name': '',
                    'custom_url': '',
                    'subscriber_count': '',
                    'video_count': '',
                    'view_count': '',
                    'country': '',
                    'published_at': '',
                    'note': 'Empty channel name'
                })
                continue
            
            logging.info(f"[{idx}/{len(channels)}] Searching: {channel_name}")
            result = self.search_channel(channel_name)
            
            if result:
                channel_id, found_name = result
                logging.info(f"  Found: {found_name} ({channel_id})")
                
                details = self.get_channel_details(channel_id)
                
                if details:
                    logging.info(f"  Subs: {int(details['subscriber_count']):,}, Videos: {int(details['video_count']):,}")
                    
                    results.append({
                        'num_id': num_id,
                        'status': 'FOUND',
                        'search_name': channel_name,
                        'channel_id': details['channel_id'],
                        'channel_name': details['channel_name'],
                        'custom_url': details['custom_url'],
                        'subscriber_count': details['subscriber_count'],
                        'video_count': details['video_count'],
                        'view_count': details['view_count'],
                        'country': details['country'],
                        'published_at': details['published_at'],
                        'description': details['description'],
                        'thumbnail_url': details['thumbnail_url'],
                        'note': 'Successfully found'
                    })
                else:
                    results.append({
                        'num_id': num_id,
                        'status': 'FOUND_NO_DETAILS',
                        'search_name': channel_name,
                        'channel_id': channel_id,
                        'channel_name': found_name,
                        'custom_url': '',
                        'subscriber_count': '',
                        'video_count': '',
                        'view_count': '',
                        'country': '',
                        'published_at': '',
                        'note': 'Channel found but details unavailable'
                    })
            else:
                logging.info(f"  Not found")
                results.append({
                    'num_id': num_id,
                    'status': 'NOT_FOUND',
                    'search_name': channel_name,
                    'channel_id': '',
                    'channel_name': '',
                    'custom_url': '',
                    'subscriber_count': '',
                    'video_count': '',
                    'view_count': '',
                    'country': '',
                    'published_at': '',
                    'note': 'No matching channel found'
                })
            
            time.sleep(delay)
        
        found_count = sum(1 for r in results if r['status'].startswith('FOUND'))
        not_found_count = sum(1 for r in results if r['status'] == 'NOT_FOUND')
        
        logging.info(f"Search completed - Found: {found_count}/{len(results)}, Not found: {not_found_count}/{len(results)}")
        
        if output_csv:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                fieldnames = [
                    'num_id', 'status', 'search_name', 'channel_id', 'channel_name',
                    'custom_url', 'subscriber_count', 'video_count', 'view_count',
                    'country', 'published_at', 'description', 'thumbnail_url', 'note'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            logging.info(f"Results saved to: {output_csv}")
            
            simple_output = output_path.parent / f"{output_path.stem}_simple.csv"
            with open(simple_output, 'w', encoding='utf-8', newline='') as f:
                simple_fieldnames = [
                    'channel_id', 'channel_name', 'subscriber_count', 
                    'video_count', 'view_count', 'country'
                ]
                writer = csv.DictWriter(f, fieldnames=simple_fieldnames)
                writer.writeheader()
                for row in results:
                    if row['status'].startswith('FOUND'):
                        writer.writerow({k: row.get(k, '') for k in simple_fieldnames})
            
            logging.info(f"Simplified results saved to: {simple_output}")
        
        return results
    
    def add_channels_to_database(
        self, 
        channels: List[Dict],
        pg_manager,
        default_frequency: Optional[int] = None,
        default_priority: Optional[int] = None,
        update_existing: bool = False
    ) -> Dict:
        if not channels:
            logging.warning("No channels to add to database")
            return {'added': 0, 'skipped': 0, 'updated': 0, 'failed': 0}
        
        logging.info(f"Adding {len(channels)} channels to database")
        
        channel_ids = [ch.get('channel_id') for ch in channels if ch.get('channel_id')]
        existing_ids = pg_manager.get_existing_channels(channel_ids)
        
        summary = {'added': 0, 'skipped': 0, 'updated': 0, 'failed': 0}
        channels_to_add = []
        
        for channel in channels:
            channel_id = channel.get('channel_id')
            if not channel_id:
                logging.warning(f"Skipping channel without ID: {channel.get('channel_name')}")
                summary['failed'] += 1
                continue
            
            channel_name = channel.get('channel_name', 'Unknown')
            
            if channel_id in existing_ids:
                if update_existing:
                    try:
                        priority = default_priority or self.calculate_priority(
                            channel.get('subscriber_count', 0),
                            channel.get('video_count', 0)
                        )
                        frequency = default_frequency or self.calculate_crawl_frequency(
                            channel.get('subscriber_count', 0),
                            channel.get('video_count', 0)
                        )
                        
                        pg_manager.add_channel(channel_id, channel_name, frequency)
                        logging.info(f"  Updated: {channel_name} (Priority: {priority}, Frequency: {frequency}h)")
                        summary['updated'] += 1
                    except Exception as e:
                        logging.error(f"  Failed to update {channel_name}: {e}")
                        summary['failed'] += 1
                else:
                    logging.info(f"  Skipped (exists): {channel_name}")
                    summary['skipped'] += 1
                continue
            
            try:
                priority = default_priority or self.calculate_priority(
                    channel.get('subscriber_count', 0),
                    channel.get('video_count', 0)
                )
                frequency = default_frequency or self.calculate_crawl_frequency(
                    channel.get('subscriber_count', 0),
                    channel.get('video_count', 0)
                )
                
                channels_to_add.append((channel_id, channel_name, frequency))
                logging.info(f"  Will add: {channel_name} (Subs: {int(channel.get('subscriber_count', 0)):,}, Priority: {priority}, Frequency: {frequency}h)")
            except Exception as e:
                logging.error(f"  Error preparing {channel_name}: {e}")
                summary['failed'] += 1
        
        if channels_to_add:
            try:
                pg_manager.add_channels_batch(channels_to_add)
                summary['added'] = len(channels_to_add)
                logging.info(f"Successfully added {summary['added']} new channels to database")
            except Exception as e:
                logging.error(f"Failed to batch insert channels: {e}")
                summary['failed'] += len(channels_to_add)
                summary['added'] = 0
        
        logging.info(f"Database import summary - Added: {summary['added']}, Updated: {summary['updated']}, Skipped: {summary['skipped']}, Failed: {summary['failed']}")
        
        return summary
    
    def search_and_add_from_csv(
        self,
        input_csv: str,
        pg_manager,
        output_csv: Optional[str] = None,
        delay: float = 0.3,
        default_frequency: Optional[int] = None,
        update_existing: bool = False
    ) -> Dict:
        search_results = self.search_from_csv(input_csv, output_csv, delay)
        
        valid_channels = [
            r for r in search_results 
            if r['status'].startswith('FOUND') and r['channel_id']
        ]
        
        if not valid_channels:
            logging.warning("No valid channels found to add to database")
            return {'added': 0, 'skipped': 0, 'updated': 0, 'failed': len(search_results)}
        
        db_summary = self.add_channels_to_database(
            valid_channels,
            pg_manager,
            default_frequency=default_frequency,
            update_existing=update_existing
        )
        
        logging.info(f"Estimated API quota used: {self.quota_used} units")
        
        try:
            pg_manager.update_api_quota(self.quota_used)
        except Exception as e:
            logging.warning(f"Failed to update quota tracking: {e}")
        
        return db_summary


def search_single_channel(channel_name: str, api_key: Optional[str] = None) -> Optional[Dict]:
    finder = ChannelFinder(api_key)
    result = finder.search_channel(channel_name)
    
    if result:
        channel_id, found_name = result
        details = finder.get_channel_details(channel_id)
        return details
    
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        logging.info("Usage:")
        logging.info("  python -m extract.channel_finder <channel_name>")
        logging.info("  python -m extract.channel_finder --csv <input.csv> [output.csv]")
        sys.exit(1)
    
    if sys.argv[1] == '--csv':
        if len(sys.argv) < 3:
            logging.error("Please provide input CSV file")
            sys.exit(1)
        
        input_csv = sys.argv[2]
        output_csv = sys.argv[3] if len(sys.argv) > 3 else None
        
        if not output_csv:
            input_path = Path(input_csv)
            output_csv = str(input_path.parent / f"{input_path.stem}_found.csv")
        
        finder = ChannelFinder()
        finder.search_from_csv(input_csv, output_csv)
    else:
        channel_name = ' '.join(sys.argv[1:])
        logging.info(f"Searching for channel: {channel_name}")
        
        details = search_single_channel(channel_name)
        
        if details:
            logging.info(f"Found: {details['channel_name']}")
            logging.info(f"  ID: {details['channel_id']}")
            logging.info(f"  Subscribers: {int(details['subscriber_count']):,}")
            logging.info(f"  Videos: {int(details['video_count']):,}")
            logging.info(f"  Views: {int(details['view_count']):,}")
            if details['country']:
                logging.info(f"  Country: {details['country']}")
        else:
            logging.error(f"Channel not found: {channel_name}")
