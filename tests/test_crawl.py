import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from extract.crawlers import YouTubeCrawler
from extract.config import validate_config, YOUTUBE_API_KEY, GCP_PROJECT_ID

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_crawl_channel(channel_id: str = 'UCXuqSBlHAE6Xw-yeJA0Tunw'):
    try:
        validate_config()
        print(f"[OK] Config validated | API Key: {YOUTUBE_API_KEY[:10]}... | GCP: {GCP_PROJECT_ID}")
    except Exception as e:
        print(f"[ERROR] Config error: {e}")
        return

    print(f"Target Channel: {channel_id}")

    try:
        crawler = YouTubeCrawler()

        channel_data = crawler.crawl_channel(channel_id)
        if channel_data and channel_data.get('items'):
            item = channel_data['items'][0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            print(f"[OK] {snippet.get('title')} | Subs: {stats.get('subscriberCount')} | Videos: {stats.get('videoCount')}")

        videos = crawler.crawl_videos(channel_id, max_results=10)
        print(f"[OK] Crawled {len(videos)} videos")

        if videos:
            for i, video in enumerate(videos[:3], 1):
                title = video.get('snippet', {}).get('title', 'N/A')
                views = video.get('statistics', {}).get('viewCount', '0')
                print(f"   {i}. {title[:60]} (Views: {views})")

        crawler.crawl_playlists(channel_id, max_results=10)
        print("[OK] Playlists crawled")

        if videos:
            video_id = videos[0].get('id', '')
            if video_id:
                try:
                    crawler.crawl_comments(video_id, channel_id, max_results=10)
                    print("[OK] Comments crawled")
                except Exception as e:
                    print(f"[WARNING] Comments: {e}")

        print("\nData saved to BigQuery dataset: raw_yt (raw_channels, raw_videos, raw_playlists, raw_comments)")

    except Exception as e:
        print(f"[ERROR] CRAWL FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_simple_api():
    try:
        from googleapiclient.discovery import build

        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY not found in .env")
            return

        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        channel_id = 'UCXuqSBlHAE6Xw-yeJA0Tunw'

        request = youtube.channels().list(part='snippet,statistics', id=channel_id)
        response = request.execute()

        if response.get('items'):
            item = response['items'][0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            print(f"[OK] {snippet.get('title')} | Subs: {stats.get('subscriberCount')} | Videos: {stats.get('videoCount')}")
        else:
            print("[ERROR] No data returned from API")

    except Exception as e:
        print(f"[ERROR] API Test Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test YouTube Crawler')
    parser.add_argument('--simple', action='store_true', help='Run simple API test only (no database)')
    parser.add_argument('--channel', default='UCXuqSBlHAE6Xw-yeJA0Tunw', help='Channel ID to test')

    args = parser.parse_args()

    if args.simple:
        test_simple_api()
    else:
        test_crawl_channel(args.channel)
