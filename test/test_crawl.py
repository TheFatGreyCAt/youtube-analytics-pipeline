import sys
import logging
from pathlib import Path

# Add extract module to path
sys.path.insert(0, str(Path(__file__).parent))

from extract.crawlers import YouTubeCrawler
from extract.config import validate_config, YOUTUBE_API_KEY, GCP_PROJECT_ID

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_crawl_channel(channel_id: str = 'UCXuqSBlHAE6Xw-yeJA0Tunw'):
    print("\n" + "="*80)
    print("YOUTUBE CRAWLER TEST")
    print("="*80)
    
    # Kiểm tra config
    try:
        validate_config()
        print(f"[OK] Config validated")
        print(f"   - API Key: {YOUTUBE_API_KEY[:10]}...")
        print(f"   - GCP Project: {GCP_PROJECT_ID}")
    except Exception as e:
        print(f"[ERROR] Config error: {e}")
        print("\nPlease create .env file with required configs:")
        print("   - YOUTUBE_API_KEY")
        print("   - GCP_PROJECT_ID")
        print("   - PG_CONN_STR or PG_* variables")
        return
    
    print(f"\nTarget Channel: {channel_id}")
    print("-"*80)
    
    try:
        crawler = YouTubeCrawler()
        
        # 1. Crawl channel info
        print("\nStep 1: Crawling channel information...")
        channel_data = crawler.crawl_channel(channel_id)
        
        if channel_data and channel_data.get('items'):
            item = channel_data['items'][0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            
            print(f"   [OK] Channel Name: {snippet.get('title', 'N/A')}")
            print(f"   [OK] Subscribers: {stats.get('subscriberCount', 'N/A')}")
            print(f"   [OK] Total Videos: {stats.get('videoCount', 'N/A')}")
            print(f"   [OK] Total Views: {stats.get('viewCount', 'N/A')}")
        
        # 2. Crawl videos (limited to 10 for testing)
        print("\nStep 2: Crawling videos (max 10)...")
        videos = crawler.crawl_videos(channel_id, max_results=10)
        print(f"   [OK] Crawled {len(videos)} videos")
        
        if videos:
            print("\n   Sample videos:")
            for i, video in enumerate(videos[:3], 1):
                title = video.get('snippet', {}).get('title', 'N/A')
                views = video.get('statistics', {}).get('viewCount', '0')
                print(f"      {i}. {title[:60]}... (Views: {views})")
        
        # 3. Crawl playlists
        print("\nStep 3: Crawling playlists...")
        crawler.crawl_playlists(channel_id, max_results=10)
        print(f"   [OK] Playlists crawled")
        
        # 4. Crawl comments (optional - only first video)
        if videos:
            video_id = videos[0].get('id', '')
            if video_id:
                print(f"\nStep 4: Crawling comments from first video...")
                try:
                    crawler.crawl_comments(video_id, channel_id, max_results=10)
                    print(f"   [OK] Comments crawled")
                except Exception as e:
                    print(f"   [WARNING] Comments: {e}")
        
        print("\n" + "="*80)
        print("CRAWL TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nData has been saved to BigQuery")
        print("Check your BigQuery dataset: raw_yt")
        print("   - raw_channels")
        print("   - raw_videos")
        print("   - raw_playlists")
        print("   - raw_comments")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"[ERROR] CRAWL TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


def test_simple_api():
    print("\n" + "="*80)
    print("SIMPLE API TEST (No Database)")
    print("="*80)
    
    try:
        from googleapiclient.discovery import build
        
        if not YOUTUBE_API_KEY:
            print("[ERROR] YOUTUBE_API_KEY not found in .env")
            return
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        print("\nTesting API call...")
        channel_id = 'UCXuqSBlHAE6Xw-yeJA0Tunw'  # Linus Tech Tips
        
        request = youtube.channels().list(
            part='snippet,statistics',
            id=channel_id
        )
        response = request.execute()
        
        if response.get('items'):
            item = response['items'][0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            
            print("\n[OK] API Call Successful")
            print(f"   Channel: {snippet.get('title', 'N/A')}")
            print(f"   Description: {snippet.get('description', 'N/A')[:100]}...")
            print(f"   Subscribers: {stats.get('subscriberCount', 'N/A')}")
            print(f"   Videos: {stats.get('videoCount', 'N/A')}")
            print(f"   Views: {stats.get('viewCount', 'N/A')}")
        else:
            print("[ERROR] No data returned from API")
        
    except Exception as e:
        print(f"\n[ERROR] API Test Failed: {e}")
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
