"""
Test script to crawl YouTube data from API
Run: python test_crawl.py
"""
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
    """
    Test crawl một channel (mặc định: Linus Tech Tips)
    
    Args:
        channel_id: YouTube Channel ID
    """
    print("\n" + "="*80)
    print("🚀 YOUTUBE CRAWLER TEST")
    print("="*80)
    
    # Kiểm tra config
    try:
        validate_config()
        print(f"✅ Config validated")
        print(f"   - API Key: {YOUTUBE_API_KEY[:10]}...")
        print(f"   - GCP Project: {GCP_PROJECT_ID}")
    except Exception as e:
        print(f"❌ Config error: {e}")
        print("\n📝 Please create .env file with required configs:")
        print("   - YOUTUBE_API_KEY")
        print("   - GCP_PROJECT_ID")
        print("   - PG_CONN_STR or PG_* variables")
        return
    
    print(f"\n🎯 Target Channel: {channel_id}")
    print("-"*80)
    
    try:
        crawler = YouTubeCrawler()
        
        # 1. Crawl channel info
        print("\n📺 Step 1: Crawling channel information...")
        channel_data = crawler.crawl_channel(channel_id)
        
        if channel_data and channel_data.get('items'):
            item = channel_data['items'][0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            
            print(f"   ✅ Channel Name: {snippet.get('title', 'N/A')}")
            print(f"   ✅ Subscribers: {stats.get('subscriberCount', 'N/A')}")
            print(f"   ✅ Total Videos: {stats.get('videoCount', 'N/A')}")
            print(f"   ✅ Total Views: {stats.get('viewCount', 'N/A')}")
        
        # 2. Crawl videos (limited to 10 for testing)
        print("\n🎬 Step 2: Crawling videos (max 10)...")
        videos = crawler.crawl_videos(channel_id, max_results=10)
        print(f"   ✅ Crawled {len(videos)} videos")
        
        if videos:
            print("\n   📋 Sample videos:")
            for i, video in enumerate(videos[:3], 1):
                title = video.get('snippet', {}).get('title', 'N/A')
                views = video.get('statistics', {}).get('viewCount', '0')
                print(f"      {i}. {title[:60]}... (Views: {views})")
        
        # 3. Crawl playlists
        print("\n📚 Step 3: Crawling playlists...")
        crawler.crawl_playlists(channel_id, max_results=10)
        print(f"   ✅ Playlists crawled")
        
        # 4. Crawl comments (optional - only first video)
        if videos:
            video_id = videos[0].get('id', '')
            if video_id:
                print(f"\n💬 Step 4: Crawling comments from first video...")
                try:
                    crawler.crawl_comments(video_id, channel_id, max_results=10)
                    print(f"   ✅ Comments crawled")
                except Exception as e:
                    print(f"   ⚠️  Comments: {e}")
        
        print("\n" + "="*80)
        print("✅ CRAWL TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📊 Data has been saved to BigQuery")
        print("🔍 Check your BigQuery dataset: raw_yt")
        print("   - raw_channels")
        print("   - raw_videos")
        print("   - raw_playlists")
        print("   - raw_comments")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ CRAWL TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


def test_simple_api():
    """Test đơn giản chỉ gọi YouTube API (không lưu database)"""
    print("\n" + "="*80)
    print("🧪 SIMPLE API TEST (No Database)")
    print("="*80)
    
    try:
        from googleapiclient.discovery import build
        
        if not YOUTUBE_API_KEY:
            print("❌ YOUTUBE_API_KEY not found in .env")
            return
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        print("\n🔍 Testing API call...")
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
            
            print("\n✅ API Call Successful!")
            print(f"   Channel: {snippet.get('title', 'N/A')}")
            print(f"   Description: {snippet.get('description', 'N/A')[:100]}...")
            print(f"   Subscribers: {stats.get('subscriberCount', 'N/A')}")
            print(f"   Videos: {stats.get('videoCount', 'N/A')}")
            print(f"   Views: {stats.get('viewCount', 'N/A')}")
        else:
            print("❌ No data returned from API")
        
    except Exception as e:
        print(f"\n❌ API Test Failed: {e}")
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
