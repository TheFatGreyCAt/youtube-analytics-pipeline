import sys
import argparse
import logging
from typing import Optional

from .db_manager import PostgresManager, BigQueryManager
from .crawlers import YouTubeCrawler, crawl_scheduled_channels, crawl_from_config_file
from .config import validate_config, get_active_channels, CHANNELS_CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_database():
    print("Setting up databases...")
    
    pg_manager = PostgresManager()
    pg_manager.setup_tables()
    
    bq_manager = BigQueryManager()
    bq_manager.setup_tables()
    
    print("Database setup completed!")


def list_channels_from_file():
    channels = get_active_channels()
    
    if not channels:
        print(f"\nNo channels found in {CHANNELS_CONFIG_PATH}")
        return
    
    print("\n" + "="*100)
    print(f"CHANNELS FROM {CHANNELS_CONFIG_PATH}")
    print("="*100)
    print(f"{'NAME':<30} {'CHANNEL ID':<30} {'PRIORITY':<10} {'ACTIVE':<10} {'COMMENTS':<10}")
    print("="*100)
    
    for ch in channels:
        name = ch.get('name', 'N/A')[:28]
        channel_id = ch.get('id', 'N/A')
        priority = ch.get('priority', 1)
        active = 'Yes' if ch.get('active', True) else 'No'
        comments = 'Yes' if ch.get('include_comments', False) else 'No'
        
        print(f"{name:<30} {channel_id:<30} {priority:<10} {active:<10} {comments:<10}")
    
    print("="*100)
    print(f"Total: {len(channels)} active channel(s)\n")


def add_channel(channel_id: str, channel_name: str, frequency_hours: int = 24):
    pg_manager = PostgresManager()
    pg_manager.add_channel(channel_id, channel_name, frequency_hours)
    print(f"Added channel: {channel_name} ({channel_id})")


def list_channels():
    pg_manager = PostgresManager()
    channels = pg_manager.list_channels()
    
    if not channels:
        print("No channels configured")
        return
    
    print("\n" + "="*100)
    print(f"{'CHANNEL ID':<30} {'NAME':<25} {'STATUS':<10} {'LAST CRAWL':<20} {'NEXT CRAWL':<20}")
    print("="*100)
    
    for ch in channels:
        channel_id, name, status, last_crawl, next_crawl, freq, is_active = ch
        last_str = last_crawl.strftime('%Y-%m-%d %H:%M') if last_crawl else 'Never'
        next_str = next_crawl.strftime('%Y-%m-%d %H:%M') if next_crawl else 'N/A'
        active_str = 'Yes' if is_active else 'No'
        
        print(f"{channel_id:<30} {name:<25} {status:<10} {last_str:<20} {next_str:<20}")
    
    print("="*100 + "\n")


def remove_channel(channel_id: str):
    pg_manager = PostgresManager()
    pg_manager.remove_channel(channel_id)
    print(f"Removed channel: {channel_id}")


def view_history(channel_id: Optional[str] = None, limit: int = 10):
    pg_manager = PostgresManager()
    logs = pg_manager.get_crawl_history(channel_id, limit)
    
    if not logs:
        print("No crawl history found")
        return
    
    print("\n" + "="*120)
    print(f"{'CHANNEL ID':<30} {'TIMESTAMP':<20} {'RECORDS':<10} {'STATUS':<10} {'ERROR':<50}")
    print("="*120)
    
    for log in logs:
        ch_id, crawl_ts, records, status, error = log
        ts_str = crawl_ts.strftime('%Y-%m-%d %H:%M:%S')
        error_str = (error[:47] + '...') if error and len(error) > 50 else (error or '')
        
        print(f"{ch_id:<30} {ts_str:<20} {records:<10} {status:<10} {error_str:<50}")
    
    print("="*120 + "\n")


def crawl_now(channel_id: Optional[str] = None, limit: int = 10, with_comments: bool = False):
    if channel_id:
        print(f"Crawling channel: {channel_id}")
        crawler = YouTubeCrawler()
        crawler.crawl_channel_full(channel_id, include_comments=with_comments)
        print(f"Crawl completed for {channel_id}")
    else:
        print(f"Running scheduled crawl (limit: {limit} channels)")
        crawl_scheduled_channels(limit=limit, include_comments=with_comments)
        print("Scheduled crawl completed")


def crawl_from_file(limit: int = None):
    print(f"Running crawl from {CHANNELS_CONFIG_PATH}")
    if limit:
        print(f"Limit: {limit} channel(s)")
    crawl_from_config_file(limit=limit)
    print("\nCrawl from file completed")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='YouTube Analytics Extract CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup databases
  python -m extract.cli setup
  
  # Crawl from channels.yml file (RECOMMENDED)
  python -m extract.cli crawl-file
  python -m extract.cli crawl-file --limit 2
  python -m extract.cli channels
  
  # Legacy database methods
  python -m extract.cli add UCXuqSBlHAE6Xw-yeJA0Tunw "Linus Tech Tips" --frequency 24
  python -m extract.cli list
  python -m extract.cli crawl --limit 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    subparsers.add_parser('setup', help='Setup PostgreSQL and BigQuery databases')
    
    subparsers.add_parser('channels', help='List channels from channels.yml file')
    
    crawl_file_parser = subparsers.add_parser('crawl-file', help='Crawl channels from channels.yml (recommended)')
    crawl_file_parser.add_argument('--limit', type=int, help='Max channels to crawl')
    
    add_parser = subparsers.add_parser('add', help='Add a new channel to database')
    add_parser.add_argument('channel_id', help='YouTube channel ID')
    add_parser.add_argument('channel_name', help='Channel name')
    add_parser.add_argument('--frequency', type=int, default=24, help='Crawl frequency in hours (default: 24)')
    
    subparsers.add_parser('list', help='List all configured channels in database')
    
    remove_parser = subparsers.add_parser('remove', help='Remove a channel from database')
    remove_parser.add_argument('channel_id', help='YouTube channel ID to remove')
    
    history_parser = subparsers.add_parser('history', help='View crawl history')
    history_parser.add_argument('--channel', help='Filter by channel ID')
    history_parser.add_argument('--limit', type=int, default=10, help='Number of records (default: 10)')
    
    crawl_parser = subparsers.add_parser('crawl', help='Run crawl from database (legacy)')
    crawl_parser.add_argument('--channel', help='Crawl specific channel (optional)')
    crawl_parser.add_argument('--limit', type=int, default=10, help='Max channels to crawl (default: 10)')
    crawl_parser.add_argument('--with-comments', action='store_true', help='Include comments crawl')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        validate_config()
        
        if args.command == 'setup':
            setup_database()
        
        elif args.command == 'channels':
            list_channels_from_file()
        
        elif args.command == 'crawl-file':
            crawl_from_file(args.limit)
        
        elif args.command == 'add':
            add_channel(args.channel_id, args.channel_name, args.frequency)
        
        elif args.command == 'list':
            list_channels()
        
        elif args.command == 'remove':
            remove_channel(args.channel_id)
        
        elif args.command == 'history':
            view_history(args.channel, args.limit)
        
        elif args.command == 'crawl':
            crawl_now(args.channel, args.limit, args.with_comments)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
