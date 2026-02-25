import sys
import argparse
import logging
from typing import Optional

from .db_manager import PostgresManager, BigQueryManager
from .crawlers import (
    YouTubeCrawler, 
    crawl_scheduled_channels, 
    crawl_scheduled_channels_smart,
    crawl_from_config_file
)
from .channel_finder import ChannelFinder
from .config import validate_config, get_active_channels, CHANNELS_CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_database():
    logging.info("Setting up databases...")
    pg_manager = PostgresManager()
    pg_manager.setup_tables()
    bq_manager = BigQueryManager()
    bq_manager.setup_tables()
    logging.info("Database setup completed")


def list_channels_from_file():
    channels = get_active_channels()
    if not channels:
        logging.info(f"No channels found in {CHANNELS_CONFIG_PATH}")
        return
    
    logging.info(f"{'NAME':<30} {'CHANNEL ID':<30} {'PRIORITY':<10} {'ACTIVE':<10} {'COMMENTS':<10}")
    for ch in channels:
        name = ch.get('name', 'N/A')[:28]
        channel_id = ch.get('id', 'N/A')
        priority = ch.get('priority', 1)
        active = 'Yes' if ch.get('active', True) else 'No'
        comments = 'Yes' if ch.get('include_comments', False) else 'No'
        logging.info(f"{name:<30} {channel_id:<30} {priority:<10} {active:<10} {comments:<10}")
    
    logging.info(f"Total: {len(channels)} active channel(s)")


def add_channel(channel_id: str, channel_name: str, frequency_hours: int = 24):
    pg_manager = PostgresManager()
    pg_manager.add_channel(channel_id, channel_name, frequency_hours)


def list_channels():
    pg_manager = PostgresManager()
    channels = pg_manager.list_channels()
    
    if not channels:
        logging.info("No channels configured")
        return
    
    logging.info(f"{'CHANNEL ID':<30} {'NAME':<25} {'STATUS':<10} {'LAST CRAWL':<20} {'NEXT CRAWL':<20}")
    for ch in channels:
        channel_id, name, status, last_crawl, next_crawl, freq, is_active = ch
        last_str = last_crawl.strftime('%Y-%m-%d %H:%M') if last_crawl else 'Never'
        next_str = next_crawl.strftime('%Y-%m-%d %H:%M') if next_crawl else 'N/A'
        logging.info(f"{channel_id:<30} {name:<25} {status:<10} {last_str:<20} {next_str:<20}")


def remove_channel(channel_id: str):
    pg_manager = PostgresManager()
    pg_manager.remove_channel(channel_id)


def view_history(channel_id: Optional[str] = None, limit: int = 10):
    pg_manager = PostgresManager()
    logs = pg_manager.get_crawl_history(channel_id, limit)
    
    if not logs:
        logging.info("No crawl history found")
        return
    
    logging.info(f"{'CHANNEL ID':<30} {'TIMESTAMP':<20} {'RECORDS':<10} {'STATUS':<10} {'ERROR':<50}")
    for log in logs:
        ch_id, crawl_ts, records, status, error = log
        ts_str = crawl_ts.strftime('%Y-%m-%d %H:%M:%S')
        error_str = (error[:47] + '...') if error and len(error) > 50 else (error or '')
        logging.info(f"{ch_id:<30} {ts_str:<20} {records:<10} {status:<10} {error_str:<50}")


def crawl_now(channel_id: Optional[str] = None, limit: int = 10, with_comments: bool = False):
    if channel_id:
        crawler = YouTubeCrawler()
        crawler.crawl_channel_full(channel_id, include_comments=with_comments)
    else:
        crawl_scheduled_channels(limit=limit, include_comments=with_comments)


def crawl_from_file(limit: Optional[int] = None):
    crawl_from_config_file(limit=limit)


def discover_and_add_channels(input_csv: str, output_csv: Optional[str] = None, update_existing: bool = False):
    validate_config()
    finder = ChannelFinder()
    pg_manager = PostgresManager()
    
    summary = finder.search_and_add_from_csv(
        input_csv=input_csv,
        pg_manager=pg_manager,
        output_csv=output_csv,
        delay=0.5,
        update_existing=update_existing
    )
    
    logging.info(f"Channel discovery completed - Added: {summary['added']}, Updated: {summary['updated']}, Skipped: {summary['skipped']}, Failed: {summary['failed']}")


def add_single_channel_by_name(channel_name: str, frequency: Optional[int] = None, priority: Optional[int] = None):
    validate_config()
    finder = ChannelFinder()
    pg_manager = PostgresManager()
    
    result = finder.search_channel(channel_name)
    if not result:
        logging.error(f"Channel not found: {channel_name}")
        return
    
    channel_id, found_name = result
    logging.info(f"Found: {found_name} ({channel_id})")
    
    details = finder.get_channel_details(channel_id)
    if not details:
        logging.warning("Could not get channel details")
        return
    
    calculated_priority = finder.calculate_priority(
        details.get('subscriber_count', 0),
        details.get('video_count', 0)
    )
    calculated_frequency = finder.calculate_crawl_frequency(
        details.get('subscriber_count', 0),
        details.get('video_count', 0)
    )
    
    final_frequency = frequency or calculated_frequency
    final_priority = priority or calculated_priority
    
    logging.info(f"Subscribers: {int(details['subscriber_count']):,}, Videos: {int(details['video_count']):,}")
    logging.info(f"Priority: {final_priority}/5, Frequency: {final_frequency}h")
    
    pg_manager.add_channel(channel_id, found_name, final_frequency)
    pg_manager.update_api_quota(102)
    logging.info("Channel added successfully")


def crawl_new_channels(limit: int = 10, with_comments: bool = False):
    pg_manager = PostgresManager()
    
    import psycopg2
    conn = psycopg2.connect(pg_manager.conn_str)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT channel_id, channel_name, priority
        FROM channels_config
        WHERE last_crawl_ts IS NULL AND is_active = TRUE
        ORDER BY priority DESC, created_at ASC
        LIMIT %s
    """, (limit,))
    
    channels = cur.fetchall()
    cur.close()
    conn.close()
    
    if not channels:
        logging.info("All channels have been crawled")
        return
    
    logging.info(f"Found {len(channels)} new channels to crawl")
    
    crawler = YouTubeCrawler()
    success_count = 0
    
    for i, (channel_id, channel_name, priority) in enumerate(channels, 1):
        try:
            logging.info(f"[{i}/{len(channels)}] Crawling: {channel_name} (Priority: {priority})")
            crawler.crawl_channel_full(channel_id, include_comments=with_comments)
            success_count += 1
        except Exception as e:
            logging.error(f"Failed: {e}")
    
    logging.info(f"Completed: {success_count}/{len(channels)} successful")


def show_quota_status():
    pg_manager = PostgresManager()
    status = pg_manager.get_api_quota_status()
    
    logging.info(f"Used: {status['quota_used']:,} / {status['daily_limit']:,} units")
    logging.info(f"Remaining: {status['daily_limit'] - status['quota_used']:,} units")
    logging.info(f"Percentage: {status['percentage_used']:.1f}%")
    
    if status['percentage_used'] < 50:
        logging.info("Status: Good")
    elif status['percentage_used'] < 80:
        logging.info("Status: Caution")
    elif status['percentage_used'] < 90:
        logging.info("Status: Warning")
    else:
        logging.info("Status: Critical")


def main():
    parser = argparse.ArgumentParser(description='YouTube Analytics Extract CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    subparsers.add_parser('setup', help='Setup databases')
    
    discover_parser = subparsers.add_parser('discover', help='Discover channels from CSV and add to database')
    discover_parser.add_argument('input_csv', help='Path to CSV file')
    discover_parser.add_argument('--output', help='Output CSV path')
    discover_parser.add_argument('--update', action='store_true', help='Update existing channels')
    
    add_by_name_parser = subparsers.add_parser('add-by-name', help='Search and add channel by name')
    add_by_name_parser.add_argument('channel_name', help='Channel name')
    add_by_name_parser.add_argument('--frequency', type=int, help='Crawl frequency in hours')
    add_by_name_parser.add_argument('--priority', type=int, choices=[1,2,3,4,5], help='Priority 1-5')
    
    crawl_smart_parser = subparsers.add_parser('crawl-smart', help='Smart crawl with auto full/incremental')
    crawl_smart_parser.add_argument('--limit', type=int, default=20, help='Max channels (default: 20)')
    crawl_smart_parser.add_argument('--with-comments', action='store_true', help='Include comments')
    
    crawl_new_parser = subparsers.add_parser('crawl-new', help='Crawl never-crawled channels')
    crawl_new_parser.add_argument('--limit', type=int, default=10, help='Max channels (default: 10)')
    crawl_new_parser.add_argument('--with-comments', action='store_true', help='Include comments')
    
    crawl_scheduled_parser = subparsers.add_parser('crawl-scheduled', help='Crawl scheduled channels')
    crawl_scheduled_parser.add_argument('--limit', type=int, default=10, help='Max channels (default: 10)')
    crawl_scheduled_parser.add_argument('--with-comments', action='store_true', help='Include comments')
    
    subparsers.add_parser('channels', help='List channels from channels.yml')
    
    crawl_file_parser = subparsers.add_parser('crawl-file', help='Crawl from channels.yml')
    crawl_file_parser.add_argument('--limit', type=int, help='Max channels')
    
    subparsers.add_parser('quota', help='Show API quota status')
    subparsers.add_parser('list', help='List all channels in database')
    
    history_parser = subparsers.add_parser('history', help='View crawl history')
    history_parser.add_argument('--channel', help='Filter by channel ID')
    history_parser.add_argument('--limit', type=int, default=10, help='Number of records (default: 10)')
    
    add_parser = subparsers.add_parser('add', help='Add channel by ID')
    add_parser.add_argument('channel_id', help='YouTube channel ID')
    add_parser.add_argument('channel_name', help='Channel name')
    add_parser.add_argument('--frequency', type=int, default=24, help='Crawl frequency in hours')
    
    remove_parser = subparsers.add_parser('remove', help='Remove channel')
    remove_parser.add_argument('channel_id', help='YouTube channel ID')
    
    crawl_parser = subparsers.add_parser('crawl', help='Run crawl')
    crawl_parser.add_argument('--channel', help='Crawl specific channel')
    crawl_parser.add_argument('--limit', type=int, default=10, help='Max channels')
    crawl_parser.add_argument('--with-comments', action='store_true', help='Include comments')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        validate_config()
        
        if args.command == 'setup':
            setup_database()
        elif args.command == 'discover':
            discover_and_add_channels(args.input_csv, args.output, args.update)
        elif args.command == 'add-by-name':
            add_single_channel_by_name(args.channel_name, args.frequency, args.priority)
        elif args.command == 'crawl-smart':
            crawl_scheduled_channels_smart(args.limit, args.with_comments)
        elif args.command == 'crawl-new':
            crawl_new_channels(args.limit, args.with_comments)
        elif args.command == 'crawl-scheduled':
            crawl_scheduled_channels(args.limit, args.with_comments)
        elif args.command == 'channels':
            list_channels_from_file()
        elif args.command == 'crawl-file':
            crawl_from_file(args.limit)
        elif args.command == 'quota':
            show_quota_status()
        elif args.command == 'list':
            list_channels()
        elif args.command == 'history':
            view_history(args.channel, args.limit)
        elif args.command == 'add':
            add_channel(args.channel_id, args.channel_name, args.frequency)
        elif args.command == 'remove':
            remove_channel(args.channel_id)
        elif args.command == 'crawl':
            crawl_now(args.channel, args.limit, args.with_comments)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
