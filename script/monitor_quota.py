#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from extract.db_manager import PostgresManager
from extract.config import validate_config


def print_separator(char="=", length=70):
    print(char * length)


def display_quota_status():
    pg_manager = PostgresManager()
    quota_status = pg_manager.get_api_quota_status()
    
    print_separator()
    print("YOUTUBE API QUOTA STATUS")
    print_separator()
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Quota Used:     {quota_status['quota_used']:,} units")
    print(f"Daily Limit:    {quota_status['daily_limit']:,} units")
    print(f"Remaining:      {quota_status['daily_limit'] - quota_status['quota_used']:,} units")
    print(f"Usage:          {quota_status['percentage_used']:.1f}%")
    
    used_bars = int(quota_status['percentage_used'] / 5)
    remaining_bars = 20 - used_bars
    bar = "#" * used_bars + "-" * remaining_bars
    print(f"\n[{bar}]")
    
    if quota_status['percentage_used'] > 90:
        print("\nWARNING: Quota usage is very high!")
    elif quota_status['percentage_used'] > 80:
        print("\nCAUTION: Approaching quota limit")
    else:
        print("\nQuota usage is healthy")
    
    print_separator()


def display_channel_status():
    pg_manager = PostgresManager()
    channels = pg_manager.list_channels()
    
    print()
    print_separator()
    print("CHANNEL STATUS")
    print_separator()
    
    if not channels:
        print("No channels configured")
        return
    
    print(f"\nTotal Channels: {len(channels)}")
    print()
    
    active = [ch for ch in channels if ch[6]]
    inactive = [ch for ch in channels if not ch[6]]
    
    print(f"Active:   {len(active)}")
    print(f"Inactive: {len(inactive)}")
    print()
    
    print(f"{'Channel Name':<25} {'Status':<10} {'Last Crawl':<20} {'Next Crawl':<20}")
    print("-" * 75)
    
    for ch in active[:10]:
        name = ch[1][:24]
        status = ch[2]
        last_crawl = ch[3].strftime('%Y-%m-%d %H:%M') if ch[3] else 'Never'
        next_crawl = ch[4].strftime('%Y-%m-%d %H:%M') if ch[4] else 'N/A'
        
        print(f"{name:<25} {status:<10} {last_crawl:<20} {next_crawl:<20}")
    
    if len(active) > 10:
        print(f"\n... and {len(active) - 10} more channels")
    
    print_separator()


def display_recent_crawls():
    pg_manager = PostgresManager()
    logs = pg_manager.get_crawl_history(limit=10)
    
    print()
    print_separator()
    print("RECENT CRAWL HISTORY")
    print_separator()
    
    if not logs:
        print("No crawl history found")
        return
    
    print()
    print(f"{'Channel ID':<26} {'Timestamp':<20} {'Records':<10} {'Status':<10}")
    print("-" * 75)
    
    for log in logs:
        channel_id = log[0][:25]
        timestamp = log[1].strftime('%Y-%m-%d %H:%M')
        records = log[2] or 0
        status = log[3]
        
        print(f"{channel_id:<26} {timestamp:<20} {records:<10} {status:<10}")
    
    print_separator()


def estimate_quota_cost(num_channels: int, with_comments: bool = False):
    print()
    print_separator()
    print("QUOTA COST ESTIMATION")
    print_separator()
    print(f"\nEstimating cost for {num_channels} channel(s)...")
    print()
    
    base_cost = 3
    videos_cost = 2
    comments_cost = 10 if with_comments else 0
    
    cost_per_channel = base_cost + videos_cost + comments_cost
    total_cost = cost_per_channel * num_channels
    
    print(f"Base operations:        {base_cost} units/channel")
    print(f"Video fetching:         {videos_cost} units/channel")
    if with_comments:
        print(f"Comments (10 videos):   {comments_cost} units/channel")
    print()
    print(f"Cost per channel:       ~{cost_per_channel} units")
    print(f"Total estimated cost:   ~{total_cost} units")
    print()
    
    pg_manager = PostgresManager()
    quota_status = pg_manager.get_api_quota_status()
    remaining = quota_status['daily_limit'] - quota_status['quota_used']
    
    if total_cost <= remaining:
        print(f"You have enough quota ({remaining:,} units remaining)")
    else:
        print(f"WARNING: Estimated cost exceeds remaining quota!")
        print(f"   Remaining: {remaining:,} units")
        print(f"   Needed:    {total_cost:,} units")
        print(f"   Deficit:   {total_cost - remaining:,} units")
    
    print_separator()


def main():
    try:
        validate_config()
    except Exception as e:
        print(f"Configuration error: {e}")
        return
    
    import argparse
    parser = argparse.ArgumentParser(description="Monitor YouTube Analytics Pipeline")
    parser.add_argument('--estimate', type=int, metavar='N', help='Estimate quota cost for N channels')
    parser.add_argument('--with-comments', action='store_true', help='Include comments in estimation')
    parser.add_argument('--channels-only', action='store_true', help='Show only channel status')
    parser.add_argument('--quota-only', action='store_true', help='Show only quota status')
    
    args = parser.parse_args()
    
    if args.estimate:
        estimate_quota_cost(args.estimate, args.with_comments)
    elif args.quota_only:
        display_quota_status()
    elif args.channels_only:
        display_channel_status()
    else:
        display_quota_status()
        display_channel_status()
        display_recent_crawls()


if __name__ == "__main__":
    main()
