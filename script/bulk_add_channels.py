#!/usr/bin/env python3
import sys
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from extract.db_manager import PostgresManager


def read_channels_from_csv(file_path: Path):
    channels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            channels.append((
                row['channel_id'],
                row['channel_name'],
                int(row.get('frequency_hours', 24))
            ))
    return channels


def read_channels_from_txt(file_path: Path):
    channels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            channel_id = line.strip()
            if channel_id and not channel_id.startswith('#'):
                channels.append((channel_id, channel_id, 24))
    return channels


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Bulk add YouTube channels to database",
        epilog="""
Examples:
  python bulk_add_channels.py channels.csv
  python bulk_add_channels.py channels.txt
  python bulk_add_channels.py channels.csv --dry-run
        """
    )
    parser.add_argument('file', help='CSV or TXT file with channel IDs')
    parser.add_argument('--dry-run', action='store_true', help='Preview without adding')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    try:
        if file_path.suffix.lower() == '.csv':
            channels = read_channels_from_csv(file_path)
        else:
            channels = read_channels_from_txt(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    if not channels:
        print("Error: No channels found in file")
        return
    
    print(f"\n{'='*70}")
    print(f"BULK ADD CHANNELS")
    print(f"{'='*70}")
    print(f"\nFound {len(channels)} channel(s) to add:")
    print()
    
    for i, (ch_id, ch_name, freq) in enumerate(channels, 1):
        print(f"{i:3}. {ch_name:<40} ({ch_id}) - {freq}h")
    
    print(f"\n{'='*70}")
    
    if args.dry_run:
        print("DRY RUN - No changes made")
        return
    
    response = input("\nAdd these channels to database? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    pg_manager = PostgresManager()
    
    print("\nAdding channels...")
    try:
        pg_manager.add_channels_batch(channels)
        print(f"Successfully added {len(channels)} channels!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
