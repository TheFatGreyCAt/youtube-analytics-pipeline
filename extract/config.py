import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_CHANNEL_ID = os.getenv('YOUTUBE_CHANNEL_ID')

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
BQ_DATASET_ID = os.getenv('BQ_DATASET_ID', 'raw_yt')
CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str(project_root / CREDENTIALS_PATH)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

PG_HOST = os.getenv('PG_HOST', 'localhost')
PG_PORT = os.getenv('PG_PORT', '5432')
PG_DATABASE = os.getenv('PG_DATABASE', 'prefect')
PG_USER = os.getenv('PG_USER', 'prefect')
PG_PASSWORD = os.getenv('PG_PASSWORD')
PG_CONN_STR = os.getenv('PG_CONN_STR') or f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

CHANNELS_CONFIG_PATH = project_root / 'config' / 'channels.yml'

def load_channels_config():
    if not CHANNELS_CONFIG_PATH.exists():
        return {'channels': [], 'settings': {}}
    with open(CHANNELS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {'channels': [], 'settings': {}}

def get_active_channels():
    config = load_channels_config()
    channels = config.get('channels', [])
    active = [ch for ch in channels if ch.get('active', True)]
    active.sort(key=lambda x: x.get('priority', 1), reverse=True)
    return active

def get_crawl_settings():
    config = load_channels_config()
    default_settings = {
        'max_videos_per_channel': 50,
        'max_comments_per_video': 100,
        'batch_size': 10,
        'retry_failed_after_hours': 1,
        'api_delay_seconds': 0.5
    }
    return {**default_settings, **config.get('settings', {})}

def validate_config():
    required = {
        'YOUTUBE_API_KEY': YOUTUBE_API_KEY,
        'GCP_PROJECT_ID': GCP_PROJECT_ID,
        'PG_CONN_STR': PG_CONN_STR,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required config: {', '.join(missing)}")
    return True
