import os
import sys
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_status(service, status, details=""):
    status_text = "OK" if status else "FAILED"
    print(f"[{status_text}] {service:.<40}")
    if details:
        print(f"   {details}")


def check_port(host, port, service_name):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print_status(f"{service_name} ({host}:{port})", True, "Port is open")
            return True
        else:
            print_status(f"{service_name} ({host}:{port})", False, "Port is closed")
            return False
    except Exception as e:
        print_status(f"{service_name} ({host}:{port})", False, str(e))
        return False


def check_env_variables():
    print_header("1. ENVIRONMENT VARIABLES")

    required_vars = {
        'YouTube API': {
            'YOUTUBE_API_KEY': os.getenv('YOUTUBE_API_KEY'),
        },
        'PostgreSQL': {
            'PG_HOST': os.getenv('PG_HOST', 'localhost'),
            'PG_PORT': os.getenv('PG_PORT', '5432'),
            'PG_DATABASE': os.getenv('PG_DATABASE'),
            'PG_USER': os.getenv('PG_USER'),
            'PG_PASSWORD': os.getenv('PG_PASSWORD'),
        },
        'BigQuery': {
            'GCP_PROJECT_ID': os.getenv('GCP_PROJECT_ID'),
            'BQ_DATASET_ID': os.getenv('BQ_DATASET_ID', 'raw_yt'),
            'GOOGLE_APPLICATION_CREDENTIALS': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        }
    }

    all_good = True
    for category, vars_dict in required_vars.items():
        print(f"\n{category}:")
        for var_name, var_value in vars_dict.items():
            if var_value:
                if 'KEY' in var_name or 'PASSWORD' in var_name:
                    display = f"{var_value[:8]}..." if len(var_value) > 8 else "***"
                else:
                    display = var_value[:50] + "..." if len(var_value) > 50 else var_value
                print(f"  [OK] {var_name} = {display}")
            else:
                print(f"  [FAILED] {var_name} = NOT SET")
                all_good = False

    return all_good


def check_ports():
    print_header("2. PORT CONNECTIVITY")

    ports_to_check = [
        ('localhost', 5432, 'PostgreSQL'),
        ('localhost', 4200, 'Prefect UI'),
        ('localhost', 6379, 'Redis'),
    ]

    results = []
    for host, port, service in ports_to_check:
        results.append(check_port(host, port, service))

    return all(results)


def check_postgresql():
    print_header("3. POSTGRESQL CONNECTION & SCHEMA")

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv('PG_HOST', 'localhost'),
            port=os.getenv('PG_PORT', '5432'),
            database=os.getenv('PG_DATABASE'),
            user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD')
        )
        cursor = conn.cursor()

        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0].split(',')[0]
        print_status("PostgreSQL Connection", True, version)

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = ['channels_config', 'crawl_log', 'api_quota_usage']

        print(f"\nTables found: {len(tables)}")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            is_expected = "[OK]" if table in expected_tables else "[INFO]"
            print(f"   {is_expected} {table}: {count:,} rows")

        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'channels_config'
            )
        """)
        has_postgres_schema = cursor.fetchone()[0]

        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name IN ('fact_videos', 'dim_channels')
            )
        """)
        has_bigquery_schema = cursor.fetchone()[0]

        print(f"\nSchema Analysis:")
        print_status("PostgreSQL Schema (metadata)", has_postgres_schema,
                    "channels_config, crawl_log, api_quota_usage")
        print_status("BigQuery Schema NOT in PostgreSQL", not has_bigquery_schema,
                    "fact_videos, dim_channels should only be in BigQuery")

        cursor.close()
        conn.close()

        return has_postgres_schema and not has_bigquery_schema

    except Exception as e:
        print_status("PostgreSQL", False, str(e))
        return False


def check_bigquery():
    print_header("4. BIGQUERY CONNECTION & TABLES")

    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path or not Path(creds_path).exists():
            print_status("BigQuery Credentials", False, f"File not found: {creds_path}")
            return False

        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = bigquery.Client(credentials=credentials, project=os.getenv('GCP_PROJECT_ID'))

        print_status("BigQuery Connection", True, f"Project: {client.project}")

        dataset_id = os.getenv('BQ_DATASET_ID', 'raw_yt')
        dataset_ref = client.dataset(dataset_id)

        try:
            dataset = client.get_dataset(dataset_ref)
            print_status(f"Dataset '{dataset_id}'", True, f"Location: {dataset.location}")

            tables = list(client.list_tables(dataset))
            print(f"\nTables in {dataset_id}: {len(tables)}")

            expected_tables = ['fact_videos', 'dim_channels', 'raw_videos', 'raw_channels', 'raw_playlists', 'raw_comments']

            for table in tables:
                table_ref = dataset_ref.table(table.table_id)
                table_obj = client.get_table(table_ref)
                is_expected = "[OK]" if table.table_id in expected_tables else "[INFO]"
                print(f"   {is_expected} {table.table_id}: {table_obj.num_rows:,} rows")

            return True

        except Exception as e:
            print_status(f"Dataset '{dataset_id}'", False, str(e))
            return False

    except Exception as e:
        print_status("BigQuery", False, str(e))
        return False


def check_docker():
    print_header("5. DOCKER CONTAINERS")

    try:
        result = subprocess.run(
            ['docker', 'version', '--format', '{{.Server.Version}}'],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            print_status("Docker Daemon", True, f"Version: {result.stdout.strip()}")
        else:
            print_status("Docker Daemon", False, "Not running")
            return False

        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            containers = [line.split('\t') for line in result.stdout.strip().split('\n') if line]
            expected_services = ['postgres', 'redis', 'prefect-server', 'prefect-services', 'prefect-worker']

            print(f"\nRunning Containers: {len(containers)}")
            for name, status in containers:
                is_expected = "[OK]" if any(svc in name for svc in expected_services) else "[INFO]"
                print(f"   {is_expected} {name}: {status}")

            running_services = [name for name, _ in containers]
            return all(any(svc in container for container in running_services)
                       for svc in ['postgres', 'prefect-server'])
        else:
            print_status("Docker Containers", False, "Cannot list containers")
            return False

    except Exception as e:
        print_status("Docker", False, str(e))
        return False


def check_dbt():
    print_header("6. DBT CONFIGURATION")

    try:
        result = subprocess.run(
            ['dbt', '--version'], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            print_status("dbt Installation", True, result.stdout.strip().split('\n')[0])
        else:
            print_status("dbt Installation", False, "Not installed")
            return False

        dbt_project_path = Path('dbt_project/dbt_project.yml')
        if dbt_project_path.exists():
            print_status("dbt Project File", True, str(dbt_project_path))
        else:
            print_status("dbt Project File", False, f"Not found: {dbt_project_path}")
            return False

        profiles_path = Path('dbt_project/profiles.yml')
        if profiles_path.exists():
            print_status("dbt Profiles", True, str(profiles_path))
        else:
            print_status("dbt Profiles", False, f"Not found: {profiles_path}")
            return False

        print("\nRunning dbt debug...")
        result = subprocess.run(
            ['dbt', 'debug', '--project-dir', 'dbt_project', '--profiles-dir', 'dbt_project'],
            capture_output=True, text=True, timeout=30
        )

        success = 'All checks passed!' in result.stdout
        if success:
            print_status("dbt Debug", True, "All checks passed")
        else:
            print_status("dbt Debug", False, "Some checks failed")
            print("\n" + result.stdout[-500:])

        return success

    except Exception as e:
        print_status("dbt", False, str(e))
        return False


def check_youtube_api():
    print_header("7. YOUTUBE API")

    try:
        from googleapiclient.discovery import build

        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            print_status("YouTube API Key", False, "Not set")
            return False

        youtube = build('youtube', 'v3', developerKey=api_key)

        request = youtube.channels().list(
            part='snippet,statistics',
            id='UCXuqSBlHAE6Xw-yeJA0Tunw'
        )
        response = request.execute()

        if response.get('items'):
            channel = response['items'][0]
            print_status("YouTube API", True, f"Test successful: {channel['snippet']['title']}")
            print(f"   Subscribers: {int(channel['statistics']['subscriberCount']):,}")
            return True
        else:
            print_status("YouTube API", False, "No data returned")
            return False

    except Exception as e:
        print_status("YouTube API", False, str(e))
        return False


def check_prefect_ui():
    print_header("8. PREFECT UI")

    try:
        import urllib.request

        prefect_url = f"http://localhost:{os.getenv('PREFECT_SERVER_PORT', '4200')}/api/health"

        try:
            response = urllib.request.urlopen(prefect_url, timeout=5)
            if response.status == 200:
                print_status("Prefect API Health", True, prefect_url)
                print(f"\n   Access Prefect UI at: http://localhost:4200")
                return True
            else:
                print_status("Prefect API Health", False, f"Status: {response.status}")
                return False
        except Exception as e:
            print_status("Prefect API Health", False, str(e))
            return False

    except Exception as e:
        print_status("Prefect UI", False, str(e))
        return False


def print_summary(results):
    print_header("SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"\n{'Check':<40} Status")
    print("-" * 50)
    for check, status in results.items():
        print(f"{check:<40} {'[OK]' if status else '[FAILED]'}")

    print("-" * 50)
    print(f"{'Score:':<40} {passed}/{total} ({percentage:.0f}%)")

    if passed == total:
        print("\nAll systems operational!")
        return 0
    else:
        print(f"\n{total - passed} checks failed.")
        return 1


def main():
    print("\n" + "="*70)
    print(" " * 20 + "YOUTUBE ANALYTICS PIPELINE")
    print(" " * 23 + "System Health Check")
    print(" " * 20 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    results = {}
    results['Environment Variables'] = check_env_variables()
    results['Port Connectivity'] = check_ports()
    results['PostgreSQL'] = check_postgresql()
    results['BigQuery'] = check_bigquery()
    results['Docker Containers'] = check_docker()
    results['dbt Configuration'] = check_dbt()
    results['YouTube API'] = check_youtube_api()
    results['Prefect UI'] = check_prefect_ui()

    return print_summary(results)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nHealth check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
