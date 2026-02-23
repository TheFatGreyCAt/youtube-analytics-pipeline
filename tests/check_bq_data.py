"""
Check BigQuery data across all tables
"""
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT = 'project-8fd99edc-9e20-4b82-b43'

credentials = service_account.Credentials.from_service_account_file(
    'credentials/project-8fd99edc-9e20-4b82-b43-41fc5f2ccbcd.json'
)
client = bigquery.Client(project=PROJECT, credentials=credentials)


def check_rows(dataset_id, table_id):
    try:
        result = client.query(
            f"SELECT COUNT(*) as n FROM `{PROJECT}.{dataset_id}.{table_id}`"
        ).result()
        return list(result)[0]['n']
    except Exception as e:
        return f"ERR: {e}"


def section(title, dataset, tables):
    print(f"\n{title}")
    print("-" * 70)
    total = 0
    for t in tables:
        n = check_rows(dataset, t)
        ok = isinstance(n, int)
        status = "✅" if ok and n > 0 else ("⚠️ " if ok else "❌")
        print(f"  {status} {dataset}.{t:<35} {str(n):>10} rows")
        if ok:
            total += n
    return total


def main():
    print("=" * 70)
    print(f"BIGQUERY DATA CHECK  —  project: {PROJECT}")
    print("=" * 70)

    section("📦  RAW  (source — crawler writes here)",
            "raw_yt",
            ["raw_videos", "raw_channels", "raw_playlists", "raw_comments"])

    section("🌱  SEEDS",
            "seeds",
            ["youtube_categories"])

    section("🔄  STAGING  (cast / rename / flatten)",
            "staging",
            ["stg_youtube__videos", "stg_youtube__channels", "stg_youtube__playlists"])

    section("⚙️   INTERMEDIATE  (joins / metrics)",
            "intermediate",
            ["int_videos__enhanced", "int_engagement_metrics", "int_channel_summary"])

    section("📊  MART  (BI-ready)",
            "mart",
            ["fct_video_performance", "dim_channel_summary", "agg_daily_metrics"])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
