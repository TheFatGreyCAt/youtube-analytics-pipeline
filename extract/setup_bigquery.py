import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

PROJECT_ID = os.getenv('GCP_PROJECT_ID')
DATASET_ID = os.getenv('BQ_DATASET_ID', 'raw_yt')
CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not PROJECT_ID:
    print("Error: GCP_PROJECT_ID not found in .env")
    exit(1)

print(f"Creating tables in {PROJECT_ID}.{DATASET_ID}...")

# Initialize BigQuery client
credentials = service_account.Credentials.from_service_account_file(f'../{CREDENTIALS_PATH}')
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# Read schema.sql
with open('schema.sql', 'r', encoding='utf-8') as f:
    schema_sql = f.read()

# Replace placeholders
schema_sql = schema_sql.replace('{PROJECT_ID}', PROJECT_ID)
schema_sql = schema_sql.replace('{DATASET_ID}', DATASET_ID)

# Split into statements
statements = [s.strip() for s in schema_sql.split(';') if s.strip()]

print(f"Creating {len(statements)} objects...\n")

# Create dataset
dataset_id = f"{PROJECT_ID}.{DATASET_ID}"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"
client.create_dataset(dataset, exists_ok=True)
print(f"Dataset {dataset_id} ready\n")

# Create tables/views
for i, statement in enumerate(statements, 1):
    try:
        query_job = client.query(statement)
        query_job.result()
        
        table_type = 'TABLE' if 'TABLE' in statement else 'VIEW'
        print(f"[{i}/{len(statements)}] {table_type} created")
    except Exception as e:
        print(f"[{i}/{len(statements)}] Error: {e}")

print(f"\nSetup completed!")
