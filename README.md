# Hướng dẫn cài đặt

## Bước 1: Chuẩn bị môi trường Python và dbt-bigquery
Tạo virtual env:
- python -m venv .venv
- .venv\Scripts\activate

Cài dbt-core và adapter BigQuery:
- python -m pip install dbt-core dbt-bigquery
- dbt --version (kiểm tra xem đã cài chưa)