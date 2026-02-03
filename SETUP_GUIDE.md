# Hướng dẫn Cài Đặt Prefect và Docker

## 📋 Yêu Cầu Hệ Thống

- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+
- **Python**: 3.11+ (nếu chạy locally)
- **Git**: Để clone repository

## 🚀 Cài Đặt Nhanh

### 1. Chuẩn Bị Credentials

```bash
# Tạo thư mục credentials
mkdir -p credentials

# Copy Google Cloud Service Account JSON file vào thư mục
# File này có tên: service-account-key.json
cp /path/to/service-account-key.json credentials/
```

**Lấy Service Account Key:**
1. Vào [Google Cloud Console](https://console.cloud.google.com/)
2. Chọn Project → IAM & Admin → Service Accounts
3. Tạo Service Account hoặc chọn account hiện có
4. Vào tab "Keys" → "Create new key" → JSON format
5. Download file và lưu vào thư mục `credentials/`

### 2. Cấu Hình Environment Variables

```bash
# Copy .env.example thành .env
cp .env.example .env

# Mở .env và điền các thông tin:
# - YOUTUBE_API_KEY: Từ YouTube Data API v3
# - YOUTUBE_CHANNEL_ID: ID channel YouTube muốn scrape
# - GCP_PROJECT_ID: ID project Google Cloud
```

**Lấy YouTube API Key:**
1. Vào [Google Cloud Console](https://console.cloud.google.com/)
2. APIs & Services → Enable APIs and Services
3. Tìm "YouTube Data API v3" → Enable
4. Credentials → Create Credentials → API Key

### 3. Chạy Docker Containers

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Kiểm tra status
docker-compose ps

# View logs
docker-compose logs -f prefect-server
```

### 4. Truy Cập Prefect UI

- **Prefect Server UI**: http://localhost:4200
- **Jupyter Lab**: http://localhost:8888

## 📦 Cài Đặt Local (không dùng Docker)

### 1. Tạo Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài Đặt Dependencies

```bash
# Cài main dependencies
pip install -r requirements.txt

# Cài extract dependencies (optional)
pip install -r extract/requirements.txt

# Cài prefect dependencies (optional)
pip install -r prefect/requirements.txt
```

### 3. Cấu Hình Environment

```bash
# Copy .env.example
cp .env.example .env

# Chỉnh sửa .env với thông tin của bạn
```

### 4. Chạy Prefect Server Local

```bash
# Terminal 1: Start Prefect Server
prefect server start

# Terminal 2: Start Prefect Agent
prefect agent start -q default

# Terminal 3: Register và chạy flow
python prefect/youtube_flow.py
```

## 🔧 Commands Hữu Ích

### Docker Compose

```bash
# View logs
docker-compose logs -f prefect-server
docker-compose logs -f prefect-agent

# Stop services
docker-compose down

# Remove volumes (clear data)
docker-compose down -v

# Rebuild specific service
docker-compose build prefect-server

# Scale services
docker-compose up -d --scale prefect-agent=2
```

### Prefect CLI (local)

```bash
# Deploy flow
prefect deployment build prefect/youtube_flow.py:youtube_analytics_flow -n "youtube-pipeline" -q default

# List flows
prefect flow ls

# View runs
prefect run ls

# Check agent status
prefect agent status
```

## 📊 Architecture

```
┌─────────────────────────────────────────────┐
│         Docker Compose Network              │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────┐  ┌──────────────────┐│
│  │  Prefect Server  │  │  PostgreSQL DB   ││
│  │  (Port: 4200)    │  │  (Port: 5432)    ││
│  └──────────────────┘  └──────────────────┘│
│          ▲                      ▲           │
│          │                      │           │
│  ┌──────────────────┐           │           │
│  │  Prefect Agent   │           │           │
│  │  (Task Executor) │───────────┘           │
│  └──────────────────┘                       │
│          │                                  │
│          ▼                                  │
│  ┌──────────────────┐  ┌──────────────────┐│
│  │   YouTube API    │  │   BigQuery       ││
│  │   (External)     │  │   (External)     ││
│  └──────────────────┘  └──────────────────┘│
│                                             │
│  ┌──────────────────┐                      │
│  │   Jupyter Lab    │  (Optional)          │
│  │   (Port: 8888)   │                      │
│  └──────────────────┘                      │
│                                             │
└─────────────────────────────────────────────┘
```

## 🧪 Testing Pipeline

### 1. Verify Containers Running

```bash
docker-compose ps
# Output:
# NAME                      STATUS
# youtube-prefect-server    Up (healthy)
# youtube-prefect-db        Up (healthy)
# youtube-prefect-agent     Up
# youtube-jupyter           Up
```

### 2. Check Prefect Server UI

```
curl http://localhost:4200/api/health
```

### 3. Deploy Test Flow

```bash
# Inside prefect-agent or locally
prefect deployment build prefect/youtube_flow.py:youtube_analytics_flow \
  -n "youtube-pipeline" \
  -q default \
  --apply

# Trigger flow run
prefect deployment run youtube_analytics_flow/youtube-pipeline
```

### 4. Monitor Flow Execution

```bash
# Watch logs in real-time
docker-compose logs -f prefect-agent

# Or check Prefect UI: http://localhost:4200
```

## 🛠️ Troubleshooting

### 1. Containers không start

```bash
# Check logs chi tiết
docker-compose logs prefect-server

# Rebuild images
docker-compose build --no-cache

# Start with verbose logging
docker-compose up --verbose
```

### 2. Connection refused

```bash
# Kiểm tra network
docker network ls
docker network inspect youtube-network

# Restart containers
docker-compose restart
```

### 3. Permission denied

```bash
# Fix permissions (macOS/Linux)
sudo chmod -R 755 credentials/
sudo chown -R $(id -u):$(id -g) .
```

### 4. Out of memory

```bash
# Tăng Docker memory limit
# File: docker-compose.yml
services:
  prefect-server:
    deploy:
      resources:
        limits:
          memory: 2G  # Tăng từ 512M
```

## 📝 File Structure

```
youtube-analytics-pipeline/
├── docker/
│   ├── Dockerfile              # Main app image
│   └── Dockerfile.jupyter      # Jupyter image
├── prefect/
│   ├── youtube_flow.py         # Prefect flow definition
│   └── requirements.txt        # Prefect dependencies
├── extract/
│   ├── yt_pipeline.py          # ETL pipeline code
│   ├── schema_utilities.py     # Schema validators
│   └── requirements.txt        # Extract dependencies
├── credentials/                # GCP service account key (git ignored)
│   └── service-account-key.json
├── docker-compose.yml          # Docker compose config
├── requirements.txt            # Main dependencies
├── .env                        # Environment variables (git ignored)
├── .env.example               # Example env file
├── .dockerignore              # Docker build excludes
└── README.md                   # This file
```

## 🔐 Security Best Practices

1. **Never commit .env file** - Already in .gitignore
2. **Never commit credentials** - Already in .gitignore
3. **Use read-only volumes** for credentials (`:ro` flag)
4. **Change default passwords** - PostgreSQL password
5. **Use non-root user** - Containers run as `appuser`
6. **Limit container resources** - Set memory/CPU limits

## 📚 Thêm Tài Liệu

- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect GCP Integration](https://prefect-gcp.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Google Cloud APIs](https://cloud.google.com/apis/docs)

## 🤝 Contributing

Để cập nhật documentation này:

1. Edit file này
2. Test lại các commands
3. Submit PR

---

**Last Updated**: January 2026
**Prefect Version**: 3.0.0+
**Python Version**: 3.11+
