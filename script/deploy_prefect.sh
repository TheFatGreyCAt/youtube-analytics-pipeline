#!/bin/bash

set -e

echo "Bước 1: Khởi động Docker services..."
docker compose up -d

echo ""
echo "Bước 2: Đợi Prefect server sẵn sàng (60 giây)..."
sleep 60

echo ""
echo "Bước 3: Tạo work pool..."
docker compose exec prefect-server prefect work-pool create default-pool --type process || echo "Work pool đã tồn tại"

echo ""
echo "Bước 4: Deploy workflows từ config/prefect.yaml..."
docker compose exec prefect-worker prefect deploy --all -c /app/config/prefect.yaml

echo ""
echo "Hoàn tất! Truy cập Prefect UI tại: http://localhost:4200"
echo ""
echo "Các lệnh hữu ích:"
echo "  docker compose logs -f prefect-worker    # Xem logs worker"
echo "  docker compose logs -f prefect-server    # Xem logs server"
echo "  docker compose ps                        # Kiểm tra services"
echo "  docker compose restart prefect-worker    # Restart worker"
