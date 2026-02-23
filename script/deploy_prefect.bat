@echo off
setlocal

echo Buoc 1: Khoi dong Docker services...
docker compose up -d

echo.
echo Buoc 2: Doi Prefect server san sang (60 giay)...
timeout /t 60 /nobreak

echo.
echo Buoc 3: Tao work pool...
docker compose exec prefect-server prefect work-pool create default-pool --type process 2>nul || echo Work pool da ton tai

echo.
echo Buoc 4: Deploy workflows tu config/prefect.yaml...
docker compose exec prefect-worker prefect deploy --all -c /app/config/prefect.yaml

echo.
echo Hoan tat! Truy cap Prefect UI tai: http://localhost:4200
echo.
echo Cac lenh huu ich:
echo   docker compose logs -f prefect-worker    # Xem logs worker
echo   docker compose logs -f prefect-server    # Xem logs server
echo   docker compose ps                        # Kiem tra services
echo   docker compose restart prefect-worker    # Restart worker
