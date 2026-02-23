import subprocess
import time
import sys


def run_command(cmd, description):
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "already exists" not in result.stderr.lower():
        print(f"Lỗi: {result.stderr}")
        return False
    print("Hoàn thành")
    return True


def main():
    print("=== Deploy Prefect Workflows từ Docker ===\n")
    
    if not run_command("docker compose up -d", "Bước 1: Khởi động Docker services"):
        sys.exit(1)
    
    print("\nBước 2: Đợi Prefect server sẵn sàng (60 giây)...")
    for i in range(60, 0, -10):
        print(f"  {i} giây...")
        time.sleep(10)
    
    run_command(
        "docker compose exec prefect-server prefect work-pool create default-pool --type process",
        "Bước 3: Tạo work pool"
    )
    
    if not run_command(
        "docker compose exec prefect-worker prefect deploy --all -c /app/config/prefect.yaml",
        "Bước 4: Deploy workflows"
    ):
        print("\nLỗi khi deploy. Kiểm tra logs:")
        print("  docker compose logs prefect-worker")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("Hoàn tất! Prefect UI: http://localhost:4200")
    print("="*50)
    print("\nCác lệnh hữu ích:")
    print("  docker compose logs -f prefect-worker    # Xem logs worker")
    print("  docker compose logs -f prefect-server    # Xem logs server")
    print("  docker compose ps                        # Kiểm tra services")
    print("  docker compose restart prefect-worker    # Restart worker")
    print("\nĐể dừng tất cả:")
    print("  docker compose down")


if __name__ == "__main__":
    main()
