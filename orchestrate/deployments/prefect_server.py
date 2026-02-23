import subprocess
import time
import os
import sys
from pathlib import Path


def start_prefect_server():
    print("Đang khởi động Prefect server...")
    print("Prefect UI sẽ chạy tại: http://localhost:4200")
    
    try:
        subprocess.Popen(
            ["prefect", "server", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("Prefect server đang khởi động...")
        print("Đợi 10 giây để server sẵn sàng...")
        time.sleep(10)
        
        return True
    except Exception as e:
        print(f"Lỗi khi khởi động server: {e}")
        return False


def start_worker():
    print("\nĐang khởi động Prefect worker...")
    
    try:
        subprocess.run(
            ["prefect", "worker", "start", "--pool", "default-agent-pool"],
            check=True
        )
    except KeyboardInterrupt:
        print("\nWorker đã dừng")
    except Exception as e:
        print(f"Lỗi khi khởi động worker: {e}")


def check_prefect_status():
    try:
        result = subprocess.run(
            ["prefect", "deployment", "ls"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Danh sách deployments hiện tại:\n")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError:
        print("Không thể kết nối tới Prefect server")
        print("Hãy chạy: python orchestrate/deployments/prefect_server.py start")
        return False
    except FileNotFoundError:
        print("Prefect chưa được cài đặt")
        print("Hãy chạy: pip install prefect")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quản lý Prefect Server")
    parser.add_argument(
        "action",
        choices=["start", "worker", "status"],
        help="start: Khởi động server | worker: Khởi động worker | status: Kiểm tra trạng thái"
    )
    
    args = parser.parse_args()
    
    if args.action == "start":
        start_prefect_server()
        print("\nCác bước tiếp theo:")
        print("  1. Mở trình duyệt: http://localhost:4200")
        print("  2. Deploy workflows: python orchestrate/deployments/deploy_daily_schedule.py")
        print("  3. Khởi động worker: python orchestrate/deployments/prefect_server.py worker")
        
    elif args.action == "worker":
        print("Worker sẽ chạy các scheduled flows tự động")
        print("Nhấn Ctrl+C để dừng worker\n")
        start_worker()
        
    elif args.action == "status":
        check_prefect_status()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Hướng dẫn sử dụng:")
        print("\n1. Khởi động Prefect server:")
        print("   python orchestrate/deployments/prefect_server.py start")
        print("\n2. Deploy workflows với lịch:")
        print("   python orchestrate/deployments/deploy_daily_schedule.py")
        print("\n3. Khởi động worker để chạy scheduled jobs:")
        print("   python orchestrate/deployments/prefect_server.py worker")
        print("\n4. Kiểm tra trạng thái:")
        print("   python orchestrate/deployments/prefect_server.py status")
        print("\nPrefect UI: http://localhost:4200")
        sys.exit(0)
    
    main()
