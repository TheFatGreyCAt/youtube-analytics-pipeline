import subprocess
import sys


def start_prefect_server():
    print("Đang khởi động Prefect server...")
    
    try:
        subprocess.run(["prefect", "server", "start"], check=True)
    except KeyboardInterrupt:
        print("\nServer đã dừng")
    except Exception as e:
        print(f"Lỗi: {e}")


def deploy_workflows():
    print("Đang deploy workflows từ prefect.yaml...")
    
    try:
        result = subprocess.run(
            ["prefect", "deploy", "--all"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("\nĐã deploy thành công!")
        print("Prefect UI: http://localhost:4200")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi deploy: {e.stderr}")


def start_worker():
    print("Đang khởi động worker...")
    print("Worker sẽ chạy các scheduled flows tự động")
    
    try:
        subprocess.run(["prefect", "worker", "start", "--pool", "default-agent-pool"], check=True)
    except KeyboardInterrupt:
        print("\nWorker đã dừng")


def check_status():
    try:
        result = subprocess.run(
            ["prefect", "deployment", "ls"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Danh sách deployments:\n")
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("Không thể kết nối tới Prefect server")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quản lý Prefect")
    parser.add_argument(
        "action",
        choices=["server", "deploy", "worker", "status"],
        help="server: Khởi động server | deploy: Deploy workflows | worker: Khởi động worker | status: Kiểm tra"
    )
    
    args = parser.parse_args()
    
    if args.action == "server":
        start_prefect_server()
    elif args.action == "deploy":
        deploy_workflows()
    elif args.action == "worker":
        start_worker()
    elif args.action == "status":
        check_status()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Hướng dẫn sử dụng Prefect 3.x:")
        print("\n1. Khởi động server:")
        print("   python orchestrate/deployments/prefect_manager.py server")
        print("\n2. Deploy workflows (terminal mới):")
        print("   python orchestrate/deployments/prefect_manager.py deploy")
        print("\n3. Khởi động worker:")
        print("   python orchestrate/deployments/prefect_manager.py worker")
        print("\n4. Kiểm tra status:")
        print("   python orchestrate/deployments/prefect_manager.py status")
        print("\nPrefect UI: http://localhost:4200")
        sys.exit(0)
    
    main()
