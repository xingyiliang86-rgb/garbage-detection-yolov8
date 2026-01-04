import requests
import os


def download_file(url, filename):
    print(f"正在下载 {filename}...")

    # 创建目录
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 下载文件
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

    print(f"\n 下载完成: {filename}")


if __name__ == "__main__":
    # 下载 YOLOv8n 模型
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    download_file(url, "models/yolov8n.pt")

    print("\n模型下载完成！可以运行测试了。")
    print("运行: python test.py")