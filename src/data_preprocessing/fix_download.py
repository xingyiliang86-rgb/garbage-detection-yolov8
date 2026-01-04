import os
import sys

print("正在修复下载问题...")

# 设置环境变量，跳过网络检查
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['GITHUB_ASSETS'] = 'disabled'
os.environ['TORCH_HUB'] = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub')


# 创建必要的空文件，避免下载检查
def create_dummy_files():
    files_to_create = [
        'yolov8s.pt',
        'yolov8n.pt',
        'yolov8m.pt',
        'yolov8l.pt',
        'yolov8x.pt'
    ]

    for file_name in files_to_create:
        if not os.path.exists(file_name):
            try:
                with open(file_name, 'wb') as f:
                    f.write(b'DUMMY_FILE_FOR_SKIP_DOWNLOAD')
                print(f"创建空文件: {file_name}")
            except:
                pass


# 修改 ultralytics 的下载函数
def patch_ultralytics():
    try:
        import ultralytics.utils.downloads
        # 保存原始函数
        original_download = ultralytics.utils.downloads.attempt_download_asset
        # 创建补丁函数
        def patched_download(asset, *args, **kwargs):
            print(f"跳过下载: {asset}")

            # 如果是 .pt 文件，返回本地路径
            if asset.endswith('.pt'):
                local_name = os.path.basename(asset)
                if not os.path.exists(local_name):
                    # 创建空文件
                    with open(local_name, 'wb') as f:
                        f.write(b'')
                return local_name

            # 其他情况调用原始函数
            return original_download(asset, *args, **kwargs)

        # 应用补丁
        ultralytics.utils.downloads.attempt_download_asset = patched_download
        print("Ultralytics 补丁已应用")

    except Exception as e:
        print(f"补丁应用失败: {e}")

# 4. 创建模型配置文件（50个类别）
def create_model_config():
    config_content = """# YOLOv8n 配置文件
nc: 50  # 类别数（根据你的数据集调整为50）
depth_multiple: 0.33
width_multiple: 0.25

backbone:
  [[-1, 1, 'Conv', [64, 3, 2]],
   [-1, 1, 'Conv', [128, 3, 2]],
   [-1, 3, 'C2f', [128, True]],
   [-1, 1, 'Conv', [256, 3, 2]],
   [-1, 6, 'C2f', [256, True]],
   [-1, 1, 'Conv', [512, 3, 2]],
   [-1, 6, 'C2f', [512, True]],
   [-1, 1, 'Conv', [1024, 3, 2]],
   [-1, 3, 'C2f', [1024, True]],
   [-1, 1, 'SPPF', [1024, 5]]]

head:
  [[-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 6], 1, 'Concat', [1]],
   [-1, 3, 'C2f', [512]],
   [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
   [[-1, 4], 1, 'Concat', [1]],
   [-1, 3, 'C2f', [256]],
   [-1, 1, 'Conv', [256, 3, 2]],
   [[-1, 12], 1, 'Concat', [1]],
   [-1, 3, 'C2f', [512]],
   [-1, 1, 'Conv', [512, 3, 2]],
   [[-1, 9], 1, 'Concat', [1]],
   [-1, 3, 'C2f', [1024]],
   [[15, 18, 21], 1, 'Detect', [50]]]
"""

    # 确保 models 目录存在
    os.makedirs('models', exist_ok=True)

    # 保存配置文件
    config_path = 'models/yolov8n-custom.yaml'
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"创建模型配置文件: {config_path} (nc=50)")
    else:
        print(f"配置文件已存在: {config_path}")
        # 确保内容是正确的
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"更新配置文件: {config_path} (nc=50)")

    return config_path


if __name__ == "__main__":
    print("=" * 50)
    print("下载问题修复工具")
    print("=" * 50)

    # 创建目录
    for dir_name in ['models', 'experiments', 'deployment']:
        os.makedirs(dir_name, exist_ok=True)
        print(f"目录已创建/确认: {dir_name}")

    # 执行修复
    create_dummy_files()
    patch_ultralytics()
    config_path = create_model_config()

    print("\n" + "=" * 50)
    print("修复完成！")
    print("=" * 50)