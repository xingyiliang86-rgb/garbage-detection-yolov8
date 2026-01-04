import argparse
from ultralytics import YOLO
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description='YOLOv8垃圾检测测试')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='模型路径，默认: models/best.pt')
    parser.add_argument('--source', type=str, default='test_images/',
                        help='测试图片路径，默认: test_images/')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值，默认: 0.25')
    args = parser.parse_args()

    print("=" * 50)
    print("YOLOv8垃圾检测系统")
    print(f"模型: {args.model}")
    print(f"测试源: {args.source}")
    print(f"置信度: {args.conf}")
    print("=" * 50)

    # 加载模型
    print("加载模型中...")
    try:
        model = YOLO(args.model)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 创建输出目录
    os.makedirs('runs/detect', exist_ok=True)

    # 执行推理
    print("开始推理...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        save_txt=True,
        project='runs/detect',
        name='test',
        exist_ok=True
    )

    print("\n" + "=" * 50)
    print("测试完成！")
    print(f"结果保存在: runs/detect/test/")
    print("=" * 50)


if __name__ == '__main__':
    main()