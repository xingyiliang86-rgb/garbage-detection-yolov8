import os
import sys
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# YOLO相关导入
from ultralytics import YOLO
import ultralytics

# 禁用自动下载和更新检查
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['GITHUB_ASSETS'] = 'disabled'
os.environ['YOLO_DOWNLOAD_TIMEOUT'] = '10'
os.environ['YOLO_HOME'] = str(Path.cwd() / '.yolo')
torch.set_num_threads(2)

class CPUOptimizedTrainer:

    def __init__(self, args):
        self.args = args
        self.device = 'cpu'
        self.experiment_name = self.create_experiment_name()

        # 创建实验目录
        self.exp_dir = Path('experiments') / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存实验配置
        self.save_experiment_config()

        print("=" * 60)
        print("CPU优化垃圾检测训练系统")
        print("=" * 60)
        print(f"实验名称: {self.experiment_name}")
        print(f"实验目录: {self.exp_dir}")
        print(f"训练设备: {self.device}")
        print(f"PyTorch线程数: {torch.get_num_threads()}")

    def create_experiment_name(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.args.exp_name if self.args.exp_name else f"garbage_cpu_{timestamp}"
        return base_name

    def save_experiment_config(self):
        config = vars(self.args)
        config['experiment_name'] = self.experiment_name
        config['start_time'] = datetime.now().isoformat()
        config['device'] = self.device

        config_file = self.exp_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self):
        model_path = self.args.model

        # 检查是否为.pt文件且不存在
        if model_path.endswith('.pt') and not os.path.exists(model_path):
            print(f"模型文件 {model_path} 不存在，使用配置文件")

            # 检查自定义配置文件是否存在
            custom_config = 'models/yolov8n-custom.yaml'
            if os.path.exists(custom_config):
                model = YOLO(custom_config)
                print(f"使用自定义配置文件: {custom_config}")
            else:
                # 创建简单的配置文件
                self.create_simple_config()
                model = YOLO('models/yolov8n-custom.yaml')
                print("使用创建的配置文件")
        else:
            # 尝试加载
            try:
                model = YOLO(model_path)
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("使用默认配置从头训练")
                self.create_simple_config()
                model = YOLO('models/yolov8n-custom.yaml')

        return model

        def create_simple_config(self):
            """创建简单的模型配置文件"""
            config_content = """# YOLOv8n 配置文件
     nc: 60  # 修改为50，匹配标签文件
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
        [[15, 18, 21], 1, 'Detect', [nc]]]
     """
            # 确保models目录存在
            os.makedirs('models', exist_ok=True)

            # 保存配置文件
            config_path = 'models/yolov8n-custom.yaml'
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)

            print(f"创建配置文件: {config_path} (nc=50)")
            return config_path

    def get_training_params(self):
        params = {
            # 基础参数
            'data': self.args.data,
            'epochs': self.args.epochs,
            'imgsz': self.args.imgsz,
            'batch': self.args.batch,
            'workers': min(2, self.args.workers),  # CPU上减少workers
            'device': self.device,
            'patience': self.args.patience,
            'save': True,
            'save_period': 10,
            'exist_ok': True,
            'project': str(self.exp_dir),
            'name': 'train',

            # 学习率参数
            'lr0': self.args.lr,
            'lrf': self.args.lrf,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # 数据增强
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.5,  # 降低mosaic概率减少计算
            'mixup': 0.1,  # 降低mixup概率

            # 其他优化
            'label_smoothing': 0.1,
            'close_mosaic': 5,
            'cos_lr': True,
            'verbose': False,  # 减少输出
            'plots': True,
            'cache': False,  # 禁用缓存减少内存使用
        }

        # 根据图像尺寸调整参数
        if self.args.imgsz <= 320:
            params['batch'] = min(16, params['batch'])  # 小尺寸可以增大batch
        else:
            params['batch'] = min(4, params['batch'])  # 大尺寸减小batch

        return params

    def train(self):
        print("\n开始训练过程...")

        # 1. 检查数据配置文件
        if not os.path.exists(self.args.data):
            print(f"错误: 数据配置文件不存在: {self.args.data}")
            print("请确保文件存在或使用绝对路径")
            return None, None

        # 2. 加载模型
        print("加载模型...")
        model = self.load_model()

        # 3. 获取训练参数
        train_params = self.get_training_params()

        # 4. 显示训练参数
        print("\n训练参数:")
        print(f"- 轮次: {train_params['epochs']}")
        print(f"- 批次: {train_params['batch']}")
        print(f"- 图像尺寸: {train_params['imgsz']}")
        print(f"- 学习率: {train_params['lr0']}")
        print(f"- 数据增强: HSV+翻转+马赛克")

        # 5. 开始训练
        print(f"\n开始训练 {train_params['epochs']} 轮次...")
        start_time = datetime.now()

        try:
            results = model.train(**train_params)
            training_time = (datetime.now() - start_time).total_seconds()

            print(f"\n训练完成!")
            print(f"训练耗时: {training_time:.1f}秒 ({training_time / 60:.1f}分钟)")

            # 保存训练摘要
            self.save_training_summary(results, training_time, model)

            return results, model

        except Exception as e:
            print(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_training_summary(self, results, training_time, model):
        summary_path = self.exp_dir / 'training_summary.txt'

        with open(summary_path, 'w') as f:
            f.write("训练摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"实验名称: {self.experiment_name}\n")
            f.write(f"训练时间: {training_time:.1f}秒\n")
            f.write(f"训练轮次: {self.args.epochs}\n")
            f.write(f"批次大小: {self.args.batch}\n")
            f.write(f"图像尺寸: {self.args.imgsz}\n")

            # 保存最佳模型路径
            if hasattr(model, 'trainer') and model.trainer.save_dir:
                best_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
                if best_path.exists():
                    f.write(f"最佳模型: {best_path}\n")

                    # 验证最佳模型
                    print("\n验证最佳模型...")
                    val_results = model.val(data=self.args.data, imgsz=self.args.imgsz, batch=4)

                    if hasattr(val_results, 'box'):
                        f.write(f"mAP50: {val_results.box.map50:.6f}\n")
                        f.write(f"精确率: {val_results.box.p:.6f}\n")
                        f.write(f"召回率: {val_results.box.r:.6f}\n")

                        print(f"验证结果 - mAP50: {val_results.box.map50:.6f}")

        print(f"训练摘要保存到: {summary_path}")


class QuickValidator:

    @staticmethod
    def validate(model_path, data_yaml, imgsz=320):
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return

        print(f"验证模型: {model_path}")

        model = YOLO(model_path)
        results = model.val(data=data_yaml, imgsz=imgsz, batch=4, device='cpu', verbose=False)

        if hasattr(results, 'box'):
            print(f"mAP50: {results.box.map50:.6f}")
            print(f"mAP50-95: {results.box.map:.6f}")
            print(f"精确率: {results.box.p:.6f}")
            print(f"召回率: {results.box.r:.6f}")

        return results

    @staticmethod
    def test_image(model_path, image_path, output_dir='test_results'):
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return

        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return

        os.makedirs(output_dir, exist_ok=True)

        model = YOLO(model_path)
        results = model.predict(image_path, imgsz=320, conf=0.25, save=True, project=output_dir)

        if results and len(results) > 0:
            print(f"检测到 {len(results[0].boxes)} 个目标")
            print(f"结果保存到: {output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(description='CPU优化垃圾检测训练系统')

    # 基础参数
    parser.add_argument('--data', type=str, required=True, help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.yaml', help='模型文件或配置')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=320, help='图像尺寸')
    parser.add_argument('--workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率因子')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')

    # 其他参数
    parser.add_argument('--exp-name', type=str, help='实验名称')
    parser.add_argument('--validate', type=str, help='验证模型路径')
    parser.add_argument('--test', type=str, help='测试图片路径')

    args = parser.parse_args()

    # 创建必要目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)

    # 验证模式
    if args.validate:
        QuickValidator.validate(args.validate, args.data, args.imgsz)
        return

    # 测试模式
    if args.test:
        QuickValidator.test_image(args.validate, args.test)
        return

    # 训练模式
    trainer = CPUOptimizedTrainer(args)
    results, model = trainer.train()

    if results and model:
        print(f"\n训练完成!")
        print(f"实验结果保存在: experiments/{trainer.experiment_name}")

        # 自动验证最佳模型
        if hasattr(model, 'trainer') and model.trainer.save_dir:
            best_model = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
            if best_model.exists():
                print(f"\n自动验证最佳模型...")
                QuickValidator.validate(str(best_model), args.data, args.imgsz)

    print("\n" + "=" * 60)
    print("程序结束")
    print("=" * 60)


if __name__ == '__main__':
    main()