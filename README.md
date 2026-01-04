# 基于YOLOv8的垃圾检测与分类系统

## 项目简介
这是一个基于YOLOv8的实时垃圾检测与分类系统，包含数据预处理、模型训练和验证的完整流程。

## 项目结构


## 使用方法
1. 克隆项目：`git clone <仓库地址>`
2. 安装依赖：`pip install -r requirements.txt`
3. 数据预处理：`python src/data_preprocessing/check_labels.py`
4. 创建数据配置：`python src/data_preprocessing/create_data_config.py`
5. 训练模型：`python src/training/garbage_detector.py --data configs/garbage.yaml`

## 作者
xingyiliang86-rgb