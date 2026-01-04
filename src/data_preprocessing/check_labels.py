import os
import glob

# 设置标签目录路径
label_dir = r"D:\visiondata\data\label"

max_class_id = 0
problem_files = []

print(f"正在检查目录: {label_dir}")
print(f"目录存在: {os.path.exists(label_dir)}")

# 统计标签文件数量
label_files = list(glob.glob(os.path.join(label_dir, "*.txt")))
print(f"找到 {len(label_files)} 个标签文件")

# 检查每个标签文件
for i, label_file in enumerate(label_files):
    if i % 100 == 0:
        print(f"已处理 {i}/{len(label_files)} 个文件...")

    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:  # 确保不是空行
                        # 类别索引是第一个数字
                        class_id = int(float(parts[0]))
                        if class_id > max_class_id:
                            max_class_id = class_id
                        if class_id >= 50:  # 超过50的就是问题
                            problem_files.append((label_file, line_num, class_id))
    except UnicodeDecodeError:
        # 如果utf-8不行，尝试其他编码
        with open(label_file, 'r', encoding='gbk') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        class_id = int(float(parts[0]))
                        if class_id > max_class_id:
                            max_class_id = class_id
                        if class_id >= 50:
                            problem_files.append((label_file, line_num, class_id))
    except Exception as e:
        print(f"读取文件 {label_file} 时出错: {e}")

print(f"\n{'=' * 50}")
print(f"最大类别索引: {max_class_id}")
print(f"发现 {len(problem_files)} 个有问题标签:")

# 显示前20个问题文件
for file, line_num, class_id in problem_files[:20]:
    file_name = os.path.basename(file)
    print(f"  {file_name}: 第 {line_num} 行 - 类别索引 {class_id}")

# 统计各类别出现次数
if len(problem_files) > 0:
    print(f"\n问题类别分布:")
    class_counts = {}
    for _, _, class_id in problem_files:
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 次")

print(f"\n建议:")
print(f"如果最大索引是 {max_class_id}，需要在 data.yaml 中将 nc 设置为 {max_class_id + 1}")