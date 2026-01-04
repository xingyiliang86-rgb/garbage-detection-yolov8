import yaml
import os

def create_garbage_yaml():
    # 原始48个类别
    original_names = [
        'Aluminium foil', 'Cadded blister pack', 'Other plastic bottle', 'Clear plastic bottle',
        'Glass bottle', 'Plastic bottle cap', 'Metal bottle cap', 'Broken glass', 'Food Can',
        'Aerosol', 'Drink can', 'Other carton', 'Drink carton', 'Corrugated carton', 'Meal carton',
        'Paper cup', 'Disposable plastic cup', 'Foam cup', 'Glass cup', 'Other plastic cup',
        'Food waste', 'Glass jar', 'Plastic lid', 'Metal lid', 'Other plastic', 'Tissues',
        'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastic film', 'Six pack rings',
        'Garbage bag', 'Other plastic wrapper', 'Single-use carrier bag', 'Crisp packet',
        'Disposable food container', 'Foam food container', 'Other plastic container',
        'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe',
        'Plastic straw', 'Paper straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette'
    ]

    if len(original_names) < 50:
        extra_needed = 50 - len(original_names)
        for i in range(extra_needed):
            original_names.append(f"extra_class_{len(original_names)}")

    names = original_names

    base_path = "D:/visiondata/data"

    config = {
        'path': base_path,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 50,
        'names': names
    }

    # 保存配置文件
    output_file = "data/garbage.yaml"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"数据配置文件已创建: {output_file}")
    print(f"类别数量: 50")
    print(f"基础路径: {base_path}")
    print(f"训练路径: images/train")
    print(f"验证路径: images/val")

    return output_file

if __name__ == "__main__":
    create_garbage_yaml()