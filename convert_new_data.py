#!/usr/bin/env python3
"""
将 new/ 文件夹的数据转换为 YOLO 格式

策略:
- 使用 nc=1 (class-agnostic)，所有标注统一为 class 0
- 矩形框转换为 4 个角点的多边形格式
- 坐标归一化到 [0,1]
- 仅保留 label "1" 的标注（高质量标注）
"""

import json
import glob
import os
import shutil
from pathlib import Path

# 配置
INPUT_DIR = "new"
OUTPUT_DIR = "datasets/dami_new_yolo"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")


def convert_annotation(json_path):
    """转换单个 JSON 文件到 YOLO 格式 (nc=1, class-agnostic)"""
    with open(json_path, "r") as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]
    img_name = data["imagePath"]

    yolo_lines = []

    for shape in data["shapes"]:
        # 仅保留 label "1" 的标注（有用目标）
        label = str(shape.get("label", ""))
        if label != "1":
            continue

        # nc=1: 所有标注统一为 class 0
        class_id = 0

        points = shape.get("points", [])
        if not points:
            continue

        # 归一化坐标到 [0, 1]
        normalized_points = []
        for x, y in points:
            norm_x = max(0.0, min(1.0, x / img_width))
            norm_y = max(0.0, min(1.0, y / img_height))
            normalized_points.extend([norm_x, norm_y])

        # YOLO 格式: class x1 y1 x2 y2 x3 y3 x4 y4
        yolo_line = f"{class_id} " + " ".join(
            f"{coord:.6f}" for coord in normalized_points
        )
        yolo_lines.append(yolo_line)

    return img_name, yolo_lines


def main():
    # 确保输出目录存在
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    print(f"处理新数据集:")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  JSON 文件数: {len(json_files)}")
    print()

    converted_count = 0
    total_annotations = 0
    skipped_count = 0

    for json_path in json_files:
        try:
            img_name, yolo_lines = convert_annotation(json_path)

            if not yolo_lines:
                print(f"跳过 {json_path}: 无有效标注")
                skipped_count += 1
                continue

            # 写入标签文件
            base_name = Path(img_name).stem
            label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            # 复制图片
            src_img = os.path.join(INPUT_DIR, img_name)
            dst_img = os.path.join(IMAGES_DIR, img_name)

            if os.path.exists(src_img):
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)
            else:
                print(f"警告: 图片不存在: {src_img}")
                skipped_count += 1
                continue

            converted_count += 1
            total_annotations += len(yolo_lines)

            if converted_count % 50 == 0:
                print(f"已处理: {converted_count}/{len(json_files)}")

        except Exception as e:
            print(f"错误处理 {json_path}: {e}")
            skipped_count += 1

    print()
    print(f"转换完成:")
    print(f"  转换图片: {converted_count}")
    print(f"  跳过文件: {skipped_count}")
    print(f"  总标注数: {total_annotations}")
    print(f"  平均标注: {total_annotations / converted_count:.1f} 个/张")
    print(f"  图片目录: {IMAGES_DIR}")
    print(f"  标签目录: {LABELS_DIR}")


if __name__ == "__main__":
    main()
