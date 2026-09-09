#!/usr/bin/env python3
"""将原始 LabelMe 标注转换为 FastSAM 训练数据。"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "fastsam"


def convert_annotation(json_path):
    """转换单个 JSON 标注，返回图片名及 YOLO 标注行。"""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"图片尺寸非法: {json_path}")

    # 只使用文件名，避免标注中的绝对路径或上级目录逃逸输出目录。
    img_name = Path(data["imagePath"]).name

    yolo_lines = []

    for shape in data["shapes"]:
        # 仅保留 label "1" 的标注（有用目标）
        label = str(shape.get("label", ""))
        if label != "1":
            continue

        # nc=1: 所有标注统一为 class 0
        class_id = 0

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        # 归一化坐标到 [0, 1]
        normalized_points = []
        for x, y in points:
            norm_x = max(0.0, min(1.0, x / img_width))
            norm_y = max(0.0, min(1.0, y / img_height))
            normalized_points.extend([norm_x, norm_y])

        # YOLO 分割格式: class x1 y1 x2 y2 x3 y3 ...
        yolo_line = f"{class_id} " + " ".join(
            f"{coord:.6f}" for coord in normalized_points
        )
        yolo_lines.append(yolo_line)

    return img_name, yolo_lines


def parse_args():
    """解析数据转换参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="原始数据目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="YOLO 数据集目录")
    return parser.parse_args()


def main():
    """执行整个数据集转换。"""
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在: {input_dir}")
    if (images_dir / "train").exists() or (images_dir / "val").exists():
        raise FileExistsError(f"输出数据集已经完成划分，请勿重复转换: {output_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"没有找到 JSON 标注: {input_dir}")

    print("处理新数据集:")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
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

            src_img = input_dir / img_name
            if not src_img.is_file():
                print(f"警告: 图片不存在: {src_img}")
                skipped_count += 1
                continue

            base_name = Path(img_name).stem
            label_path = labels_dir / f"{base_name}.txt"
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            shutil.copy2(src_img, images_dir / img_name)

            converted_count += 1
            total_annotations += len(yolo_lines)

            if converted_count % 50 == 0:
                print(f"已处理: {converted_count}/{len(json_files)}")

        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"错误处理 {json_path}: {exc}")
            skipped_count += 1

    print()
    if converted_count == 0:
        raise RuntimeError("没有成功转换任何样本")

    print("转换完成:")
    print(f"  转换图片: {converted_count}")
    print(f"  跳过文件: {skipped_count}")
    print(f"  总标注数: {total_annotations}")
    print(f"  平均标注: {total_annotations / converted_count:.1f} 个/张")
    print(f"  图片目录: {images_dir}")
    print(f"  标签目录: {labels_dir}")


if __name__ == "__main__":
    main()
