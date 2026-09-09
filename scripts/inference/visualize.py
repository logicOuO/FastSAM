#!/usr/bin/env python3
"""运行 FastSAM 验证集推理并输出可视化图和合并二值掩码。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "train" / "fastsam" / "weights" / "best.pt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "fastsam" / "images" / "val"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "visualize"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args():
    """解析可视化参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="模型权重")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入图片目录")
    parser.add_argument("--device", default="0", help="推理设备，例如 0 或 cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IOU 阈值")
    return parser.parse_args()


def main():
    """执行推理并生成合并二值掩码。"""
    args = parse_args()
    model_path = args.model.resolve()
    input_dir = args.input.resolve()
    image_files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    if not image_files:
        raise FileNotFoundError(f"没有找到验证图片: {input_dir}")

    output_dir = DEFAULT_PROJECT / "fastsam"
    mask_dir = output_dir / "binary_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {model_path}")
    print(f"输入目录: {input_dir}")
    print(f"图片数量: {len(image_files)}")
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(input_dir),
        imgsz=1920,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=False,
        save_conf=False,
        project=str(DEFAULT_PROJECT),
        name="fastsam",
        exist_ok=True,
        show_labels=False,
        show_conf=True,
        line_width=1,
        device=args.device,
        retina_masks=True,
    )

    total_detections = 0
    total_masks = 0
    for result in results:
        image_name = Path(result.path).name
        if result.boxes is not None:
            total_detections += len(result.boxes)
        if result.masks is None:
            continue

        image_height, image_width = result.orig_shape
        binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for mask in result.masks.data:
            mask_array = mask.cpu().numpy()
            if mask_array.shape != binary_mask.shape:
                mask_array = cv2.resize(mask_array, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            binary_mask = cv2.bitwise_or(binary_mask, (mask_array > 0.5).astype(np.uint8) * 255)

        output_path = mask_dir / image_name
        if not cv2.imwrite(str(output_path), binary_mask):
            raise OSError(f"二值掩码保存失败: {output_path}")
        total_masks += 1

    average = total_detections / len(results) if results else 0.0
    print("\n可视化完成！")
    print(f"  可视化结果: {output_dir}")
    print(f"  二值掩码: {mask_dir}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张: {average:.1f} 个")
    print(f"  生成掩码图: {total_masks} 张")


if __name__ == "__main__":
    main()
