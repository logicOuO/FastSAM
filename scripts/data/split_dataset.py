#!/usr/bin/env python3
"""以可复现方式将转换后的数据集划分为训练集和验证集。"""

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "fastsam"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args():
    """解析数据划分参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="YOLO 数据集目录")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def count_annotations(labels_dir):
    """统计目录内的标注实例数。"""
    return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in labels_dir.glob("*.txt"))


def partition_images(images, train_ratio, seed):
    """按固定随机种子划分图片列表。"""
    if not 0 < train_ratio < 1:
        raise ValueError("训练集比例必须位于 0 和 1 之间")
    shuffled_images = list(images)
    random.Random(seed).shuffle(shuffled_images)
    train_count = int(len(shuffled_images) * train_ratio)
    train_images = shuffled_images[:train_count]
    val_images = shuffled_images[train_count:]
    if not train_images or not val_images:
        raise ValueError("划分结果为空，请调整训练集比例或增加样本")
    return train_images, val_images


def main():
    """校验输入后移动图片和对应标签。"""
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("训练集比例必须位于 0 和 1 之间")

    dataset_dir = args.dataset.resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    split_dirs = {
        "train": (images_dir / "train", labels_dir / "train"),
        "val": (images_dir / "val", labels_dir / "val"),
    }
    all_images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    if not all_images:
        existing_images = sum(
            len(list(image_dir.glob("*")))
            for image_dir, _ in split_dirs.values()
            if image_dir.exists()
        )
        if existing_images:
            print(f"数据集已经完成划分，共 {existing_images} 张图片: {dataset_dir}")
            return
        raise FileNotFoundError(f"没有找到待划分图片: {images_dir}")

    missing_labels = [
        labels_dir / f"{path.stem}.txt"
        for path in all_images
        if not (labels_dir / f"{path.stem}.txt").is_file()
    ]
    if missing_labels:
        missing_text = "\n".join(f"- {path}" for path in missing_labels[:10])
        raise FileNotFoundError(f"存在缺失标签，已停止划分：\n{missing_text}")

    train_images, val_images = partition_images(all_images, args.train_ratio, args.seed)

    print(f"数据集划分 ({args.train_ratio:.0%}/{1 - args.train_ratio:.0%}):")
    print(f"  总图片数: {len(all_images)}")
    print(f"  随机种子: {args.seed}")
    print()

    print("划分结果:")
    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")
    print()

    for image_dir, label_dir in split_dirs.values():
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    def move_files(images, split):
        image_dir, label_dir = split_dirs[split]
        for image_path in images:
            shutil.move(str(image_path), image_dir / image_path.name)
            label_path = labels_dir / f"{image_path.stem}.txt"
            shutil.move(str(label_path), label_dir / label_path.name)

    move_files(train_images, "train")
    move_files(val_images, "val")

    print()
    print("划分完成:")
    print(f"  训练集图片: {split_dirs['train'][0]}")
    print(f"  训练集标签: {split_dirs['train'][1]}")
    print(f"  验证集图片: {split_dirs['val'][0]}")
    print(f"  验证集标签: {split_dirs['val'][1]}")

    train_annotations = count_annotations(split_dirs["train"][1])
    val_annotations = count_annotations(split_dirs["val"][1])

    print()
    print("标注统计:")
    print(f"  训练集标注: {train_annotations} 个")
    print(f"  验证集标注: {val_annotations} 个")
    print(f"  总标注数: {train_annotations + val_annotations} 个")


if __name__ == "__main__":
    main()
