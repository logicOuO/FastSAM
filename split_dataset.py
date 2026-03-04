#!/usr/bin/env python3
"""
将数据集划分为训练集和验证集 (90/10)

策略:
- 随机打乱所有图片
- 90% 训练集 (199张)
- 10% 验证集 (22张)
- 设置随机种子保证可复现
"""

import os
import shutil
import random
from pathlib import Path

# 配置
DATASET_DIR = "datasets/dami_new_yolo"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

TRAIN_RATIO = 0.9
RANDOM_SEED = 42


def main():
    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 获取所有图片文件
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]

    if not all_images:
        print(f"错误: 在 {IMAGES_DIR} 中未找到图片文件")
        return

    print(f"数据集划分 (90/10):")
    print(f"  总图片数: {len(all_images)}")
    print(f"  随机种子: {RANDOM_SEED}")
    print()

    # 随机打乱
    random.shuffle(all_images)

    # 计算划分点
    train_count = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:train_count]
    val_images = all_images[train_count:]

    print(f"划分结果:")
    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")
    print()

    # 创建目录
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)

    # 移动训练集
    print("移动训练集文件...")
    for img_name in train_images:
        # 移动图片
        src_img = os.path.join(IMAGES_DIR, img_name)
        dst_img = os.path.join(TRAIN_IMAGES_DIR, img_name)
        shutil.move(src_img, dst_img)

        # 移动标签
        label_name = Path(img_name).stem + ".txt"
        src_label = os.path.join(LABELS_DIR, label_name)
        dst_label = os.path.join(TRAIN_LABELS_DIR, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

    # 移动验证集
    print("移动验证集文件...")
    for img_name in val_images:
        # 移动图片
        src_img = os.path.join(IMAGES_DIR, img_name)
        dst_img = os.path.join(VAL_IMAGES_DIR, img_name)
        shutil.move(src_img, dst_img)

        # 移动标签
        label_name = Path(img_name).stem + ".txt"
        src_label = os.path.join(LABELS_DIR, label_name)
        dst_label = os.path.join(VAL_LABELS_DIR, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

    print()
    print("划分完成:")
    print(f"  训练集图片: {TRAIN_IMAGES_DIR}")
    print(f"  训练集标签: {TRAIN_LABELS_DIR}")
    print(f"  验证集图片: {VAL_IMAGES_DIR}")
    print(f"  验证集标签: {VAL_LABELS_DIR}")

    # 统计标注数量
    train_annotations = 0
    for label_file in os.listdir(TRAIN_LABELS_DIR):
        with open(os.path.join(TRAIN_LABELS_DIR, label_file)) as f:
            train_annotations += len(f.readlines())

    val_annotations = 0
    for label_file in os.listdir(VAL_LABELS_DIR):
        with open(os.path.join(VAL_LABELS_DIR, label_file)) as f:
            val_annotations += len(f.readlines())

    print()
    print("标注统计:")
    print(f"  训练集标注: {train_annotations} 个")
    print(f"  验证集标注: {val_annotations} 个")
    print(f"  总标注数: {train_annotations + val_annotations} 个")


if __name__ == "__main__":
    main()
