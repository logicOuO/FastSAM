#!/usr/bin/env python3
"""
可视化测试新训练的模型 - 保存二值化 mask

输出:
1. 原图 + box + mask 叠加（彩色）
2. 二值化 mask 图（黑白，所有目标合并为白色）
"""

from ultralytics import YOLO
import os
import cv2
import numpy as np

# 加载最佳模型
model_path = "runs/train/fastsam_dami_new/weights/best.pt"
print(f"加载模型: {model_path}")
model = YOLO(model_path)

# 验证集图片路径
val_images_dir = "datasets/dami_new_yolo/images/val"
output_dir = "runs/visualize/dami_new_final"
mask_dir = os.path.join(output_dir, "binary_masks")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

print(f"\n在验证集上运行推理...")
print(f"  输入目录: {val_images_dir}")
print(f"  可视化输出: {output_dir}")
print(f"  二值化 mask: {mask_dir}")
print(
    f"  图片数量: {len([f for f in os.listdir(val_images_dir) if f.endswith('.png')])}"
)

# 运行推理
results = model.predict(
    source=val_images_dir,
    imgsz=1920,
    conf=0.25,  # 置信度阈值
    iou=0.7,  # NMS IOU 阈值
    save=True,  # 保存可视化结果
    save_txt=False,
    save_conf=False,
    project="runs/visualize",
    name="dami_new_final",
    exist_ok=True,
    show_labels=False,  # 不显示标签（class-agnostic）
    show_conf=True,  # 显示置信度
    line_width=1,
    device=0,
)

print(f"\n生成二值化 mask 图...")

# 统计
total_detections = 0
total_masks = 0

for result in results:
    # 获取原图文件名
    img_path = result.path
    img_name = os.path.basename(img_path)

    # 统计检测数
    if result.boxes is not None:
        total_detections += len(result.boxes)

    # 生成二值化 mask
    if result.masks is not None:
        # 获取图片尺寸
        img_height, img_width = result.orig_shape

        # 创建空白 mask（黑色背景）
        binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 合并所有检测到的 mask
        for mask in result.masks.data:
            # mask 是 tensor，转为 numpy
            mask_np = mask.cpu().numpy()

            # 二值化（>0.5 为前景）
            mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

            # 合并到总 mask（OR 操作）
            binary_mask = cv2.bitwise_or(binary_mask, mask_binary)

        # 保存二值化 mask
        mask_save_path = os.path.join(mask_dir, img_name)
        cv2.imwrite(mask_save_path, binary_mask)

        total_masks += 1

print(f"\n完成!")
print(f"  可视化结果: {output_dir}")
print(f"  二值化 mask: {mask_dir}")
print(f"  总检测数: {total_detections}")
print(f"  平均每张: {total_detections / len(results):.1f} 个")
print(f"  生成 mask 图: {total_masks} 张")
