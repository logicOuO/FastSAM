#!/usr/bin/env python3
"""Re-visualize with better display options"""

from ultralytics import YOLO
import glob

model = YOLO("runs/train/fastsam_dami_1920/weights/best.pt")
image_files = sorted(glob.glob("dami/*.png"))[:5]

print("选项1: 只显示框，不显示文字标签")
results = model.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_1920_no_labels",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,  # 不显示标签
    show_conf=False,  # 不显示置信度
    line_width=2,  # 框线宽度
)
print("✓ 保存到: runs/visualize/dami_1920_no_labels/")

print("\n选项2: 显示标签但框更粗")
results = model.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_1920_thick_box",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=True,
    show_conf=True,
    line_width=4,  # 更粗的框
)
print("✓ 保存到: runs/visualize/dami_1920_thick_box/")

print("\n选项3: 只显示置信度，不显示类别名")
results = model.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_1920_conf_only",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,  # 不显示类别
    show_conf=True,  # 只显示置信度
    line_width=3,
)
print("✓ 保存到: runs/visualize/dami_1920_conf_only/")

print("\n完成！请查看三个文件夹选择你喜欢的样式")
