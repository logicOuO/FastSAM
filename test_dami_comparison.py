#!/usr/bin/env python3
"""Visualize dami 1920 model predictions and compare with 640 model"""

from ultralytics import YOLO
import glob
import time

# Load both models
model_640 = YOLO("runs/train/fastsam_dami_finetune/weights/best.pt")
model_1920 = YOLO("runs/train/fastsam_dami_1920/weights/best.pt")

# Get first 5 dami images
image_files = sorted(glob.glob("dami/*.png"))[:5]

print("=" * 60)
print("Testing imgsz=640 model")
print("=" * 60)

start = time.time()
results_640 = model_640.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_640_test",
    conf=0.25,
    iou=0.5,
    imgsz=640,
    device=0,
)
time_640 = time.time() - start

print(
    f"\n640 model - Total time: {time_640:.3f}s ({time_640 / len(image_files):.3f}s per image)"
)
for i, (img_path, result) in enumerate(zip(image_files, results_640)):
    num_det = len(result.boxes) if result.boxes is not None else 0
    print(f"  {i + 1}. {img_path.split('/')[-1]}: {num_det} detections")

print("\n" + "=" * 60)
print("Testing imgsz=1920 model")
print("=" * 60)

start = time.time()
results_1920 = model_1920.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_1920_test",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
)
time_1920 = time.time() - start

print(
    f"\n1920 model - Total time: {time_1920:.3f}s ({time_1920 / len(image_files):.3f}s per image)"
)
for i, (img_path, result) in enumerate(zip(image_files, results_1920)):
    num_det = len(result.boxes) if result.boxes is not None else 0
    print(f"  {i + 1}. {img_path.split('/')[-1]}: {num_det} detections")
    if num_det > 0:
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for cls, conf in zip(classes, confs):
            print(f"     - class {int(cls)} (conf: {conf:.3f})")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(
    f"640 model:  {time_640:.3f}s total, {time_640 / len(image_files) * 1000:.1f}ms per image"
)
print(
    f"1920 model: {time_1920:.3f}s total, {time_1920 / len(image_files) * 1000:.1f}ms per image"
)
print(f"Speed ratio: {time_1920 / time_640:.2f}x")
