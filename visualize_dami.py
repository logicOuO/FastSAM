#!/usr/bin/env python3
"""Visualize dami fine-tuned model predictions"""

from ultralytics import YOLO
import glob
import os

# Load fine-tuned model
model = YOLO("runs/train/fastsam_dami_finetune/weights/best.pt")

# Get first 5 dami images
image_files = sorted(glob.glob("dami/*.png"))[:5]

print(f"Running inference on {len(image_files)} images...")

# Run inference
results = model.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_finetune_test",
    conf=0.01,  # very low threshold to see any predictions
    iou=0.5,
    imgsz=640,
    device=0,
)

print(f"\nResults saved to: runs/visualize/dami_finetune_test/")

# Print detection summary
for i, (img_path, result) in enumerate(zip(image_files, results)):
    img_name = os.path.basename(img_path)
    num_detections = len(result.boxes) if result.boxes is not None else 0
    print(f"{i + 1}. {img_name}: {num_detections} detections")

    if num_detections > 0:
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for cls, conf in zip(classes, confs):
            print(f"   - class {int(cls)} (conf: {conf:.3f})")
