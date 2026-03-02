#!/usr/bin/env python3
"""Fine-tune FastSAM on dami dataset with imgsz=1920"""

from ultralytics import YOLO

# Load COCO-trained model
model = YOLO("pretrain.pt")

# Fine-tune on dami with imgsz=1920
model.train(
    data="ultralytics/datasets/dami.yaml",
    epochs=15,
    imgsz=1920,  # Keep original resolution
    batch=8,  # Reduce batch size for larger images
    device=0,
    workers=8,
    project="runs/train",
    name="fastsam_dami_1920",
    exist_ok=True,
    rect=True,  # Maintain aspect ratio (1920x64)
    lr0=0.001,  # Lower learning rate for fine-tuning
)

print("Fine-tuning with imgsz=1920 completed!")
