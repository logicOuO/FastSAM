#!/usr/bin/env python3
"""Fine-tune FastSAM on dami dataset with 100 epochs"""

from ultralytics import YOLO

# Load COCO-trained model
model = YOLO("pretrain.pt")

# Fine-tune on dami with 100 epochs
model.train(
    data="ultralytics/datasets/dami.yaml",
    epochs=100,
    imgsz=1920,
    batch=8,
    device=0,
    workers=8,
    project="runs/train",
    name="fastsam_dami_1920_100ep",
    exist_ok=True,
    rect=True,
    lr0=0.001,
    patience=20,  # Early stopping if no improvement for 20 epochs
)

print("Fine-tuning with 100 epochs completed!")
