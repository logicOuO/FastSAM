#!/usr/bin/env python3
"""Fine-tune FastSAM on dami with single-class (class-agnostic) mode"""

from ultralytics import YOLO

# Load COCO-trained model
model = YOLO("pretrain.pt")

# Fine-tune with single_cls=True (class-agnostic segmentation)
model.train(
    data="ultralytics/datasets/dami.yaml",
    epochs=100,
    imgsz=1920,
    batch=16,  # Increased from 8 (peak was 5.27GB, have 15GB available)
    device=0,
    workers=8,
    project="runs/train",
    name="fastsam_dami_single_cls",
    exist_ok=True,
    rect=True,
    lr0=0.001,
    patience=20,
    single_cls=True,  # Class-agnostic: merge all classes into one
)

print("Single-class training completed!")
