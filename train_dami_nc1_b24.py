#!/usr/bin/env python3
"""Fine-tune FastSAM with nc=1 and higher batch size"""

from ultralytics import YOLO

# Load COCO-trained model
model = YOLO("pretrain.pt")

# Fine-tune with nc=1 (truly single-class) and batch=24
model.train(
    data="ultralytics/datasets/dami.yaml",  # Now nc=1
    epochs=100,
    imgsz=1920,
    batch=24,  # Increased from 16 (6.71GB -> ~10GB estimated)
    device=0,
    workers=8,
    project="runs/train",
    name="fastsam_dami_nc1_b24",
    exist_ok=True,
    rect=True,
    lr0=0.001,
    patience=20,
    single_cls=True,  # Redundant with nc=1 but explicit
)

print("Training with nc=1, batch=24 completed!")
