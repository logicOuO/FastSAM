#!/usr/bin/env python3
"""Fine-tune FastSAM on dami dataset"""

from ultralytics import YOLO

# Load COCO-trained model
model = YOLO("pretrain.pt")

# Fine-tune on dami
model.train(
    data="ultralytics/datasets/dami.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    project="runs/train",
    name="fastsam_dami_finetune",
    exist_ok=True,
    rect=True,  # rectangular training for extreme aspect ratios
    lr0=0.001,  # lower learning rate for fine-tuning
)

print("Fine-tuning completed!")
