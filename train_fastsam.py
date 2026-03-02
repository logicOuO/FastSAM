#!/usr/bin/env python3
"""FastSAM Training Script"""

from ultralytics import YOLO

# Load pretrained FastSAM model
model = YOLO("weights/FastSAM.pt")

# Train on coco128-seg dataset
# Using small epochs for demo, increase for real training
model.train(
    data="coco128-seg.yaml",  # dataset yaml config
    epochs=10,  # number of epochs
    imgsz=640,  # image size
    batch=8,  # batch size (adjust based on GPU memory)
    device=0,  # GPU device (use 'cpu' if no GPU)
    workers=4,  # dataloader workers
    project="runs/train",
    name="fastsam_coco128",
    exist_ok=True,
)

print("Training completed!")
