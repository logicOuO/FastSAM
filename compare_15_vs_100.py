#!/usr/bin/env python3
"""Compare 15ep vs 100ep models"""

from ultralytics import YOLO
import glob

model_15 = YOLO("runs/train/fastsam_dami_1920/weights/best.pt")
model_100 = YOLO("runs/train/fastsam_dami_1920_100ep/weights/best.pt")

image_files = sorted(glob.glob("dami/*.png"))[:5]

print("=" * 60)
print("15 epoch model")
print("=" * 60)
results_15 = model_15.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_15ep_final",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,
    line_width=3,
)

total_15 = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_15)
print(f"Total detections: {total_15}")

print("\n" + "=" * 60)
print("100 epoch model (best@56)")
print("=" * 60)
results_100 = model_100.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_100ep_final",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,
    line_width=3,
)

total_100 = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_100)
print(f"Total detections: {total_100}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
for i, (img_path, r15, r100) in enumerate(zip(image_files, results_15, results_100)):
    n15 = len(r15.boxes) if r15.boxes is not None else 0
    n100 = len(r100.boxes) if r100.boxes is not None else 0
    print(f"{i + 1}. {img_path.split('/')[-1]}")
    print(f"   15ep: {n15} detections")
    print(f"   100ep: {n100} detections")
    if n100 > 0:
        classes = r100.boxes.cls.cpu().numpy()
        confs = r100.boxes.conf.cpu().numpy()
        for cls, conf in zip(classes, confs):
            print(f"      - class {int(cls)} (conf: {conf:.3f})")
