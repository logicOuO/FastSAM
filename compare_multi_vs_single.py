#!/usr/bin/env python3
"""Compare multi-class vs single-class models"""

from ultralytics import YOLO
import glob

model_multi = YOLO("runs/train/fastsam_dami_1920_100ep/weights/best.pt")
model_single = YOLO("runs/train/fastsam_dami_single_cls/weights/best.pt")

image_files = sorted(glob.glob("dami/*.png"))[:5]

print("=" * 60)
print("Multi-class model (3 classes)")
print("=" * 60)
results_multi = model_multi.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_multi_class",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,
    line_width=3,
)

total_multi = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_multi)
print(f"Total detections: {total_multi}")

print("\n" + "=" * 60)
print("Single-class model (class-agnostic)")
print("=" * 60)
results_single = model_single.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_single_class",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,
    line_width=3,
)

total_single = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_single)
print(f"Total detections: {total_single}")

print("\n" + "=" * 60)
print("DETAILED COMPARISON")
print("=" * 60)
for i, (img_path, rm, rs) in enumerate(zip(image_files, results_multi, results_single)):
    nm = len(rm.boxes) if rm.boxes is not None else 0
    ns = len(rs.boxes) if rs.boxes is not None else 0
    print(f"\n{i + 1}. {img_path.split('/')[-1]}")
    print(f"   Multi-class:  {nm} detections")
    if nm > 0:
        classes = rm.boxes.cls.cpu().numpy()
        confs = rm.boxes.conf.cpu().numpy()
        for cls, conf in zip(classes, confs):
            print(f"      - class {int(cls)} (conf: {conf:.3f})")

    print(f"   Single-class: {ns} detections")
    if ns > 0:
        confs = rs.boxes.conf.cpu().numpy()
        for conf in confs:
            print(f"      - object (conf: {conf:.3f})")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Multi-class:  {total_multi} total detections")
print(f"Single-class: {total_single} total detections")
print(
    f"Difference:   {total_single - total_multi:+d} ({(total_single / total_multi - 1) * 100:+.1f}%)"
)
