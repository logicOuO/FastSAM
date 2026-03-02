#!/usr/bin/env python3
"""Visualize nc=1 model predictions"""

from ultralytics import YOLO
import glob

model = YOLO("runs/train/fastsam_dami_nc1_b24/weights/best.pt")
image_files = sorted(glob.glob("dami/*.png"))[:5]

print("=" * 60)
print("nc=1 Model Inference (batch=24 trained)")
print("=" * 60)

results = model.predict(
    source=image_files,
    save=True,
    project="runs/visualize",
    name="dami_nc1_final",
    conf=0.25,
    iou=0.5,
    imgsz=1920,
    device=0,
    show_labels=False,
    line_width=3,
)

total = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
print(f"\nTotal detections: {total}")

print("\nDetailed results:")
for i, (img_path, result) in enumerate(zip(image_files, results)):
    img_name = img_path.split("/")[-1]
    num_det = len(result.boxes) if result.boxes is not None else 0
    print(f"\n{i + 1}. {img_name}")
    print(f"   Detections: {num_det}")

    if num_det > 0:
        confs = result.boxes.conf.cpu().numpy()
        print(f"   Confidence range: {confs.min():.3f} - {confs.max():.3f}")
        for j, conf in enumerate(confs, 1):
            print(f"      {j}. object (conf: {conf:.3f})")

print("\n" + "=" * 60)
print(f"Results saved to: runs/visualize/dami_nc1_final/")
print("=" * 60)
