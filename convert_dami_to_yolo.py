#!/usr/bin/env python3
"""
Convert dami JSON annotations to YOLO segmentation format.

Strategy:
- Prefer enhancePolygon/polygon over rectangle
- Label mapping: "1"->0, "2"->1, "3"->2
- Normalize coordinates to [0,1]
- Output format: class x1 y1 x2 y2 ... (normalized polygon points)
"""

import json
import glob
import os
from pathlib import Path

# Configuration
INPUT_DIR = "dami"
OUTPUT_DIR = "datasets/dami_yolo"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Label mapping
LABEL_MAP = {"1": 0, "2": 1, "3": 2}


def convert_annotation(json_path):
    """Convert single JSON file to YOLO format."""
    with open(json_path, "r") as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]
    img_name = data["imagePath"]

    # Prepare output lines
    yolo_lines = []

    for shape in data["shapes"]:
        label = str(shape.get("label", ""))
        if label not in LABEL_MAP:
            print(f"Warning: Unknown label '{label}' in {json_path}, skipping")
            continue

        class_id = LABEL_MAP[label]
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if not points:
            print(f"Warning: Empty points in {json_path}, skipping shape")
            continue

        # Normalize points to [0, 1]
        normalized_points = []
        for x, y in points:
            norm_x = x / img_width
            norm_y = y / img_height
            # Clamp to [0, 1]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            normalized_points.extend([norm_x, norm_y])

        # Build YOLO line: class x1 y1 x2 y2 ...
        yolo_line = f"{class_id} " + " ".join(
            f"{coord:.6f}" for coord in normalized_points
        )
        yolo_lines.append(yolo_line)

    return img_name, yolo_lines


def main():
    # Create output directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    print(f"Found {len(json_files)} JSON files")
    print(f"Output: {OUTPUT_DIR}")
    print()

    converted_count = 0
    skipped_count = 0

    for json_path in json_files:
        try:
            img_name, yolo_lines = convert_annotation(json_path)

            if not yolo_lines:
                print(f"Skipped {json_path}: no valid annotations")
                skipped_count += 1
                continue

            # Write label file
            base_name = Path(img_name).stem
            label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            # Copy image (symlink to save space)
            src_img = os.path.join(INPUT_DIR, img_name)
            dst_img = os.path.join(IMAGES_DIR, img_name)

            if os.path.exists(src_img):
                if not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(src_img), dst_img)
            else:
                print(f"Warning: Image not found: {src_img}")

            converted_count += 1

        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            skipped_count += 1

    print()
    print(f"Conversion complete:")
    print(f"  Converted: {converted_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Labels: {LABELS_DIR}")


if __name__ == "__main__":
    main()
