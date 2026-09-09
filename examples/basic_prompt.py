#!/usr/bin/env python3
"""FastSAM 点提示推理的最小示例。"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastsam import FastSAM, FastSAMPrompt

MODEL_PATH = PROJECT_ROOT / "weights" / "FastSAM.pt"
IMAGE_PATH = PROJECT_ROOT / "examples" / "images" / "dogs.jpg"
OUTPUT_PATH = PROJECT_ROOT / "runs" / "examples" / "basic_prompt.jpg"


def main():
    """使用一个前景点执行分割。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastSAM(str(MODEL_PATH))
    results = model(
        str(IMAGE_PATH),
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt = FastSAMPrompt(str(IMAGE_PATH), results, device=device)
    annotations = prompt.point_prompt(points=[[620, 360]], pointlabel=[1])
    prompt.plot(
        annotations=annotations,
        output_path=str(OUTPUT_PATH),
        mask_random_color=True,
        better_quality=True,
        retina=True,
        withContours=True,
    )
    print(f"示例结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
