#!/usr/bin/env python3
"""运行 FastSAM Prompt 推理并保存可视化结果。"""

import argparse
import ast
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastsam import FastSAM, FastSAMPrompt
from fastsam.demo_utils import convert_box_xywh_to_xyxy

DEFAULT_MODEL = PROJECT_ROOT / "weights" / "FastSAM.pt"
DEFAULT_IMAGE = PROJECT_ROOT / "examples" / "images" / "dogs.jpg"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "inference" / "prompt"


def parse_args():
    """解析 Prompt 推理参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL, help="模型权重")
    parser.add_argument("--img_path", type=Path, default=DEFAULT_IMAGE, help="输入图片")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--imgsz", type=int, default=1024, help="推理输入尺寸")
    parser.add_argument("--iou", type=float, default=0.9, help="NMS IOU 阈值")
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--text_prompt", default=None, help='文本提示，例如 "a dog"')
    parser.add_argument(
        "--point_prompt",
        default="[[0, 0]]",
        help="点提示，例如 [[520, 360], [620, 300]]",
    )
    parser.add_argument("--point_label", default="[0]", help="点标签，1 为前景、0 为背景")
    parser.add_argument(
        "--box_prompt",
        default="[[0, 0, 0, 0]]",
        help="矩形提示，格式为 [[x, y, w, h]]",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 0、cuda 或 cpu",
    )
    parser.add_argument(
        "--retina",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否绘制高分辨率掩码",
    )
    parser.add_argument(
        "--better_quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用形态学操作优化边缘",
    )
    parser.add_argument(
        "--with_contours",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否绘制掩码轮廓",
    )
    parser.add_argument(
        "--random_color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用随机掩码颜色",
    )
    return parser.parse_args()


def main():
    """执行 FastSAM Prompt 推理。"""
    args = parse_args()
    model_path = args.model_path.resolve()
    image_path = args.img_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")

    point_prompt = ast.literal_eval(args.point_prompt)
    point_label = ast.literal_eval(args.point_label)
    box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))

    image = Image.open(image_path).convert("RGB")
    model = FastSAM(str(model_path))
    results = model(
        image,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )

    bboxes = None
    points = None
    selected_point_labels = None
    prompt = FastSAMPrompt(image, results, device=args.device)
    if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
        annotations = prompt.box_prompt(bboxes=box_prompt)
        bboxes = box_prompt
    elif args.text_prompt is not None:
        annotations = prompt.text_prompt(text=args.text_prompt)
    elif point_prompt[0] != [0, 0]:
        annotations = prompt.point_prompt(points=point_prompt, pointlabel=point_label)
        points = point_prompt
        selected_point_labels = point_label
    else:
        annotations = prompt.everything_prompt()

    output_path = args.output.resolve() / image_path.name
    prompt.plot(
        annotations=annotations,
        output_path=str(output_path),
        bboxes=bboxes,
        points=points,
        point_label=selected_point_labels,
        mask_random_color=args.random_color,
        withContours=args.with_contours,
        better_quality=args.better_quality,
        retina=args.retina,
    )
    print(f"推理结果已保存: {output_path}")


if __name__ == "__main__":
    main()
