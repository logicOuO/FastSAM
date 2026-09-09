#!/usr/bin/env python3
"""将 FastSAM 最佳权重导出为固定尺寸 ONNX 模型。"""

import argparse
import shutil
from pathlib import Path

import onnx

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "train" / "fastsam" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / "FastSAM-64x1920.onnx"


def parse_args():
    """解析导出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="输入权重")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 ONNX 文件")
    parser.add_argument("--height", type=int, default=64, help="模型输入高度")
    parser.add_argument("--width", type=int, default=1920, help="模型输入宽度")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本")
    return parser.parse_args()


def main():
    """执行导出并打印模型输入输出信息。"""
    args = parse_args()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("输入尺寸必须为正整数")

    print(f"加载模型: {model_path}")
    print(f"导出尺寸: 1x3x{args.height}x{args.width}")
    model = YOLO(str(model_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=(args.height, args.width),
            dynamic=False,
            simplify=True,
            opset=args.opset,
        )
    ).resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"导出未生成有效文件: {exported_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != output_path:
        output_path.unlink(missing_ok=True)
        shutil.move(str(exported_path), output_path)

    onnx_model = onnx.load(str(output_path))
    print(f"\n导出成功: {output_path}")
    print("输入:")
    for tensor in onnx_model.graph.input:
        shape = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
        print(f"  {tensor.name}: {shape}")
    print("输出:")
    for index, tensor in enumerate(onnx_model.graph.output):
        shape = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
        print(f"  Output {index} ({tensor.name}): {shape}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
