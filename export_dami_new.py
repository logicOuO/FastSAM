#!/usr/bin/env python3
"""
导出训练好的模型为 ONNX 格式

输入形状: 64x1920 (height x width)
输出: FastSAM-dami-new_64x1920.onnx
"""

from ultralytics import YOLO
import torch

# 加载训练好的最佳模型
model_path = "runs/train/fastsam_dami_new/weights/best.pt"
output_path = "FastSAM-dami-new_64x1920.onnx"

print(f"加载模型: {model_path}")
model = YOLO(model_path)

print(f"导出 ONNX 模型...")
print(f"  输入形状: 1x3x64x1920 (batch, channels, height, width)")
print(f"  输出路径: {output_path}")

# 导出为 ONNX
success = model.export(
    format="onnx",
    imgsz=(64, 1920),  # (height, width)
    dynamic=False,  # 固定输入尺寸
    simplify=True,  # 简化 ONNX 图
    opset=12,  # ONNX opset 版本
)

if success:
    print(f"\n导出成功: {output_path}")

    # 显示模型信息
    import onnx

    onnx_model = onnx.load(output_path)

    print(f"\nONNX 模型信息:")
    print(f"  输入:")
    for input_tensor in onnx_model.graph.input:
        print(
            f"    {input_tensor.name}: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}"
        )

    print(f"  输出:")
    for i, output_tensor in enumerate(onnx_model.graph.output):
        print(
            f"    Output {i} ({output_tensor.name}): {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}"
        )

    # 文件大小
    import os

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  文件大小: {file_size:.1f} MB")
else:
    print("导出失败!")
