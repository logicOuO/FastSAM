#!/usr/bin/env python3
"""FastSAM ONNX Export Script for RKNN deployment"""

import torch
import sys

sys.path.insert(0, "/home/keye/FastSAM")

from fastsam import FastSAM

# 输入尺寸 (和你现有 rknn 一致)
INPUT_H, INPUT_W = 352, 640

# 加载模型
print("Loading FastSAM model...")
model = FastSAM("./weights/FastSAM.pt")

# 设置为导出模式
model.model.eval()
model.model.model[-1].export = True  # Segment head export flag

# 输入
dummy_input = torch.randn(1, 3, INPUT_H, INPUT_W)

# 输出名称: 6个检测输出 + 3个mask系数 + 1个proto
output_names = [
    "reg1",
    "cls1",
    "reg2",
    "cls2",
    "reg3",
    "cls3",
    "mc1",
    "mc2",
    "mc3",
    "proto",
]

# 导出 ONNX
onnx_path = f"./FastSAM-X_{INPUT_H}x{INPUT_W}.onnx"
print(f"Exporting to {onnx_path}...")

torch.onnx.export(
    model.model,
    dummy_input,
    onnx_path,
    verbose=False,
    input_names=["images"],
    output_names=output_names,
    opset_version=12,
    do_constant_folding=True,
)

print(f"✓ ONNX exported to: {onnx_path}")
print(f"  Input shape: 1x3x{INPUT_H}x{INPUT_W}")
print(f"  Outputs: {output_names}")

# 验证 ONNX
try:
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model check passed")

    # 打印输出信息
    print("\nOutput details:")
    for output in onnx_model.graph.output:
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")
except ImportError:
    print("⚠ onnx package not installed, skipping verification")
except Exception as e:
    print(f"⚠ ONNX check failed: {e}")
