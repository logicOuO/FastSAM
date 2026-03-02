#!/usr/bin/env python3
"""Export single-class dami model to ONNX"""

import torch
from ultralytics import YOLO

# Load single-class trained model
model = YOLO("runs/train/fastsam_dami_single_cls/weights/best.pt")

# Set export mode
model.model.eval()
model.model.model[-1].export = True  # Segment head export flag

# Input shape: 1x3x64x1920 (matching training aspect ratio)
INPUT_H, INPUT_W = 64, 1920
dummy_input = torch.randn(1, 3, INPUT_H, INPUT_W)

# Output names
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

# Export ONNX
onnx_path = f"./FastSAM-dami-single-cls_{INPUT_H}x{INPUT_W}.onnx"
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
print(f"  Model: Single-class (class-agnostic)")
print(f"  Outputs: {output_names}")

# Verify ONNX
try:
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model check passed")

    print("\nOutput details:")
    for output in onnx_model.graph.output:
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")
except ImportError:
    print("⚠ onnx package not installed, skipping verification")
except Exception as e:
    print(f"⚠ ONNX check failed: {e}")
