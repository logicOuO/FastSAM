#!/usr/bin/env python3
"""训练 FastSAM 单类别分割模型。"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "weights" / "FastSAM-COCO.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"
DATASET_PATH = PROJECT_ROOT / "data" / "fastsam"
RUNS_PATH = PROJECT_ROOT / "runs" / "train"
EXPERIMENT_NAME = "fastsam"


def parse_args():
    """解析可调整的训练参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=1920, help="训练输入尺寸")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--device", default="0", help="训练设备，例如 0、0,1 或 cpu")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument(
        "--name",
        default=EXPERIMENT_NAME,
        help="实验名称；同名目录已存在时会自动追加序号",
    )
    return parser.parse_args()


def validate_inputs():
    """在启动耗时训练前检查必要输入。"""
    required_paths = (
        MODEL_PATH,
        CONFIG_PATH,
        DATASET_PATH / "images" / "train",
        DATASET_PATH / "images" / "val",
        DATASET_PATH / "labels" / "train",
        DATASET_PATH / "labels" / "val",
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"训练输入不完整：\n{missing_text}")


def main():
    """执行训练并输出核心指标。"""
    args = parse_args()
    validate_inputs()

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(CONFIG_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        single_cls=True,
        rect=True,
        lr0=0.001,
        patience=30,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_PATH),
        name=args.name,
        # 每次训练创建独立目录，避免权重、图表和指标记录相互覆盖或混合。
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=False,
        val=True,
        save=True,
        # 只保留 best.pt 和 last.pt，避免周期权重占用数 GB 空间。
        save_period=-1,
        plots=True,
        amp=True,
    )

    run_path = Path(model.trainer.save_dir).resolve()
    weights_path = run_path / "weights"
    metrics = model.metrics.results_dict if model.metrics is not None else {}

    print("\n训练完成！")
    print(f"最佳模型: {weights_path / 'best.pt'}")
    print(f"最终模型: {weights_path / 'last.pt'}")
    print("\n验证结果:")
    print(f"  Box mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Box mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Mask mAP50: {metrics.get('metrics/mAP50(M)', 'N/A')}")
    print(f"  Mask mAP50-95: {metrics.get('metrics/mAP50-95(M)', 'N/A')}")
    print("\n自动生成的可视化:")
    print(f"  训练指标曲线: {run_path / 'results.png'}")
    print(f"  混淆矩阵: {run_path / 'confusion_matrix.png'}")
    print(f"  验证预测示例: {run_path / 'val_batch0_pred.jpg'}")
    print("\n如需验证集逐张可视化及二值掩码，请运行:")
    print(f"  python scripts/inference/visualize.py --model {weights_path / 'best.pt'} --device 0")
    print("\n如需导出本次训练的 ONNX，请运行:")
    print(f"  python scripts/export/export_onnx.py --model {weights_path / 'best.pt'}")


if __name__ == "__main__":
    main()
