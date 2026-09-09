# FastSAM 训练与部署流程

本仓库维护一条完整的 FastSAM 单类别分割流水线，包括原始数据转换、数据集划分、
模型训练、PyTorch 可视化、ONNX 导出与 ONNX 验证。

## 目录结构

```text
FastSAM/
├── fastsam/                     # FastSAM 推理与 Prompt 接口
├── ultralytics/                 # YOLOv8-Seg 网络与训练框架
├── configs/
│   └── dataset.yaml             # 当前训练数据配置
├── data/                        # 本地数据，不提交 Git
│   ├── raw/                     # 原始图片和 LabelMe JSON
│   └── fastsam/                 # 转换后的 YOLO 分割数据
├── weights/                     # 权重和导出模型，不提交 Git
│   ├── FastSAM.pt               # 官方 SA 数据预训练权重
│   ├── FastSAM-COCO.pt          # FastSAM 经 COCO 二次训练的权重
│   └── FastSAM-64x1920.onnx     # 当前部署模型
├── scripts/
│   ├── data/                    # 数据转换和划分
│   ├── export/                  # 模型导出
│   ├── inference/               # Prompt、可视化和 ONNX 验证
│   └── monitor_training.py      # 训练日志监控
├── apps/                        # Gradio 演示
├── deploy/                      # 可选部署配置
├── examples/                    # 最小示例与示例图片
├── runs/                        # 训练和推理产物，不提交 Git
├── tests/                       # 单元测试
└── train_fastsam.py             # 主训练入口
```

所有命令均在仓库根目录执行。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 准备数据

将原始图片和对应的 LabelMe JSON 放入 `data/raw/`，然后执行：

```bash
python scripts/data/convert_dataset.py
python scripts/data/split_dataset.py
```

默认使用固定随机种子 `42`，按照 90%/10% 划分训练集和验证集。已经完成划分时，
划分脚本会直接报告当前状态，不会重复移动文件。

## 3. 训练

```bash
python train_fastsam.py 2>&1 | tee runs/logs/train_fastsam.log
```

常用参数：

```bash
python train_fastsam.py --epochs 150 --batch 8 --device 0 --workers 8
```

训练入口默认使用 `weights/FastSAM-COCO.pt` 预训练权重。该权重来源于官方
FastSAM 权重，并经过 COCO 2017 二次训练。训练默认只保存 `best.pt` 和
`last.pt`，避免周期权重长期占用大量磁盘空间。

首次训练会在 `runs/train/fastsam/` 生成结果。若该目录已经存在，后续训练会自动
使用 `fastsam2/`、`fastsam3/` 等独立目录，避免覆盖已有权重或混合指标记录。
也可以通过 `--name` 指定本次实验名称。每个实验目录会生成：

- `results.png`：训练损失和验证指标曲线
- `Box*_curve.png`、`Mask*_curve.png`：PR、F1、Precision、Recall 曲线
- `confusion_matrix.png`：混淆矩阵
- `val_batch*_pred.jpg`：验证批次预测示例

这些图片用于检查训练过程和整体指标。若需要验证集每张图片的完整分割结果及二值
掩码，请继续执行下一节的可视化脚本。

监控训练日志：

```bash
python scripts/monitor_training.py
```

## 4. 可视化 PyTorch 模型

```bash
python scripts/inference/visualize.py \
  --model runs/train/fastsam2/weights/best.pt \
  --device 0
```

请将 `--model` 替换为训练结束时打印的最佳模型路径。若不传该参数，默认使用
`runs/train/fastsam/weights/best.pt`。可视化输出位于 `runs/visualize/fastsam/`。

## 5. 导出 ONNX

```bash
python scripts/export/export_onnx.py \
  --model runs/train/fastsam2/weights/best.pt
```

请同样将 `--model` 替换为训练结束时打印的最佳模型路径。默认导出固定输入尺寸
`1x3x64x1920` 的 `weights/FastSAM-64x1920.onnx`。

## 6. 验证 ONNX

快速验证一张图片：

```bash
python scripts/inference/validate_onnx.py --limit 1
```

验证全部图片：

```bash
python scripts/inference/validate_onnx.py
```

## 7. 单元测试

```bash
python -m unittest discover -s tests -v
```
