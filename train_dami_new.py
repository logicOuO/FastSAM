#!/usr/bin/env python3
"""
FastSAM 新数据集训练脚本

数据集: new/ (221张图片, 14549个标注)
划分: 198训练 + 23验证
配置: nc=1, imgsz=1920, batch=8, epochs=150 (降低 batch 避免 OOM)
"""

from ultralytics import YOLO

# 加载预训练模型
model = YOLO("pretrain.pt")

# 训练配置
results = model.train(
    data="ultralytics/datasets/dami_new.yaml",  # 数据集配置
    epochs=150,  # 训练轮数 (数据量增加，需要更多epochs)
    imgsz=1920,  # 输入分辨率 (保持小目标细节)
    batch=8,  # 批次大小 (降低以适应 1920 分辨率 + mask loss)
    single_cls=True,  # class-agnostic 模式
    rect=True,  # 矩形训练 (保持宽高比)
    lr0=0.001,  # 初始学习率
    patience=30,  # 早停耐心值
    device=0,  # GPU 0
    workers=8,  # 数据加载线程
    project="runs/train",  # 输出目录
    name="fastsam_dami_new",  # 实验名称
    exist_ok=True,  # 允许覆盖
    pretrained=True,  # 使用预训练权重
    optimizer="AdamW",  # 优化器
    verbose=True,  # 详细输出
    seed=42,  # 随机种子
    deterministic=False,  # 非确定性 (更快)
    val=True,  # 启用验证
    save=True,  # 保存检查点
    save_period=10,  # 每10轮保存一次
    plots=True,  # 生成训练图表
    amp=True,  # 自动混合精度
)

print("\n训练完成!")
print(f"最佳模型: runs/train/fastsam_dami_new/weights/best.pt")
print(f"最终模型: runs/train/fastsam_dami_new/weights/last.pt")
print(f"\n验证结果:")
print(f"  Box mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  Mask mAP50: {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
