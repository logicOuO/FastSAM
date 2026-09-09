#!/usr/bin/env python3
"""验证 FastSAM ONNX 模型，并保存可视化结果和合并二值掩码。"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "FastSAM-64x1920.onnx"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "fastsam" / "images" / "val"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "visualize" / "fastsam_onnx"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

INPUT_WIDTH = 1920
INPUT_HEIGHT = 64
CONF_THRESH = 0.25
IOU_THRESH = 0.7

HEAD_NUM = 3
STRIDES = [8, 16, 32]
DFL_NUM = 16


# ============ 工具函数 ============
def sigmoid(x):
    """数值稳定的 Sigmoid。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def softmax(x):
    """计算最后一维上的 Softmax。"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def decode_dfl(reg_vec):
    """
    DFL 解码
    reg_vec: [64] = [16, 16, 16, 16] (left, top, right, bottom)
    返回: [4] (left, top, right, bottom)
    """
    reg_vec = reg_vec.reshape(4, DFL_NUM)
    decoded = []
    for i in range(4):
        sf = softmax(reg_vec[i])
        val = np.sum(sf * np.arange(DFL_NUM))
        decoded.append(val)
    return decoded


def iou(box1, box2):
    """计算 IOU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def nms(detections, iou_thresh):
    """NMS"""
    if len(detections) == 0:
        return []

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    keep = []
    while len(detections) > 0:
        current = detections[0]
        keep.append(current)
        detections = detections[1:]

        # 过滤高 IOU 的框
        filtered = []
        for det in detections:
            if iou(current["box"], det["box"]) <= iou_thresh:
                filtered.append(det)
        detections = filtered

    return keep


# ============ 后处理 ============
def postprocess(outputs, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """
    后处理 ONNX 输出
    outputs: 10 个张量
        0,1: P3 (reg, cls)
        2,3: P4 (reg, cls)
        4,5: P5 (reg, cls)
        6,7,8: mask coefficients
        9: proto
    """
    if len(outputs) < 10:
        raise ValueError(f"ONNX 输出数量异常，期望至少 10 个，实际 {len(outputs)} 个")

    detections = []

    for head_idx in range(HEAD_NUM):
        reg = outputs[head_idx * 2]  # [1, 64, H, W]
        cls = outputs[head_idx * 2 + 1]  # [1, 1, H, W]
        mc = outputs[6 + head_idx]  # [1, 32, H, W]

        stride = STRIDES[head_idx]
        h, w = reg.shape[2:]

        # 遍历特征图
        for i in range(h):
            for j in range(w):
                # 置信度
                cls_val = cls[0, 0, i, j]
                conf = sigmoid(cls_val)

                if conf > conf_thresh:
                    # Anchor 中心点
                    anchor_x = j + 0.5
                    anchor_y = i + 0.5

                    # DFL 解码
                    reg_vec = reg[0, :, i, j]
                    dfl_decoded = decode_dfl(reg_vec)

                    # 转换为 xyxy
                    x1 = (anchor_x - dfl_decoded[0]) * stride
                    y1 = (anchor_y - dfl_decoded[1]) * stride
                    x2 = (anchor_x + dfl_decoded[2]) * stride
                    y2 = (anchor_y + dfl_decoded[3]) * stride

                    # 边界检查
                    x1 = max(0, min(INPUT_WIDTH, x1))
                    y1 = max(0, min(INPUT_HEIGHT, y1))
                    x2 = max(0, min(INPUT_WIDTH, x2))
                    y2 = max(0, min(INPUT_HEIGHT, y2))

                    # Mask 系数
                    mask_coef = mc[0, :, i, j]

                    detections.append(
                        {"box": [x1, y1, x2, y2], "score": conf, "mask_coef": mask_coef}
                    )

    # NMS
    detections = nms(detections, iou_thresh)

    # 提取 proto
    proto = outputs[9]  # [1, 32, 16, 480]

    return detections, proto


def generate_masks(detections, proto, img_shape):
    """
    生成 mask
    proto: [1, 32, 16, 480]
    """
    if len(detections) == 0:
        return []

    proto = proto[0]
    mask_num, seg_height, seg_width = proto.shape
    proto_flat = proto.reshape(mask_num, -1)

    masks = []
    for det in detections:
        # mask = sigmoid(mask_coef @ proto)
        mask_coef = det["mask_coef"]  # [32]
        if mask_coef.shape[0] != mask_num:
            raise ValueError(f"Mask 系数维度 {mask_coef.shape[0]} 与 Proto 维度 {mask_num} 不一致")
        mask_raw = np.matmul(mask_coef, proto_flat)
        mask_raw = mask_raw.reshape(seg_height, seg_width)
        mask_prob = sigmoid(mask_raw)

        # Resize 到原图尺寸
        mask_resized = cv2.resize(
            mask_prob, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # 裁剪到 bbox 区域（重要！）
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        mask_bbox = np.zeros_like(mask_resized)
        mask_bbox[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]

        # 二值化
        mask_binary = (mask_bbox > 0.5).astype(np.uint8)

        masks.append(mask_binary)

    return masks


def visualize(img, detections, masks):
    """可视化"""
    vis_img = img.copy()
    binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    palette = ((255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 80))
    for index, (det, mask) in enumerate(zip(detections, masks)):
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        score = det["score"]

        # 绘制 bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 绘制置信度
        label = f"{score:.2f}"
        cv2.putText(
            vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )

        # 叠加 mask（半透明）
        color = palette[index % len(palette)]
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0] = color
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_colored, 0.4, 0)

        # 合并到二值化 mask
        binary_mask = cv2.bitwise_or(binary_mask, mask * 255)

    return vis_img, binary_mask


def parse_args():
    """解析 ONNX 验证参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="ONNX 模型")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入图片目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="NMS IOU 阈值")
    parser.add_argument("--limit", type=int, default=0, help="最多处理图片数，0 表示全部")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    mask_dir = output_dir / "binary_masks"
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("置信度阈值和 IOU 阈值必须位于 [0, 1]")
    if args.limit < 0:
        raise ValueError("处理数量上限不能小于 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 加载 ONNX 模型
    print(f"加载 ONNX 模型: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    # 获取测试图片
    img_files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        img_files = img_files[: args.limit]
    if not img_files:
        raise FileNotFoundError(f"没有找到测试图片: {input_dir}")
    print(f"测试图片数量: {len(img_files)}")
    print(f"输出目录: {output_dir}\n")

    total_detections = 0

    for idx, img_path in enumerate(img_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(img_files)}] 无法读取，已跳过: {img_path}")
            continue
        orig_height, orig_width = img.shape[:2]

        # 预处理（直接 resize，不用 letterbox）
        img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_chw, axis=0)

        # ONNX 推理
        outputs = session.run(None, {input_name: img_batch})

        # 后处理
        detections, proto = postprocess(outputs, args.conf, args.iou)

        scale_x = orig_width / INPUT_WIDTH
        scale_y = orig_height / INPUT_HEIGHT
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            detection["box"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

        if len(detections) > 0:
            # 生成 mask
            masks = generate_masks(detections, proto, img.shape)

            # 可视化
            vis_img, binary_mask = visualize(img, detections, masks)

            # 保存
            visual_path = output_dir / img_path.name
            mask_path = mask_dir / img_path.name
            if not cv2.imwrite(str(visual_path), vis_img):
                raise OSError(f"可视化图片保存失败: {visual_path}")
            if not cv2.imwrite(str(mask_path), binary_mask):
                raise OSError(f"二值掩码保存失败: {mask_path}")

            total_detections += len(detections)
            print(f"[{idx}/{len(img_files)}] {img_path.name}: {len(detections)} 个目标")
        else:
            print(f"[{idx}/{len(img_files)}] {img_path.name}: 0 个目标")

    print("\n验证完成！")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张: {total_detections / len(img_files):.1f} 个")
    print(f"  可视化结果: {output_dir}")
    print(f"  二值掩码: {mask_dir}")


if __name__ == "__main__":
    main()
