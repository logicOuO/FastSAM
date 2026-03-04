#!/usr/bin/env python3
"""
ONNX 模型测试脚本（参考 C++ 后处理逻辑）

正确实现:
1. DFL 解码（softmax + weighted sum）
2. NMS
3. Mask 生成（mask_coef @ proto）
"""

import onnxruntime as ort
import cv2
import numpy as np
import os

# ============ 配置 ============
ONNX_MODEL = "FastSAM-dami-new_64x1920.onnx"
INPUT_DIR = "datasets/dami_new_yolo/images/val"
OUTPUT_DIR = "runs/visualize/onnx_test_v2"
MASK_DIR = os.path.join(OUTPUT_DIR, "binary_masks")

INPUT_WIDTH = 1920
INPUT_HEIGHT = 64
CONF_THRESH = 0.25
IOU_THRESH = 0.7

# 检测头配置
HEAD_NUM = 3
MAP_SIZES = [(8, 240), (4, 120), (2, 60)]
STRIDES = [8, 16, 32]
DFL_NUM = 16
MASK_NUM = 32

# Proto 尺寸
SEG_HEIGHT = 16
SEG_WIDTH = 480


# ============ 工具函数 ============
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """Softmax along last axis"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


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
def postprocess(outputs):
    """
    后处理 ONNX 输出
    outputs: 10 个张量
        0,1: P3 (reg, cls)
        2,3: P4 (reg, cls)
        4,5: P5 (reg, cls)
        6,7,8: mask coefficients
        9: proto
    """
    detections = []

    for head_idx in range(HEAD_NUM):
        reg = outputs[head_idx * 2]  # [1, 64, H, W]
        cls = outputs[head_idx * 2 + 1]  # [1, 1, H, W]
        mc = outputs[6 + head_idx]  # [1, 32, H, W]

        stride = STRIDES[head_idx]
        h, w = MAP_SIZES[head_idx]

        # 遍历特征图
        for i in range(h):
            for j in range(w):
                # 置信度
                cls_val = cls[0, 0, i, j]
                conf = sigmoid(cls_val)

                if conf > CONF_THRESH:
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
    detections = nms(detections, IOU_THRESH)

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

    proto = proto[0]  # [32, 16, 480]
    proto_flat = proto.reshape(MASK_NUM, -1)  # [32, 16*480]

    masks = []
    for det in detections:
        # mask = sigmoid(mask_coef @ proto)
        mask_coef = det["mask_coef"]  # [32]
        mask_raw = np.matmul(mask_coef, proto_flat)  # [16*480]
        mask_raw = mask_raw.reshape(SEG_HEIGHT, SEG_WIDTH)  # [16, 480]
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

    for det, mask in zip(detections, masks):
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
        color = np.random.randint(0, 255, 3).tolist()
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0] = color
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_colored, 0.4, 0)

        # 合并到二值化 mask
        binary_mask = cv2.bitwise_or(binary_mask, mask * 255)

    return vis_img, binary_mask


# ============ 主函数 ============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    # 加载 ONNX 模型
    print(f"加载 ONNX 模型: {ONNX_MODEL}")
    session = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    # 获取测试图片
    img_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])
    print(f"测试图片数量: {len(img_files)}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    total_detections = 0

    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(INPUT_DIR, img_file)
        img = cv2.imread(img_path)

        # 预处理（直接 resize，不用 letterbox）
        img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_chw, axis=0)

        # ONNX 推理
        outputs = session.run(None, {input_name: img_batch})

        # 后处理
        detections, proto = postprocess(outputs)

        if len(detections) > 0:
            # 生成 mask
            masks = generate_masks(detections, proto, img.shape)

            # 可视化
            vis_img, binary_mask = visualize(img, detections, masks)

            # 保存
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), vis_img)
            cv2.imwrite(os.path.join(MASK_DIR, img_file), binary_mask)

            total_detections += len(detections)
            print(f"[{idx}/{len(img_files)}] {img_file}: {len(detections)} objects")
        else:
            print(f"[{idx}/{len(img_files)}] {img_file}: 0 objects")

    print(f"\n完成!")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张: {total_detections / len(img_files):.1f} 个")
    print(f"  可视化结果: {OUTPUT_DIR}")
    print(f"  二值化 mask: {MASK_DIR}")


if __name__ == "__main__":
    main()
