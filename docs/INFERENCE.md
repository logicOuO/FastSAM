# FastSAM Prompt 推理

本页记录官方 FastSAM 权重的 Everything、点、框和文本提示推理。所有命令均在
仓库根目录执行，结果默认写入 `runs/inference/prompt/`。

## Everything 模式

```bash
python scripts/inference/prompt.py
```

可以通过 `--imgsz` 修改输入尺寸：

```bash
python scripts/inference/prompt.py --imgsz 720
```

## 点提示

```bash
python scripts/inference/prompt.py \
  --point_prompt "[[520, 360], [620, 300], [520, 300], [620, 360]]" \
  --point_label "[1, 0, 1, 0]"
```

## 框提示

`--box_prompt` 使用 `[x, y, w, h]` 格式：

```bash
python scripts/inference/prompt.py \
  --box_prompt "[[570, 200, 230, 400]]"
```

## 文本提示

文本提示需要安装 CLIP：

```bash
pip install git+https://github.com/openai/CLIP.git
python scripts/inference/prompt.py \
  --img_path examples/images/cat.jpg \
  --text_prompt "cat" \
  --better_quality \
  --with_contours
```

## 常用参数

- `--model_path`：模型权重，默认 `weights/FastSAM.pt`
- `--img_path`：输入图片，默认 `examples/images/dogs.jpg`
- `--output`：输出目录，默认 `runs/inference/prompt`
- `--device`：推理设备，例如 `0`、`cuda` 或 `cpu`
- `--conf`：置信度阈值
- `--iou`：NMS IOU 阈值
- `--retina` / `--no-retina`：启用或关闭高分辨率掩码
- `--random_color` / `--no-random_color`：启用或关闭随机颜色
