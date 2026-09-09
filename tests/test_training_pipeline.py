"""FastSAM 数据转换与 ONNX 后处理单元测试。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import train_fastsam
from fastsam import FastSAMPrompt
from scripts.data.convert_dataset import convert_annotation
from scripts.data.split_dataset import partition_images
from scripts.inference.validate_onnx import decode_dfl, iou, nms, softmax


class ConvertAnnotationTest(unittest.TestCase):
    """验证 LabelMe 到 YOLO 分割格式的转换。"""

    def test_convert_annotation_filters_and_normalizes(self):
        annotation = {
            "imageWidth": 100,
            "imageHeight": 50,
            "imagePath": "nested/sample.png",
            "shapes": [
                {"label": "1", "points": [[0, 0], [100, 0], [120, 50], [0, 50]]},
                {"label": "2", "points": [[0, 0], [10, 0], [10, 10]]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "sample.json"
            json_path.write_text(json.dumps(annotation), encoding="utf-8")
            image_name, lines = convert_annotation(json_path)

        self.assertEqual(image_name, "sample.png")
        self.assertEqual(
            lines,
            ["0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.000000 1.000000"],
        )

    def test_convert_annotation_rejects_invalid_size(self):
        annotation = {
            "imageWidth": 0,
            "imageHeight": 50,
            "imagePath": "sample.png",
            "shapes": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "sample.json"
            json_path.write_text(json.dumps(annotation), encoding="utf-8")
            with self.assertRaises(ValueError):
                convert_annotation(json_path)


class OnnxPostprocessTest(unittest.TestCase):
    """验证 DFL、IOU 与 NMS 的核心行为。"""

    def test_softmax_and_decode_dfl(self):
        probabilities = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

        decoded = decode_dfl(np.zeros(64, dtype=np.float32))
        np.testing.assert_allclose(decoded, [7.5, 7.5, 7.5, 7.5])

    def test_iou_and_nms(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)
        detections = [
            {"box": [0, 0, 10, 10], "score": 0.9},
            {"box": [1, 1, 9, 9], "score": 0.8},
            {"box": [20, 20, 30, 30], "score": 0.7},
        ]
        kept = nms(detections, 0.5)
        self.assertEqual([item["score"] for item in kept], [0.9, 0.7])


class PromptRenderTest(unittest.TestCase):
    """验证 Prompt 可视化兼容当前 Matplotlib。"""

    def test_plot_to_result_returns_rgb_image(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        mask = np.ones((10, 20), dtype=np.uint8)
        prompt = FastSAMPrompt(image=image, results=[], device="cpu")

        result = prompt.plot_to_result(
            annotations=[mask],
            better_quality=False,
            retina=True,
            withContours=False,
        )

        self.assertEqual(result.shape, (10, 20, 3))


class SplitDatasetTest(unittest.TestCase):
    """验证数据划分的数量和可复现性。"""

    def test_partition_images_is_reproducible(self):
        images = list(range(10))
        first = partition_images(images, 0.8, 42)
        second = partition_images(images, 0.8, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)
        self.assertEqual(sorted(first[0] + first[1]), images)


class TrainingOutputTest(unittest.TestCase):
    """验证每次训练使用独立目录并输出真实权重路径。"""

    def test_main_disables_existing_directory_reuse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir) / "fastsam2"
            model = SimpleNamespace(
                trainer=SimpleNamespace(save_dir=run_path),
                metrics=None,
            )
            train_arguments = {}

            def record_train(**kwargs):
                train_arguments.update(kwargs)

            model.train = record_train
            arguments = ["train_fastsam.py", "--name", "fastsam"]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(train_fastsam, "validate_inputs"),
                patch.object(train_fastsam, "YOLO", return_value=model),
                patch("builtins.print") as mock_print,
            ):
                train_fastsam.main()

        self.assertFalse(train_arguments["exist_ok"])
        self.assertEqual(train_arguments["name"], "fastsam")
        output = "\n".join(" ".join(map(str, call.args)) for call in mock_print.call_args_list)
        self.assertIn(str(run_path / "weights" / "best.pt"), output)
        self.assertIn("scripts/export/export_onnx.py --model", output)


if __name__ == "__main__":
    unittest.main()
