"""Check the real Dataset and transform using temporary synthetic files only."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.dataset import BDD100KDataset
from src.evaluate import _prediction_rows
from src.model import SquarePadTransform


class DatasetGeometryTests(unittest.TestCase):
    def setUp(self):
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
        output_dir.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_dataset_", dir=output_dir)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.color = (17, 129, 241)
        self.sizes = {"train": (72, 128), "val": (129, 73)}
        names = ("person", "rider", "car", "truck", "bus", "train", "motor",
                 "bike", "traffic light", "traffic sign")
        self.mapping = [{"name": name, "coco_id": index + 1, "yolo_id": index}
                        for index, name in enumerate(names)]
        (self.root / "class_mapping.json").write_text(json.dumps(self.mapping), encoding="utf-8")
        (self.root / "annotations").mkdir()
        self.annotations = {}
        for image_id, (split, (height, width)) in enumerate(self.sizes.items(), start=1234):
            image_dir = self.root / "images" / split
            image_dir.mkdir(parents=True)
            Image.new("RGB", (width, height), self.color).save(image_dir / "synthetic.png")
            boxes = [[11.25, 7.5, 43.5, 21.25], [2.0, 3.0, 10.0, 12.0]]
            annotations = [
                {"id": index + 1, "image_id": image_id, "category_id": category,
                 "bbox": box, "area": box[2] * box[3], "iscrowd": 0}
                for index, (box, category) in enumerate(zip(boxes, (7, 8)))
            ]
            records = {
                "images": [{"id": image_id, "file_name": "synthetic.png",
                            "width": width, "height": height}],
                "annotations": annotations,
                "categories": [{"id": entry["coco_id"], "name": entry["name"]}
                               for entry in self.mapping],
            }
            (self.root / "annotations" / f"{split}.json").write_text(
                json.dumps(records), encoding="utf-8"
            )
            self.annotations[split] = annotations
        self.files_before = self.file_snapshot()

    def file_snapshot(self):
        return {path.relative_to(self.root):
                (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file()}

    def dataset(self, split):
        with redirect_stdout(io.StringIO()):
            return BDD100KDataset(self.root, split)

    def test_dataset_returns_original_rgb_pixels_boxes_and_ids(self):
        for split, (height, width) in self.sizes.items():
            with self.subTest(split=split):
                dataset = self.dataset(split)
                image, target = dataset[0]
                annotations = self.annotations[split]
                self.assertEqual(tuple(image.shape), (3, height, width))
                self.assertEqual(image.dtype, torch.float32)
                expected = torch.tensor(self.color, dtype=torch.float32) / 255.0
                torch.testing.assert_close(image, expected[:, None, None].expand_as(image), rtol=0, atol=0)
                self.assertEqual(dataset.class_mapping, self.mapping)
                self.assertEqual(dataset.image_ids, [annotations[0]["image_id"]])
                self.assertEqual(set(target), {"boxes", "labels", "image_id", "area", "iscrowd"})
                self.assertEqual(target["boxes"].dtype, torch.float32)
                torch.testing.assert_close(
                    target["boxes"], torch.tensor([[11.25, 7.5, 54.75, 28.75], [2.0, 3.0, 12.0, 15.0]]),
                    rtol=0, atol=0,
                )
                for key in ("labels", "image_id", "iscrowd"):
                    self.assertEqual(target[key].dtype, torch.int64)
                self.assertEqual(target["labels"].tolist(), [7, 8])
                self.assertEqual(target["image_id"].tolist(), [annotations[0]["image_id"]])
                self.assertEqual(target["iscrowd"].tolist(), [0, 0])
                self.assertEqual(target["area"].dtype, torch.float32)
                self.assertEqual(target["area"].tolist(), [row["area"] for row in annotations])
        self.assertEqual(self.file_snapshot(), self.files_before)

    def test_square_transform_roundtrip_and_export_preserve_sources(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for split, original_size in self.sizes.items():
            dataset = self.dataset(split)
            image, target = dataset[0]
            image_before = image.clone()
            target_before = {key: value.clone() for key, value in target.items()}
            coco_before = copy.deepcopy(dataset.coco.dataset)
            for training in (True, False):
                with self.subTest(split=split, training=training):
                    transform = SquarePadTransform(640, mean, std).train(training)
                    image_list, resized = transform([image], [target])
                    self.assertEqual(tuple(image_list.tensors.shape), (1, 3, 640, 640))
                    height, width = image_list.image_sizes[0]
                    self.assertEqual(max(height, width), 640)
                    self.assertLess(min(height, width), 640)
                    scale = torch.tensor([width / original_size[1], height / original_size[0]] * 2)
                    torch.testing.assert_close(resized[0]["boxes"], target["boxes"] * scale)
                    expected_pixel = ((torch.tensor(self.color) / 255.0 - torch.tensor(mean))
                                      / torch.tensor(std))
                    torch.testing.assert_close(image_list.tensors[0, :, 0, 0], expected_pixel)
                    self.assertEqual(torch.count_nonzero(image_list.tensors[0, :, height:, :]).item(), 0)
                    self.assertEqual(torch.count_nonzero(image_list.tensors[0, :, :, width:]).item(), 0)

                    predictions = [{"boxes": resized[0]["boxes"].clone(),
                                    "labels": target["labels"].clone(), "scores": torch.tensor([0.9, 0.8])}]
                    transform.eval()
                    restored = transform.postprocess(predictions, image_list.image_sizes, [original_size])
                    rows = _prediction_rows(restored[0], int(target["image_id"].item()))
                    for row, annotation in zip(rows, self.annotations[split]):
                        self.assertEqual(row["image_id"], annotation["image_id"])
                        self.assertEqual(row["category_id"], annotation["category_id"])
                        torch.testing.assert_close(torch.tensor(row["bbox"]), torch.tensor(annotation["bbox"]),
                                                   rtol=1e-6, atol=1e-5)
                    torch.testing.assert_close(image, image_before, rtol=0, atol=0)
                    for key in target:
                        torch.testing.assert_close(target[key], target_before[key], rtol=0, atol=0)
                    self.assertEqual(dataset.coco.dataset, coco_before)
        self.assertEqual(self.file_snapshot(), self.files_before)

    def test_unsupported_split_is_rejected_before_file_access(self):
        with patch("src.dataset.COCO") as coco, patch.object(Path, "read_text") as read_text:
            with self.assertRaisesRegex(ValueError, "Only the fixed"):
                BDD100KDataset(self.root / "missing", "unsupported")
            coco.assert_not_called()
            read_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
