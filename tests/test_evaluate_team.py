"""Test the shared evaluator using synthetic COCO records only."""

import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluate_team import SUMMARY_NAMES, VAL_SUBSETS, evaluate, evaluate_validation


class TeamEvaluationTests(unittest.TestCase):
    def setUp(self):
        # Category IDs deliberately differ from input list order.
        self.gt = {
            "images": [{"id": 91, "width": 100, "height": 80},
                       {"id": 17, "width": 100, "height": 80}],
            "categories": [{"id": 9, "name": "traffic light"},
                           {"id": 6, "name": "train"},
                           {"id": 1, "name": "person"}],
            "annotations": [
                {"id": 31, "image_id": 91, "category_id": 1,
                 "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 32, "image_id": 17, "category_id": 9,
                 "bbox": [20, 20, 20, 20], "area": 400, "iscrowd": 0},
            ],
        }
        self.predictions = [
            {"image_id": 91, "category_id": 1,
             "bbox": [14, 10, 20, 20], "score": 0.9},
            {"image_id": 17, "category_id": 9,
             "bbox": [60, 20, 20, 20], "score": 0.8},
            # A low-score correct detection must remain in the evaluation.
            {"image_id": 17, "category_id": 9,
             "bbox": [20, 20, 20, 20], "score": 0.001},
        ]

    def test_original_protocol_matches_reference_coco_and_category_order(self):
        reference_gt = copy.deepcopy(self.gt)
        reference_gt["info"] = {}
        with redirect_stdout(io.StringIO()):
            coco = COCO()
            coco.dataset = reference_gt
            coco.createIndex()
            detections = coco.loadRes(copy.deepcopy(self.predictions))
            reference = COCOeval(coco, detections, "bbox")
            reference.params.imgIds = [17, 91]
            reference.params.maxDets = [1, 10, 100]
            reference.evaluate()
            reference.accumulate()
            reference.summarize()

        result = evaluate(self.gt, self.predictions)
        precision = reference.eval["precision"]
        old_map = precision[:, :, :, 0, -1]
        old_ap50 = precision[0, :, :, 0, -1]
        self.assertEqual(result["mAP"], float(old_map[old_map >= 0].mean()))
        self.assertEqual(result["AP50"], float(old_ap50[old_ap50 >= 0].mean()))
        for name, expected in zip(SUMMARY_NAMES, reference.stats):
            if expected < 0:
                self.assertIsNone(result[name])
            else:
                self.assertAlmostEqual(result[name], expected)
        for index, category_id in enumerate(reference.params.catIds):
            name = next(c["name"] for c in self.gt["categories"] if c["id"] == category_id)
            category_precision = precision[:, :, index, 0, -1]
            valid = category_precision[category_precision >= 0]
            if valid.size:
                self.assertEqual(result["per_class"][name], float(valid.mean()))
            else:
                self.assertIsNone(result["per_class"][name])
        self.assertAlmostEqual(result["mAP"], 0.45)
        self.assertAlmostEqual(result["AP50"], 0.75)
        self.assertAlmostEqual(result["AP75"], 0.25)
        self.assertEqual(result["settings"]["category_ids"], [1, 6, 9])
        self.assertEqual(result["settings"]["max_detections"], [1, 10, 100])
        self.assertEqual(len(result["settings"]["recall_thresholds"]), 101)
        self.assertIn("Average Precision", result["coco_summary"])
        self.assertEqual([c["category_id"] for c in result["per_category"]], [9, 6, 1])
        self.assertEqual(result["per_category"][1]["ground_truth_objects"], 0)
        self.assertIsNone(result["per_class_AP50"]["train"])
        self.assertIsNone(result["per_class_AP75"]["train"])
        json.dumps(result, allow_nan=False)

    def test_source_gt_and_predictions_are_unchanged_without_info(self):
        gt_before, predictions_before = copy.deepcopy(self.gt), copy.deepcopy(self.predictions)
        evaluate(self.gt, self.predictions)
        self.assertEqual(self.gt, gt_before)
        self.assertEqual(self.predictions, predictions_before)
        self.assertNotIn("info", self.gt)
        self.assertNotIn("ignore", self.gt["annotations"][0])
        self.assertNotIn("area", self.predictions[0])

    def test_empty_predictions_are_zero_only_for_evaluable_categories(self):
        result = evaluate(self.gt, [])
        self.assertEqual(result["mAP"], 0.0)
        self.assertEqual(result["AP50"], 0.0)
        self.assertEqual(result["per_class"]["person"], 0.0)
        self.assertIsNone(result["per_class"]["train"])

    def test_no_gt_objects_or_images_are_not_evaluable(self):
        gt = copy.deepcopy(self.gt)
        gt["annotations"] = []
        for predictions in ([], self.predictions):
            result = evaluate(gt, predictions)
            self.assertTrue(all(result[name] is None for name in SUMMARY_NAMES))
            self.assertTrue(all(value is None for value in result["per_class"].values()))
        gt["images"] = []
        result = evaluate(gt, [])
        self.assertEqual(result["images"], 0)
        self.assertTrue(all(result[name] is None for name in SUMMARY_NAMES))
        json.dumps(result, allow_nan=False)

    def test_invalid_prediction_ids_types_values_and_sizes(self):
        prediction = self.predictions[0]
        invalid_rows = [None, {**prediction, "image_id": 999},
                        {**prediction, "image_id": True},
                        {**prediction, "category_id": 80},
                        {**prediction, "category_id": 1.0},
                        {**prediction, "score": float("nan")},
                        {**prediction, "score": float("inf")},
                        {**prediction, "score": True},
                        {**prediction, "score": 1.1},
                        {**prediction, "bbox": [10, 10, 0, 20]},
                        {**prediction, "bbox": [10, 10, 20, -1]},
                        {**prediction, "bbox": [10, 10, float("inf"), 20]},
                        {**prediction, "bbox": [10, 10, 20]},
                        {**prediction, "bbox": [True, 10, 20, 20]}]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate(self.gt, [row])
        with self.assertRaises(ValueError):
            evaluate(self.gt, {})

    def test_valid_outside_image_boxes_are_neither_clipped_nor_rejected(self):
        predictions = [{**self.predictions[0], "bbox": [-5, 10, 120, 20]}]
        before = copy.deepcopy(predictions)
        result = evaluate(self.gt, predictions)
        self.assertIsInstance(result["mAP"], float)
        self.assertEqual(predictions, before)

    def test_validation_subset_filtering_overlap_and_drop_definition(self):
        with self.synthetic_validation_root() as directory:
            root = Path(directory)
            self.write_validation_files(root)
            predictions = [{"image_id": 91, "category_id": 1,
                            "bbox": [10, 10, 20, 20], "score": 0.9}]
            result = evaluate_validation(root, predictions)
            groups = result["results"]
            self.assertEqual(set(groups), {"val_all", *[f"val_{s}" for s in VAL_SUBSETS]})
            self.assertEqual(groups["val_clear_daytime"]["images"], 1)
            self.assertEqual(groups["val_rainy"]["images"], 2)
            self.assertAlmostEqual(groups["val_clear_daytime"]["mAP"], 1.0)
            self.assertEqual(groups["val_night"]["mAP"], 0.0)
            self.assertAlmostEqual(groups["val_night"]["mAP_drop_vs_clear_daytime"], 1.0)
            self.assertIsNone(groups["val_dawn_dusk"]["mAP_drop_vs_clear_daytime"])
            self.assertIsNone(groups["val_all"]["per_class"]["train"])
            self.assertIn("Not evaluable", result["protocol"]["undefined_metric"])
            self.assertFalse((root / "annotations" / "test.json").exists())
            without_subsets = evaluate_validation(root, predictions, include_subsets=False)
            self.assertEqual(set(without_subsets["results"]), {"val_all"})

    def test_validation_cli_writes_utf8_json_and_csv(self):
        with self.synthetic_validation_root() as directory:
            root = Path(directory)
            self.gt["categories"][2]["name"] = "person \u00e9"
            self.write_validation_files(root)
            prediction_path = root / "predictions.json"
            prediction_path.write_text("[]\n", encoding="utf-8")
            output_path = root / "results" / "evaluation.json"
            completed = subprocess.run(
                [sys.executable, "evaluate_team.py", "--root", str(root),
                 "--predictions", str(prediction_path), "--split", "val",
                 "--output", str(output_path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["results"]["val_all"]["mAP"], 0.0)
            self.assertIsNone(result["results"]["val_all"]["per_class"]["train"])
            self.assertIn("person \u00e9", result["results"]["val_all"]["per_class"])
            with output_path.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertIn("AP75", rows[0])

    def synthetic_validation_root(self):
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "team_evaluator_check"
        output_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=output_dir)

    def write_validation_files(self, root):
        annotations = root / "annotations"
        (annotations / "subsets").mkdir(parents=True)
        (annotations / "val.json").write_text(
            json.dumps(self.gt, ensure_ascii=False), encoding="utf-8"
        )
        membership = {"clear_daytime": {91}, "night": {17}, "dawn_dusk": set(),
                      "rainy": {91, 17}, "overcast": {17}}
        for name, ids in membership.items():
            subset = {"images": [im for im in self.gt["images"] if im["id"] in ids],
                      "annotations": [a for a in self.gt["annotations"] if a["image_id"] in ids],
                      "categories": copy.deepcopy(self.gt["categories"])}
            path = annotations / "subsets" / f"val_{name}.json"
            path.write_text(json.dumps(subset), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
