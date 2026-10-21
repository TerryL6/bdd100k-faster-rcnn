"""Check evaluation with synthetic tensors and COCO records; no dataset access."""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pycocotools.coco import COCO

from src.dataset import BDD100KDataset
from src.evaluate import NOT_EVALUABLE, evaluate, evaluate_predictions
from src.model import SquarePadTransform, get_transform_settings
from evaluate_team import evaluate as team_evaluate


class SyntheticRPN(torch.nn.Module):
    score_thresh = 0.0
    nms_thresh = 0.7

    def pre_nms_top_n(self):
        return 2000 if self.training else 1000

    def post_nms_top_n(self):
        return 2000 if self.training else 1000


class SyntheticDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = SimpleNamespace(
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
            min_size=(800,), max_size=1333, size_divisible=32, fixed_size=None,
        )
        self.roi_heads = SimpleNamespace(
            score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
            box_predictor=SimpleNamespace(cls_score=SimpleNamespace(out_features=11)),
        )
        self.rpn = SyntheticRPN()

    def forward(self, images):
        if self.training or not torch.is_inference_mode_enabled():
            raise AssertionError("Inference must use eval() and inference_mode().")
        return [{"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                 "labels": torch.tensor([1]), "scores": torch.tensor([0.9])} for _ in images]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.coco = COCO()
        self.coco.dataset = {
            "images": [{"id": 9001, "width": 100, "height": 80},
                       {"id": 9002, "width": 100, "height": 80}],
            "categories": [{"id": 1, "name": "person"}, {"id": 6, "name": "train"}],
            "annotations": [{"id": 1, "image_id": 9001, "category_id": 1,
                             "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        }
        with redirect_stdout(io.StringIO()):
            self.coco.createIndex()
        self.prediction = {"image_id": 9001, "category_id": 1,
                           "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9}
        self.dataset = BDD100KDataset.__new__(BDD100KDataset)
        self.dataset.split = "val"
        self.dataset.coco = self.coco
        self.dataset.image_ids = [9001, 9002]
        self.image = torch.zeros((3, 80, 100))
        self.target1 = {"image_id": torch.tensor([9001])}
        self.target2 = {"image_id": torch.tensor([9002])}

    def test_perfect_predictions_and_missing_category(self):
        before = copy.deepcopy(self.coco.dataset)
        predictions = [copy.deepcopy(self.prediction)]
        report = evaluate_predictions(self.coco, predictions, [9001, 9002])
        self.assertAlmostEqual(report["metrics"]["bbox_mAP"], 1.0)
        self.assertEqual(len(report["metrics"]), 12)
        missing = report["per_category"][1]
        self.assertEqual(missing["name"], "train")
        self.assertEqual(missing["ground_truth_objects"], 0)
        for key in ("AP", "AP50", "AP75"):
            self.assertEqual(missing[key], NOT_EVALUABLE)
        self.assertEqual(self.coco.dataset, before)
        self.assertEqual(predictions, [self.prediction])
        self.assertEqual(report["settings"]["max_detections"], [1, 10, 100])
        self.assertEqual(len(report["settings"]["iou_thresholds"]), 10)
        self.assertEqual(len(report["settings"]["recall_thresholds"]), 101)
        json.dumps(report, allow_nan=False)

    def test_metrics_delegate_to_the_shared_evaluator(self):
        predictions = [{**self.prediction, "bbox": [10.0, 10.0, 200.0, 20.0]}]
        with patch("src.evaluate.evaluate_team_predictions", wraps=team_evaluate) as shared:
            report = evaluate_predictions(self.coco, predictions, [9001, 9002])
        shared.assert_called_once()
        direct = team_evaluate(self.coco.dataset, predictions)
        for local, team in (("bbox_mAP", "mAP"), ("bbox_AP50", "AP50"), ("bbox_AP75", "AP75")):
            self.assertEqual(report["metrics"][local], direct[team])
        self.assertEqual(report["settings"]["shared_evaluator"], "evaluate_team.py")
        self.assertIsNone(direct["per_class"]["train"])
        self.assertEqual(report["per_category"][1]["AP"], NOT_EVALUABLE)

    def test_empty_predictions_and_no_ground_truth(self):
        report = evaluate_predictions(self.coco, [], [9001, 9002])
        self.assertEqual(report["metrics"]["bbox_mAP"], 0.0)
        self.assertEqual(report["per_category"][1]["AP"], NOT_EVALUABLE)
        empty = evaluate_predictions(self.coco, [], [9002])
        self.assertTrue(all(value == NOT_EVALUABLE for value in empty["metrics"].values()))

    def test_prediction_id_and_value_errors(self):
        cases = [([], [9001, 9001]), ([], [9999]),
                 ([{**self.prediction, "image_id": 9002}], [9001]),
                 ([{**self.prediction, "category_id": 99}], [9001]),
                 ([{**self.prediction, "score": float("nan")}], [9001]),
                 ([{**self.prediction, "bbox": [10, 10, -1, 20]}], [9001])]
        for predictions, image_ids in cases:
            with self.subTest(predictions=predictions, image_ids=image_ids), self.assertRaises(ValueError):
                evaluate_predictions(self.coco, predictions, image_ids)

    def test_inference_ids_conversion_settings_and_mode(self):
        model = SyntheticDetector()
        loader = [([self.image, self.image], [self.target1, self.target2])]
        report, predictions = evaluate(model, loader, self.dataset, "cpu")
        self.assertTrue(model.training)
        for key in ("image_id", "category_id", "bbox"):
            self.assertEqual(predictions[0][key], self.prediction[key])
        self.assertEqual(report["image_ids"], [9001, 9002])
        self.assertTrue(report["performance_result"])
        self.assertEqual(report["settings"]["model"]["rpn"]["post_nms_top_n"], 1000)
        self.assertEqual(report["settings"]["model"]["roi_heads"]["score_threshold"], 0.05)
        json.dumps(report, allow_nan=False)

    def test_explicit_sample_scope_and_loader_coverage(self):
        model = SyntheticDetector()
        loader = [([self.image], [self.target1])]
        report, _ = evaluate(model, loader, self.dataset, "cpu", image_ids=[9001])
        self.assertEqual(report["purpose"], "functional_check")
        self.assertFalse(report["performance_result"])
        for bad_loader in (loader, [([self.image, self.image], [self.target1, self.target1])]):
            with self.assertRaises(ValueError):
                evaluate(model, bad_loader, self.dataset, "cpu")
            self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            evaluate_predictions(self.coco, [], [9001], purpose="validation")

    def test_report_records_fixed_canvas_and_normalized_padding(self):
        model = SyntheticDetector()
        model.transform = SquarePadTransform(640, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        loader = [([self.image, self.image], [self.target1, self.target2])]
        report, _ = evaluate(model, loader, self.dataset, "cpu")
        settings = report["settings"]["model"]["transform"]
        self.assertEqual(settings, get_transform_settings(model.transform))
        self.assertEqual(settings["canvas_size"], [640, 640])
        self.assertEqual(settings["padding_sides"], ["right", "bottom"])
        self.assertEqual(settings["padding_value_space"], "normalized")
        self.assertEqual(settings["roi_pool_scale_reference"], "canvas_size")

    def test_empty_and_nonfinite_model_outputs(self):
        model = SyntheticDetector()
        loader = [([self.image], [self.target1])]
        empty = {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64),
                 "scores": torch.empty(0)}
        with patch.object(model, "forward", return_value=[empty]):
            report, predictions = evaluate(model, loader, self.dataset, "cpu", image_ids=[9001])
        self.assertEqual(predictions, [])
        self.assertEqual(report["metrics"]["bbox_mAP"], 0.0)
        invalid = {"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                   "labels": torch.tensor([1]), "scores": torch.tensor([float("nan")])}
        with patch.object(model, "forward", return_value=[invalid]), self.assertRaises(ValueError):
            evaluate(model, loader, self.dataset, "cpu", image_ids=[9001])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
