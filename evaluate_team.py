"""Shared COCO bbox scoring for original dataset IDs and fixed subsets.

Predictions use original image/category IDs, pixel xywh boxes and scores in
[0, 1]. Undefined metrics are JSON null (Not evaluable), excluded from means.
The evaluator applies no score filter, NMS, clipping or ID remapping.
"""

import argparse
import copy
import csv
import io
import json
import math
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


SUMMARY_NAMES = (
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)
VAL_SUBSETS = ("clear_daytime", "night", "dawn_dusk", "rainy", "overcast")
PROTOCOL = {
    "metric": "COCO bbox AP",
    "iou": "0.50:0.05:0.95",
    "maxDets": [1, 10, 100],
    "scale": "0 to 1",
    "absent_classes": None,
    "undefined_metric": "JSON null means Not evaluable; excluded from means",
    "score_filter": "none applied by evaluator; export low-score candidates consistently",
    "drop_definition": "clear_daytime mAP minus subset mAP; descriptive, not causal",
}


def validate_predictions(predictions, gt):
    """Reject invalid COCO prediction rows without changing their contents."""
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON array.")
    image_ids = {image["id"] for image in gt["images"]}
    category_ids = {category["id"] for category in gt["categories"]}
    for index, prediction in enumerate(predictions):
        prefix = f"Prediction {index}: "
        if not isinstance(prediction, dict):
            raise ValueError(prefix + "expected an object.")
        image_id = prediction.get("image_id")
        category_id = prediction.get("category_id")
        if type(image_id) is not int or image_id not in image_ids:
            raise ValueError(prefix + "unknown/incompatible image_id.")
        if type(category_id) is not int or category_id not in category_ids:
            raise ValueError(prefix + "unknown/incompatible category_id.")
        box = prediction.get("bbox")
        if (not isinstance(box, list) or len(box) != 4
                or not all(_finite_number(value) for value in box)
                or box[2] <= 0 or box[3] <= 0):
            raise ValueError(prefix + "bbox must be finite pixel xywh with positive size.")
        score = prediction.get("score")
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError(prefix + "score must be finite and in [0, 1].")


def metric(values):
    """Average evaluable COCO values, keeping undefined values as None."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("COCOeval produced a non-finite metric.")
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def evaluate(gt, predictions):
    """Score COCO xywh predictions with the group's unchanged bbox protocol.

    This retains the original images/objects/mAP/AP50/per_class fields and adds
    AP75, size AP, recall, category counts and explicit evaluation settings.
    pycocotools mutates annotations and detections; isolate both input copies.
    """
    validate_predictions(predictions, gt)
    ground_truth = copy.deepcopy(gt)
    ground_truth.setdefault("info", {})
    summary = io.StringIO()
    with redirect_stdout(summary):
        coco = COCO()
        coco.dataset = ground_truth
        coco.createIndex()
        if predictions:
            detections = coco.loadRes(copy.deepcopy(predictions))
        else:
            # COCO.loadRes does not support an empty prediction list.
            detections = COCO()
            detections.dataset = {
                "images": copy.deepcopy(ground_truth["images"]),
                "categories": copy.deepcopy(ground_truth["categories"]),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = sorted(image["id"] for image in gt["images"])
        evaluator.params.maxDets = [1, 10, 100]
        # Empty image/category sets have no precision tensor to summarize.
        if gt["images"] and gt["categories"]:
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

    result = {"images": len(gt["images"]), "objects": len(gt["annotations"])}
    if "precision" in evaluator.eval:
        result.update(zip(SUMMARY_NAMES, (metric(value) for value in evaluator.stats)))
    else:
        result.update({name: None for name in SUMMARY_NAMES})

    result["per_class"] = {}
    result["per_class_AP50"] = {}
    result["per_class_AP75"] = {}
    result["per_category"] = []
    for category in gt["categories"]:
        category_id, name = category["id"], category["name"]
        annotations = [ann for ann in gt["annotations"]
                       if ann["category_id"] == category_id]
        ap = ap50 = ap75 = None
        if "precision" in evaluator.eval:
            # COCOeval sorts category IDs, even if the input list is unsorted.
            category_index = list(evaluator.params.catIds).index(category_id)
            precision = evaluator.eval["precision"][:, :, category_index, 0, -1]
            ap = metric(precision)
            ap50 = metric(precision[np.isclose(evaluator.params.iouThrs, 0.50)])
            ap75 = metric(precision[np.isclose(evaluator.params.iouThrs, 0.75)])
        result["per_class"][name] = ap
        result["per_class_AP50"][name] = ap50
        result["per_class_AP75"][name] = ap75
        result["per_category"].append({
            "category_id": category_id,
            "name": name,
            "ground_truth_objects": len(annotations),
            "noncrowd_ground_truth_objects": sum(
                not ann.get("iscrowd", 0) for ann in annotations
            ),
            "AP": ap, "AP50": ap50, "AP75": ap75,
        })

    params = evaluator.params
    result["settings"] = {
        "evaluator": "pycocotools.cocoeval.COCOeval",
        "pycocotools_version": version("pycocotools"),
        "iou_type": "bbox",
        "iou_thresholds": params.iouThrs.tolist(),
        "recall_thresholds": params.recThrs.tolist(),
        "max_detections": list(params.maxDets),
        "area_ranges": params.areaRng,
        "area_range_labels": list(params.areaRngLbl),
        "category_ids": [int(value) for value in params.catIds],
        "use_categories": bool(params.useCats),
        "metric_scale": "0 to 1",
        "undefined_metric": "JSON null means Not evaluable",
        "missing_category_policy": "Categories with no evaluable GT are excluded from means.",
        "score_filter": "none applied by evaluator",
        "prediction_coordinates": "Original image pixels in COCO xywh format.",
    }
    result["coco_summary"] = summary.getvalue()
    return result


def evaluate_validation(root, predictions, *, include_subsets=True):
    """Score all val predictions and optionally the five fixed condition groups.

    Subsets overlap. Filter only by original image IDs, reuse predictions, and
    retain the original clear-daytime-minus-subset mAP difference definition.
    No images or test annotations are opened; source files are never written.
    """
    root = Path(root)
    gt = _read_json(root / "annotations" / "val.json")
    results = {"val_all": evaluate(gt, predictions)}
    if include_subsets:
        for name in VAL_SUBSETS:
            subset = _read_json(root / "annotations" / "subsets" / f"val_{name}.json")
            image_ids = {image["id"] for image in subset["images"]}
            subset_predictions = [row for row in predictions if row["image_id"] in image_ids]
            results[f"val_{name}"] = evaluate(subset, subset_predictions)
        base = results["val_clear_daytime"]["mAP"]
        for item in results.values():
            value = item["mAP"]
            item["mAP_drop_vs_clear_daytime"] = (
                None if base is None or value is None else base - value
            )
    return {"protocol": copy.deepcopy(PROTOCOL), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = _read_json(args.predictions)
    if args.split == "val":
        output = evaluate_validation(args.root, predictions)
    else:
        # Final testing is an explicit separate invocation, never model selection.
        gt = _read_json(args.root / "annotations" / "test.json")
        output = {"protocol": copy.deepcopy(PROTOCOL),
                  "results": {"test_all": evaluate(gt, predictions)}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["subset", "images", "objects", *SUMMARY_NAMES, "mAP_drop_vs_clear_daytime"]
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, row in output["results"].items():
            writer.writerow({"subset": name, **{key: row.get(key) for key in fields[1:]}})
    print(json.dumps(output, indent=2, allow_nan=False))


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


if __name__ == "__main__":
    main()
