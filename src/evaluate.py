"""Run Faster R-CNN validation and score predictions with the shared team evaluator."""

import torch
from torchvision.ops import box_convert

from evaluate_team import evaluate as evaluate_team_predictions

from .dataset import BDD100KDataset
from .model import WEIGHTS, get_transform_settings


NOT_EVALUABLE = "Not evaluable"
METRIC_NAMES = (
    "bbox_mAP", "bbox_AP50", "bbox_AP75", "bbox_AP_small", "bbox_AP_medium",
    "bbox_AP_large", "bbox_AR_max1", "bbox_AR_max10", "bbox_AR_max100",
    "bbox_AR_small", "bbox_AR_medium", "bbox_AR_large",
)


def evaluate(model, loader, dataset, device, *, image_ids=None):
    """Return (report, COCO predictions), without writing any files.

    By default the loader must cover the complete fixed validation split once.
    Explicit image_ids are reserved for small functionality checks and always
    produce performance_result=False, even when all validation IDs are listed.
    """
    if not isinstance(dataset, BDD100KDataset) or dataset.split != "val":
        raise ValueError("Evaluation requires the fixed validation BDD100KDataset.")
    purpose = "validation" if image_ids is None else "functional_check"
    expected_ids = _check_image_ids(
        dataset.coco, dataset.image_ids if image_ids is None else image_ids
    )
    if purpose == "validation" and set(expected_ids) != set(dataset.coco.imgs):
        raise ValueError("Validation must cover the complete fixed split.")
    seen_ids = set()
    predictions = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for images, targets in loader:
                if not images or len(images) != len(targets):
                    raise ValueError("Validation image and target counts must match.")
                batch_ids = []
                for image, target in zip(images, targets):
                    image_id = target["image_id"]
                    if image_id.dtype != torch.int64 or image_id.shape != (1,):
                        raise ValueError("image_id must be a one-element int64 tensor.")
                    image_id = int(image_id.item())
                    if image_id not in expected_ids or image_id in seen_ids:
                        raise ValueError(f"Unexpected or duplicate image ID: {image_id}")
                    record = dataset.coco.imgs[image_id]
                    if tuple(image.shape) != (3, record["height"], record["width"]):
                        raise ValueError(f"Image {image_id}: original dimensions changed.")
                    seen_ids.add(image_id)
                    batch_ids.append(image_id)
                outputs = model([image.to(device) for image in images])
                if len(outputs) != len(images):
                    raise ValueError("The model must return one output per image.")
                for image_id, output in zip(batch_ids, outputs):
                    predictions.extend(_prediction_rows(output, image_id))

        if seen_ids != set(expected_ids):
            raise ValueError(f"Validation loader omitted image IDs: {sorted(set(expected_ids) - seen_ids)}")
        report = evaluate_predictions(dataset.coco, predictions, expected_ids, purpose=purpose)
        transform, heads, rpn = model.transform, model.roi_heads, model.rpn
        report["split"] = "val"
        report["settings"].update({
            "device": str(device), "model_eval": True, "inference_mode": True,
            "model": {
                "architecture": "fasterrcnn_resnet50_fpn",
                "initial_weights": f"{WEIGHTS.__class__.__name__}.{WEIGHTS.name}",
                "num_classes": heads.box_predictor.cls_score.out_features,
                "transform": get_transform_settings(transform),
                "roi_heads": {
                    "score_threshold": heads.score_thresh, "nms_iou_threshold": heads.nms_thresh,
                    "detections_per_image": heads.detections_per_img,
                },
                "rpn": {
                    "score_threshold": rpn.score_thresh, "nms_iou_threshold": rpn.nms_thresh,
                    "pre_nms_top_n": rpn.pre_nms_top_n(), "post_nms_top_n": rpn.post_nms_top_n(),
                },
                "additional_score_filter": None,
            },
        })
        return report, predictions
    finally:
        model.train(was_training)


def evaluate_predictions(coco_gt, predictions, image_ids, *, purpose="synthetic_check"):
    """Adapt the team's scores to the training report; do not recalculate AP here."""
    if purpose not in ("validation", "functional_check", "synthetic_check"):
        raise ValueError("Unknown evaluation purpose.")
    image_ids = _check_image_ids(coco_gt, image_ids)
    if purpose == "validation" and set(image_ids) != set(coco_gt.imgs):
        raise ValueError("Formal validation requires every ground-truth image ID.")

    # Small checks select records in memory, preserving every original ID/category.
    selected_ids = set(image_ids)
    ground_truth = {
        **coco_gt.dataset,
        "images": [coco_gt.imgs[image_id] for image_id in image_ids],
        "annotations": [ann for ann in coco_gt.dataset["annotations"] if ann["image_id"] in selected_ids],
    }
    team_result = evaluate_team_predictions(ground_truth, predictions)
    team_names = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                  "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    metrics = {name: NOT_EVALUABLE if team_result[key] is None else team_result[key]
               for name, key in zip(METRIC_NAMES, team_names)}
    per_category = [
        {**row, **{key: NOT_EVALUABLE if row[key] is None else row[key] for key in ("AP", "AP50", "AP75")}}
        for row in team_result["per_category"]
    ]
    return {
        "purpose": purpose,
        "performance_result": purpose == "validation",
        "note": ("Full validation scored by the shared evaluate_team.py."
                 if purpose == "validation" else "Functionality check only; not formal validation performance."),
        "image_count": len(image_ids), "image_ids": image_ids,
        "prediction_count": len(predictions), "metrics": metrics, "per_category": per_category,
        "settings": {**team_result["settings"], "shared_evaluator": "evaluate_team.py",
                     "undefined_metric": NOT_EVALUABLE, "shared_undefined_metric": None},
        "coco_summary": team_result["coco_summary"],
    }


def _check_image_ids(coco_gt, image_ids):
    image_ids = list(image_ids)
    if (not image_ids or any(type(value) is not int for value in image_ids)
            or len(set(image_ids)) != len(image_ids) or not set(image_ids) <= set(coco_gt.imgs)):
        raise ValueError("Evaluation requires unique, original integer COCO image IDs.")
    return image_ids


def _prediction_rows(output, image_id):
    boxes, labels, scores = (output[key].detach().cpu() for key in ("boxes", "labels", "scores"))
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.is_floating_point()
            or labels.shape != (len(boxes),) or labels.dtype != torch.int64
            or scores.shape != (len(boxes),) or not scores.is_floating_point()):
        raise ValueError(f"Image {image_id}: expected boxes [N,4], int64 labels [N], and scores [N].")
    boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
    return [
        {"image_id": image_id, "category_id": label, "bbox": box, "score": score}
        for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist())
    ]
