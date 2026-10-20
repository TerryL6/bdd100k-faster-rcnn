"""Check one GPU loss/backward batch and one inference batch without updates."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
from torch.utils.data import Subset

from .dataset import BDD100KDataset, make_dataloader
from .model import NUM_CLASSES, REPO_ROOT, TORCH_HUB_DIR, WEIGHTS, build_model


def require(condition, message):
    """Keep smoke-check failures active even when Python uses -O."""
    if not condition:
        raise RuntimeError(message)


def load_batch(data_root, split):
    """Read the first and last image of a fixed split as one two-image batch."""
    dataset = BDD100KDataset(data_root, split)
    require(list(dataset.category_names) == list(range(1, NUM_CLASSES)),
            "Expected dataset category IDs 1-10 in their original order.")
    require(len(dataset) >= 2, "The check requires two distinct images per split.")
    indices = [0, len(dataset) - 1]
    loader = make_dataloader(Subset(dataset, indices), batch_size=2,
                             shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    records = [
        {"index": index, "image_id": target["image_id"].item(),
         "file_name": dataset.coco.imgs[target["image_id"].item()]["file_name"],
         "image_shape": list(image.shape), "object_count": len(target["boxes"])}
        for index, image, target in zip(indices, images, targets)
    ]
    return images, targets, records


def check_predictions(predictions, images, resized_boxes, resized_sizes):
    """Check public outputs and independently verify the coordinate rescaling."""
    require(isinstance(predictions, list) and len(predictions) == len(images),
            "Expected one prediction dictionary per input image.")
    require(len(resized_boxes) == len(resized_sizes) == len(images),
            "Missing internal inference coordinate evidence.")
    summaries = []
    for prediction, image, raw_boxes, resized_size in zip(
            predictions, images, resized_boxes, resized_sizes):
        require(isinstance(prediction, dict), "Prediction must be a dictionary.")
        require({"boxes", "labels", "scores"} <= prediction.keys(),
                "Missing prediction fields.")
        boxes, labels, scores = (prediction[key] for key in ("boxes", "labels", "scores"))
        count = len(boxes)
        for value, dtype, shape in (
            (boxes, torch.float32, (count, 4)),
            (labels, torch.int64, (count,)),
            (scores, torch.float32, (count,)),
        ):
            require(isinstance(value, torch.Tensor), "Prediction field must be a tensor.")
            require(value.dtype == dtype and tuple(value.shape) == shape,
                    "Unexpected prediction dtype or shape.")
            require(value.device == image.device, "Prediction left the input device.")
            require(not value.requires_grad and torch.isfinite(value).all().item(),
                    "Prediction must be finite and require no gradients.")
        height, width = image.shape[-2:]
        require(((labels >= 1) & (labels < NUM_CLASSES)).all().item(),
                "Prediction labels must use dataset IDs 1-10, without background.")
        require(((scores >= 0) & (scores <= 1)).all().item(), "Scores must be in [0, 1].")
        require((boxes[:, 2:] > boxes[:, :2]).all().item(), "Degenerate predicted box.")
        require((boxes >= 0).all().item(), "Negative predicted coordinate.")
        require((boxes[:, [0, 2]] <= width + 1e-3).all().item(), "Box exceeds input width.")
        require((boxes[:, [1, 3]] <= height + 1e-3).all().item(), "Box exceeds input height.")
        resized_height, resized_width = resized_size
        scale = boxes.new_tensor([width / resized_width, height / resized_height,
                                  width / resized_width, height / resized_height])
        torch.testing.assert_close(boxes, raw_boxes * scale, rtol=1e-6, atol=1e-4)
        summaries.append({
            "detections": count,
            "shapes": {key: list(prediction[key].shape) for key in ("boxes", "labels", "scores")},
            "dtypes": {key: str(prediction[key].dtype) for key in ("boxes", "labels", "scores")},
            "original_size_hw": [height, width],
            "resized_size_hw": list(resized_size),
            "coordinate_rescaling_verified": True,
            "coordinate_check_nonempty": count > 0,
        })
    return summaries


def run_check(data_root):
    require(torch.cuda.is_available(), "CUDA is required; this check does not fall back to CPU.")
    seed = 0  # Smoke-check initialization/sampling only; not a training seed decision.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    print(f"Using {sys.executable}; GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Loading {WEIGHTS.__class__.__name__}.{WEIGHTS.name}; cache: {TORCH_HUB_DIR}", flush=True)
    model = build_model(progress=False).to(device)
    require(model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES,
            "Wrong classification head size.")
    require(model.roi_heads.box_predictor.bbox_pred.out_features == NUM_CLASSES * 4,
            "Wrong box-regression head size.")
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    transform_batches = []
    resized_boxes = []

    def capture_transform(module, inputs, output):
        image_list, _ = output
        transform_batches.append({"mode": "train" if module.training else "eval",
                                  "resized_sizes_hw": [list(size) for size in image_list.image_sizes],
                                  "padded_batch_shape": list(image_list.tensors.shape)})

    def capture_roi_boxes(module, inputs, output):
        if not module.training:
            # postprocess replaces dict entries later; clone before that happens.
            resized_boxes.extend(item["boxes"].detach().clone() for item in output[0])

    handles = [model.transform.register_forward_hook(capture_transform),
               model.roi_heads.register_forward_hook(capture_roi_boxes)]
    torch.cuda.reset_peak_memory_stats(device)
    try:
        images_cpu, targets_cpu, train_records = load_batch(data_root, "train")
        images = [image.to(device) for image in images_cpu]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets_cpu]
        model.train()
        model.zero_grad(set_to_none=True)
        print("Running one two-image training forward pass (float32, no AMP).", flush=True)
        losses = model(images, targets)
        expected_losses = {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
        require(isinstance(losses, dict) and set(losses) == expected_losses, "Unexpected loss fields.")
        for name, value in losses.items():
            require(value.ndim == 0 and value.requires_grad and torch.isfinite(value).item(),
                    f"Invalid loss: {name}.")
        total_loss = sum(losses.values())
        require(torch.isfinite(total_loss).item(), "Total loss is not finite.")
        loss_values = {name: value.item() for name, value in losses.items()}
        print(f"Losses: {loss_values}; total: {total_loss.item():.6f}", flush=True)
        total_loss.backward()
        gradient_names = []
        nonzero_names = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                require(parameter.grad is not None, f"Missing gradient: {name}.")
                require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}.")
                gradient_names.append(name)
                if torch.count_nonzero(parameter.grad).item():
                    nonzero_names.append(name)
        require(any(name.startswith("backbone.") for name in nonzero_names), "No nonzero backbone gradient.")
        require(any(name.startswith("roi_heads.box_predictor.") for name in nonzero_names),
                "No nonzero prediction-head gradient.")
        for source, target in zip(targets_cpu, targets):
            for key in source:
                torch.testing.assert_close(source[key], target[key].cpu(), rtol=0, atol=0)
        training_report = {"samples": train_records, "losses": loss_values,
                           "total_loss": total_loss.item(), "backward_completed": True,
                           "finite_gradient_tensors": len(gradient_names),
                           "nonzero_gradient_tensors": len(nonzero_names), "optimizer_steps": 0,
                           "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        model.zero_grad(set_to_none=True)
        del losses, total_loss, images, targets, images_cpu, targets_cpu
        print("Backward and finite-gradient checks passed; no optimizer was created.", flush=True)

        images_cpu, _, val_records = load_batch(data_root, "val")
        images = [image.to(device) for image in images_cpu]
        model.eval()
        print("Running two validation images with eval() and inference_mode().", flush=True)
        with torch.inference_mode():
            predictions = model(images)
            prediction_summaries = check_predictions(
                predictions, images, resized_boxes, transform_batches[-1]["resized_sizes_hw"])
        for name, value in model.state_dict().items():
            require(torch.equal(value.detach().cpu(), initial_state[name]), f"Model state changed: {name}.")
        print("Prediction checks passed; all model parameters and buffers are exactly unchanged.", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    weight_path = TORCH_HUB_DIR / "checkpoints" / Path(urlparse(WEIGHTS.url).path).name
    with weight_path.open("rb") as handle:
        weight_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    transform = model.transform
    return {
        "status": "passed", "completed_at": datetime.now().astimezone().isoformat(),
        "purpose": "Model integration smoke check; not training or a performance evaluation.",
        "environment": {"python_executable": sys.executable, "python": sys.version.split()[0],
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "cuda_runtime": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device)},
        "check_configuration": {"seed": seed, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "dtype": "torch.float32", "amp": False, "batch_size": 2,
                                "num_workers": 0, "shuffle": False, "data_root": str(data_root.resolve())},
        "model": {"architecture": "fasterrcnn_resnet50_fpn", "num_classes_including_background": NUM_CLASSES,
                  "background_id": 0, "foreground_ids": list(range(1, NUM_CLASSES)),
                  "head": "New randomly initialized FastRCNNPredictor; not fine-tuned",
                  "weights": f"{WEIGHTS.__class__.__name__}.{WEIGHTS.name}", "weights_url": WEIGHTS.url,
                  "weights_path": str(weight_path), "weights_bytes": weight_path.stat().st_size,
                  "weights_sha256": weight_sha256,
                  "trainable_backbone_body_stages": sorted({name.split('.')[2] for name, parameter in
                      model.named_parameters() if name.startswith("backbone.body.") and parameter.requires_grad})},
        "transform": {"min_size": list(transform.min_size), "max_size": transform.max_size,
                      "image_mean": transform.image_mean, "image_std": transform.image_std,
                      "size_divisible": transform.size_divisible, "fixed_size": transform.fixed_size,
                      "interpolation": "bilinear", "align_corners": False,
                      "batches": transform_batches},
        "inference_defaults": {"box_score_thresh": model.roi_heads.score_thresh,
                               "box_nms_thresh": model.roi_heads.nms_thresh,
                               "box_detections_per_img": model.roi_heads.detections_per_img},
        "training_mode_check": training_report,
        "inference_mode_check": {"samples": val_records, "predictions": prediction_summaries},
        "parameters_and_buffers_exactly_unchanged": True,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "performance_metrics": "Not measured: untrained replacement head; smoke check only.",
        "test_evaluation": "Not evaluated: no test split access.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bdd100k_subset_10k")
    args = parser.parse_args()
    output_dir = REPO_ROOT / "outputs" / "model_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove ambiguity from an older successful report when a later attempt fails.
    report = {"status": "running", "started_at": datetime.now().astimezone().isoformat()}
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        report = run_check(args.data_root)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Model checks passed. Report: {report_path}")
    print("No optimizer step, checkpoint, AP calculation, tuning, or test evaluation was performed.")


if __name__ == "__main__":
    main()
