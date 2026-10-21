"""Prepare the audited legacy five-epoch continuation on CPU; never train a model."""

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from . import train
from .model import REPO_ROOT, get_transform_settings


SOURCE_RUN = REPO_ROOT / "outputs/training/20261020_182545_580059"
OUTPUT_RUN = REPO_ROOT / "outputs/training/20261021_legacy5_to_epoch10"
# Fingerprints from the read-only audit, not inferred from a directory name.
SOURCE_HASHES = {
    "latest.pth": "22c4b04f22e42fc6bb104c192d28d3199735df21826057da5b92674c657cf1a9",
    "best.pth": "53945d6e5bfddaff18db0c4dce07a300b56369172f5ebf8d9e028add1b992577",
    "history.csv": "a30473a043f7e7aa96ba6fa7190fdd94e1b96b41a6fe2efcae5852996203f620",
    "validation.json": "88fdc71e6dd8c7548e5734a37c50a7a28ad13d3f26d75a14d9d2e7f6f45515c4",
    "predictions.json": "0c5ff568b84212622134a5300b7d2602f0a965ba3cd9bceac1f58b19d3a81d78",
}
LEGACY_SETTINGS = {
    "seed": 42, "batch_size": 2, "val_batch_size": 2, "device": "cuda:0",
    "optimizer": {"lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005},
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compatible_settings(report):
    """Expand only the audited native transform using saved fields and source evidence."""
    transform = get_transform_settings(GeneralizedRCNNTransform(
        800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ))
    recorded = report["settings"]["model"]["transform"]
    fields = {"image_mean", "image_std", "min_size", "max_size", "size_divisible", "fixed_size"}
    if set(recorded) != fields or any(recorded[key] != transform[key] for key in fields):
        raise ValueError("Legacy report does not confirm the audited native 800/1333 transform.")
    if report["settings"]["model"]["roi_heads"] != {
        "score_threshold": 0.05, "nms_iou_threshold": 0.5, "detections_per_image": 100
    }:
        raise ValueError("Legacy inference thresholds differ from the audited defaults.")
    if (train.IMAGE_SIZE is not None or train.EPOCHS != 10 or train.SEED != 42
            or train.BATCH_SIZE != 2 or train.VAL_BATCH_SIZE != 2 or train.DEVICE != "cuda:0"
            or (train.LEARNING_RATE, train.MOMENTUM, train.WEIGHT_DECAY) != (0.005, 0.9, 0.0005)):
        raise ValueError("Current training constants differ from the authorized continuation plan.")
    return {**LEGACY_SETTINGS, "num_workers": 0, "transform": transform}


def upgraded_checkpoint(saved, settings, continuation, best_source):
    """Preserve all saved state; add only explicitly evidenced settings and lineage."""
    if saved.get("kind") != "bdd100k_simple_training_v1" or saved["settings"] != LEGACY_SETTINGS:
        raise ValueError("Only the audited legacy settings/format may be upgraded explicitly.")
    train.validate_resume_state(saved, 3500)
    group, = saved["optimizer_state_dict"]["param_groups"]
    if any(group[key] != value for key, value in LEGACY_SETTINGS["optimizer"].items()):
        raise ValueError("Saved SGD hyperparameters do not match the legacy settings.")
    if group["dampening"] != 0 or group["nesterov"] or group["maximize"]:
        raise ValueError("Unexpected legacy SGD behavior.")
    rng = saved["rng_state"]
    random.Random().setstate(rng["python"])
    name, keys, position, has_gauss, cached_gauss = rng["numpy"]
    np.random.RandomState().set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.Generator(device="cpu").set_state(rng["cpu"])
    if len(rng["cuda"]) != 1 or any(t.dtype != torch.uint8 or t.ndim != 1 for t in rng["cuda"]):
        raise ValueError("Expected the original single-device CUDA RNG state; no RNG may be fabricated.")
    return {**saved, "settings": settings, "continuation": continuation, "best_source": best_source}


def prepare_continuation(source_dir=SOURCE_RUN, output_dir=OUTPUT_RUN):
    """Materialize compatible copies in a new directory, without invoking the trainer."""
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir != SOURCE_RUN.resolve():
        raise ValueError("This explicit migration is restricted to the audited old five-epoch run.")
    if output_dir.parent != (REPO_ROOT / "outputs/training").resolve() or output_dir == source_dir:
        raise ValueError("Use a new independent directory under outputs/training.")
    if output_dir.exists():
        raise FileExistsError("Preparation never overwrites an existing output directory.")
    hashes = {name: sha256(source_dir / name) for name in SOURCE_HASHES}
    if hashes != SOURCE_HASHES:
        raise ValueError("Source artifacts changed since the audit; inspect them before preparing again.")
    report = json.loads((source_dir / "validation.json").read_text(encoding="utf-8"))
    if (report["epoch"] != 5 or report["split"] != "val" or report["performance_result"] is not True
            or report["image_count"] != 1000 or report["image_ids"] != list(range(9001, 10001))):
        raise ValueError("Source JSON must remain the complete epoch-5 validation report.")
    settings = compatible_settings(report)
    latest = torch.load(source_dir / "latest.pth", map_location="cpu", weights_only=True)
    best = torch.load(source_dir / "best.pth", map_location="cpu", weights_only=True)
    if (latest["epoch"], latest["global_step"], best["epoch"], best["global_step"]) != (5, 17500, 4, 14000):
        raise ValueError("Unexpected source epoch/step identity.")
    if latest["best_mAP"] != 0.25544452083836444 or latest["best_epoch"] != 4:
        raise ValueError("Unexpected inherited best identity.")
    with (source_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [{key: str(value) for key, value in row.items()} for row in latest["history"]] != rows:
        raise ValueError("Saved history and CSV disagree.")
    if report["metrics"]["bbox_mAP"] != latest["history"][-1]["mAP"]:
        raise ValueError("Epoch-5 report and latest history disagree.")
    parent = source_dir.relative_to(REPO_ROOT).as_posix()
    continuation = {
        "parent_run": parent, "source_checkpoint": parent + "/latest.pth",
        "source_checkpoint_sha256": hashes["latest.pth"],
        "branch_epoch": 5, "branch_global_step": 17500,
        "first_new_epoch": 6, "target_epoch": 10,
        "inherited_best_checkpoint": parent + "/best.pth",
        "inherited_best_sha256": hashes["best.pth"],
    }
    best_source = {"checkpoint": parent + "/best.pth", "sha256": hashes["best.pth"],
                   "epoch": 4, "global_step": 14000, "mAP": latest["best_mAP"]}
    latest = upgraded_checkpoint(latest, settings, continuation, best_source)
    best = upgraded_checkpoint(best, settings, continuation, best_source)
    train.validate_best_checkpoint(latest, best, 3500)
    source_basis = {
        "num_workers": "a65cd736:src/train.py:79-80 explicitly uses num_workers=0 for both loaders.",
        "transform_saved_fields": "Parent validation.json settings.model.transform supplies mean/std, min/max size, size_divisible and fixed_size.",
        "transform_extended_fields": "a65cd736:src/model.py:20-36 leaves factory transform/pooler unchanged. Installed torchvision transform.py resize/batch_images/resize_boxes and native pooler establish every expanded field.",
        "historical_train_blob": "5a0cf8a1bd4c866ee74c8196cd2499fb59c04d65",
        "historical_model_blob": "2da62497d062e1a07a4b44f98a4ed7f163b36898",
        "dataset_blob_unchanged": "045b3880935fd4dadd8e3c51124ac84d8b36ed3e",
        "limit": "Original run did not record executed source/runtime/data hashes. This is evidenced metadata reconstruction, not proof of exact original execution identity or bitwise CUDA trajectory.",
    }
    manifest = {
        "preparation_status": "Prepared; training not started at preparation time",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_run": output_dir.relative_to(REPO_ROOT).as_posix(),
        "continuation": continuation, "best_source": best_source,
        "source_artifact_sha256": hashes, "metadata_added": {"num_workers": 0, "transform": settings["transform"]},
        "metadata_basis": source_basis,
        "preparation_runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
        "state_policy": "All original fields except settings are preserved; only settings.transform/settings.num_workers and continuation/best_source are added. Model, optimizer/momentum, history, counters and all RNG states are unchanged.",
        "history_policy": "Rows 1-5 are inherited; zero new training epochs completed during preparation.",
        "json_policy": "Parent JSON files remain in parent, belong to epoch 5, and are not copied. Epoch-4 detailed JSON/predictions are unavailable.",
        "new_validation_results": "Not measured: epochs 6-10 have not started.",
        "planned_data": {"train_images_per_epoch": 7000, "val_images_per_epoch": 1000},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    train.save_checkpoint(output_dir / "latest.pth", latest)
    train.save_checkpoint(output_dir / "best.pth", best)
    train.write_history(output_dir / "history.csv", latest["history"])
    manifest["prepared_checkpoint_sha256"] = {name: sha256(output_dir / name) for name in ("latest.pth", "best.pth")}
    (output_dir / "continuation.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_RUN)
    args = parser.parse_args()
    result = prepare_continuation(output_dir=args.output_dir)
    print(json.dumps({"status": result["preparation_status"], "output_run": result["output_run"],
                      "first_new_epoch": 6, "target_epoch": 10}, indent=2))


if __name__ == "__main__":
    main()
