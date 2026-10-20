"""Check a few real SGD updates and checkpoint reloads on training data only."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Subset

from .dataset import BDD100KDataset, make_dataloader
from .model import NUM_CLASSES, REPO_ROOT, WEIGHTS, build_model


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_equal(actual, expected, name):
    """Compare saved/loaded tensor trees exactly, regardless of tensor device."""
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()),
                f"Checkpoint mismatch: {name}.")
    elif isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"Checkpoint keys differ: {name}.")
        for key in expected:
            check_equal(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f"Checkpoint length differs: {name}.")
        for index, (left, right) in enumerate(zip(actual, expected)):
            check_equal(left, right, f"{name}.{index}")
    else:
        require(actual == expected, f"Checkpoint value differs: {name}.")


def make_optimizer(model, config):
    return torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        **config["optimizer"],
    )


def train_step(model, optimizer, images_cpu, targets_cpu, device):
    """Perform one update and verify finite values and actual parameter changes."""
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    images = [image.to(device) for image in images_cpu]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets_cpu]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model(images, targets)
    require(set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"},
            "Unexpected loss fields.")
    for name, value in losses.items():
        require(value.ndim == 0 and value.requires_grad and torch.isfinite(value).item(),
                f"Invalid loss: {name}.")
    total_loss = sum(losses.values())
    require(torch.isfinite(total_loss).item(), "Non-finite total loss.")
    total_loss.backward()
    finite_gradients = 0
    nonzero_gradients = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None, f"Missing gradient: {name}.")
            require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}.")
            finite_gradients += 1
            nonzero_gradients += int(torch.count_nonzero(parameter.grad).item() > 0)
    require(nonzero_gradients > 0, "All gradients are zero.")
    optimizer.step()
    changed = []
    for name, parameter in model.named_parameters():
        require(torch.isfinite(parameter).all().item(), f"Non-finite parameter after update: {name}.")
        equal = torch.equal(before[name], parameter.detach().cpu())
        if parameter.requires_grad:
            if not equal:
                changed.append(name)
        else:
            require(equal, f"Frozen parameter changed: {name}.")
    require(any(name.startswith("backbone.") for name in changed), "Backbone did not update.")
    require(any(name.startswith("roi_heads.box_predictor.") for name in changed),
            "Replacement predictor did not update.")
    for state in optimizer.state.values():
        require(torch.isfinite(state["momentum_buffer"]).all().item(), "Non-finite SGD momentum.")
    optimizer.zero_grad(set_to_none=True)
    return {
        "image_ids": [target["image_id"].item() for target in targets_cpu],
        "object_counts": [len(target["boxes"]) for target in targets_cpu],
        "losses": {name: value.item() for name, value in losses.items()},
        "total_loss": total_loss.item(), "finite_gradient_tensors": finite_gradients,
        "nonzero_gradient_tensors": nonzero_gradients,
        "changed_parameter_tensors": len(changed), "frozen_parameters_unchanged": True,
        "backbone_updated": True, "replacement_head_updated": True,
    }


def save_checkpoint(path, model, optimizer, step, config, dataset, indices):
    """Save only tensors/basic Python values, compatible with weights_only=True."""
    transform = model.transform
    torch.save({
        "format_version": 1,
        "purpose": "Bounded training check, not a selected baseline checkpoint",
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step, "completed_epochs": 0,
        "next_sample_offset": step * config["batch_size"],
        "config": config, "train_indices": indices,
        "train_image_ids": [dataset.image_ids[index] for index in indices],
        "categories": dataset.coco.dataset["categories"],
        "model": {"architecture": "fasterrcnn_resnet50_fpn", "num_classes": NUM_CLASSES,
                  "weights": f"{WEIGHTS.__class__.__name__}.{WEIGHTS.name}",
                  "min_size": list(transform.min_size), "max_size": transform.max_size,
                  "image_mean": transform.image_mean, "image_std": transform.image_std,
                  "size_divisible": transform.size_divisible, "fixed_size": transform.fixed_size},
        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
    }, path)


def verify_checkpoint(path, model, optimizer, expected_step, config, dataset, indices):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved["format_version"] == 1 and saved["completed_epochs"] == 0,
            "Unexpected training-check checkpoint format.")
    transform = model.transform
    check_equal({"architecture": "fasterrcnn_resnet50_fpn", "num_classes": NUM_CLASSES,
                 "weights": f"{WEIGHTS.__class__.__name__}.{WEIGHTS.name}",
                 "min_size": list(transform.min_size), "max_size": transform.max_size,
                 "image_mean": transform.image_mean, "image_std": transform.image_std,
                 "size_divisible": transform.size_divisible, "fixed_size": transform.fixed_size},
                saved["model"], "model_definition")
    check_equal(model.state_dict(), saved["model_state_dict"], "model")
    check_equal(optimizer.state_dict(), saved["optimizer_state_dict"], "optimizer")
    check_equal(config, saved["config"], "config")
    check_equal(dataset.coco.dataset["categories"], saved["categories"], "categories")
    check_equal(indices, saved["train_indices"], "train_indices")
    check_equal([dataset.image_ids[index] for index in indices], saved["train_image_ids"], "image_ids")
    require(saved["global_step"] == expected_step, "Incorrect checkpoint step.")
    require(saved["next_sample_offset"] == expected_step * config["batch_size"],
            "Incorrect next sample offset.")
    for parameter, state in optimizer.state.items():
        require(state["momentum_buffer"].device == parameter.device, "SGD state is on the wrong device.")
    return saved


def run_check(config, data_root, output_dir, report):
    require(torch.cuda.is_available(), "CUDA is required; CPU fallback is not enabled.")
    require(type(config["seed"]) is int and config["seed"] >= 0, "Seed must be a nonnegative integer.")
    require(type(config["steps"]) is int and 2 <= config["steps"] <= 10,
            "This bounded check accepts only 2-10 total steps.")
    require(type(config["batch_size"]) is int and 1 <= config["batch_size"] <= 2,
            "This bounded check accepts only batch sizes 1-2.")
    require(set(config) == {"seed", "batch_size", "steps", "optimizer"}, "Unknown configuration fields.")
    require(set(config["optimizer"]) == {"lr", "momentum", "weight_decay"}, "Unexpected SGD settings.")
    for key, value in config["optimizer"].items():
        require(isinstance(value, (float, int)) and math.isfinite(value), f"Invalid SGD value: {key}.")
    require(config["optimizer"]["lr"] > 0 and 0 < config["optimizer"]["momentum"] < 1
            and config["optimizer"]["weight_decay"] >= 0, "Invalid SGD settings.")
    device = torch.device("cuda:0")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = build_model(progress=False).to(device)
    optimizer = make_optimizer(model, config)
    dataset = BDD100KDataset(data_root, "train")
    require(list(dataset.category_names) == list(range(1, NUM_CLASSES)), "Unexpected category IDs.")
    indices = list(range(config["steps"] * config["batch_size"]))
    require(len(dataset) >= len(indices), "Insufficient training images.")
    loader = make_dataloader(Subset(dataset, indices), batch_size=config["batch_size"],
                             shuffle=False, num_workers=0)
    iterator = iter(loader)
    report.update(
        config=config, environment={"python": sys.version.split()[0], "executable": sys.executable,
                                    "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
        check_settings={"split": "train", "data_root": str(data_root.resolve()),
                        "train_indices": indices, "batch_size": config["batch_size"],
                        "shuffle": False, "num_workers": 0, "dtype": "torch.float32", "amp": False,
                        "scheduler": None, "augmentation": None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        samples=[{"image_id": dataset.image_ids[index],
                  "file_name": dataset.coco.imgs[dataset.image_ids[index]]["file_name"]} for index in indices],
    )
    torch.cuda.reset_peak_memory_stats(device)

    def record_step(step, batch, resumed):
        result = train_step(model, optimizer, *batch, device)
        result.update(step=step, resumed_from_checkpoint=resumed)
        report["steps"].append(result)
        with (output_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
        write_json(output_dir / "report.json", report)
        print(f"step={step}, image_ids={result['image_ids']}, loss={result['total_loss']:.6f}, "
              f"changed_parameters={result['changed_parameter_tensors']}, resumed={resumed}", flush=True)

    # The final step is deliberately completed only after a disk checkpoint reload.
    for step in range(1, config["steps"]):
        record_step(step, next(iterator), resumed=False)
    checkpoint_path = output_dir / "checkpoint_before_resume.pth"
    save_checkpoint(checkpoint_path, model, optimizer, step, config, dataset, indices)
    saved = verify_checkpoint(checkpoint_path, model, optimizer, step, config, dataset, indices)
    check_equal(torch.get_rng_state(), saved["rng_cpu"], "saved_rng_cpu")
    check_equal(torch.cuda.get_rng_state_all(), saved["rng_cuda"], "saved_rng_cuda")
    del optimizer, model, iterator, loader

    model = build_model(progress=False).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer = make_optimizer(model, config)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    saved = verify_checkpoint(checkpoint_path, model, optimizer, step, config, dataset, indices)
    remaining = indices[saved["next_sample_offset"]:]
    resumed_loader = make_dataloader(Subset(dataset, remaining), batch_size=config["batch_size"],
                                     shuffle=False, num_workers=0)
    resumed_iterator = iter(resumed_loader)
    # Restore after constructors/iterator creation, which can consume random numbers.
    torch.set_rng_state(saved["rng_cpu"])
    torch.cuda.set_rng_state_all(saved["rng_cuda"])
    check_equal(torch.get_rng_state(), saved["rng_cpu"], "restored_rng_cpu")
    check_equal(torch.cuda.get_rng_state_all(), saved["rng_cuda"], "restored_rng_cuda")
    report["checkpoint_reload"] = {
        "restored_step": saved["global_step"], "next_sample_offset": saved["next_sample_offset"],
        "model_exact_match": True, "optimizer_exact_match": True, "rng_exact_match": True,
        "optimizer_state_device_verified": True,
    }
    record_step(saved["global_step"] + 1, next(resumed_iterator), resumed=True)
    actual_ids = [image_id for result in report["steps"] for image_id in result["image_ids"]]
    require(actual_ids == [dataset.image_ids[index] for index in indices], "Resume duplicated or skipped images.")
    final_path = output_dir / "checkpoint_final.pth"
    save_checkpoint(final_path, model, optimizer, config["steps"], config, dataset, indices)
    verify_checkpoint(final_path, model, optimizer, config["steps"], config, dataset, indices)
    report.update(
        status="passed", completed_steps=len(report["steps"]), completed_epochs=0,
        model=saved["model"], categories=saved["categories"],
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        checkpoints={"before_resume": str(checkpoint_path), "final": str(final_path),
                     "final_disk_roundtrip_exact_match": True,
                     "sizes_bytes": [checkpoint_path.stat().st_size, final_path.stat().st_size]},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/training_check.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/bdd100k_subset_10k")
    args = parser.parse_args()
    started = datetime.now().astimezone()
    output_dir = REPO_ROOT / "outputs/training_check" / started.strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": started.isoformat(), "steps": [],
              "purpose": "Bounded parameter-update/checkpoint check; not a full baseline experiment.",
              "performance_metrics": "Not measured: no validation evaluation.",
              "test_evaluation": "Not evaluated: train split only."}
    write_json(output_dir / "report.json", report)
    start_time = time.perf_counter()
    print(f"Training-check artifacts: {output_dir}", flush=True)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        write_json(output_dir / "config.json", config)
        report["code"] = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).splitlines(),
            "sha256": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                       for name in ("src/check_training.py", "src/model.py", "src/dataset.py")},
        }
        run_check(config, args.data_root, output_dir, report)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt: training check interrupted.")
        raise
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["elapsed_seconds"] = time.perf_counter() - start_time
        write_json(output_dir / "report.json", report)
    print(f"Training/checkpoint checks passed. Report: {output_dir / 'report.json'}")
    print("No full epoch, validation/test evaluation, AP calculation, or visualization was performed.")


if __name__ == "__main__":
    main()
