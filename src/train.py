"""Train Faster R-CNN: edit the settings below, then run python -m src.train."""

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .dataset import BDD100KDataset, make_dataloader
from .evaluate import evaluate
from .model import REPO_ROOT, TORCH_HUB_DIR, WEIGHTS, build_model, get_transform_settings


# Provisional starting values, not tuned hyperparameters.
DATA_ROOT = REPO_ROOT / "data" / "bdd100k_subset_10k"
EPOCHS = 10  # Observation window; convergence is not established.
IMAGE_SIZE = 640  # Aspect-preserving resize, then right/bottom padding inside the model.
BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SEED = 42
DEVICE = "cuda:0"
PRINT_EVERY = 50


def train_one_epoch(model, loader, optimizer, device, *, epoch=1, print_every=PRINT_EVERY):
    """Read each batch, calculate loss, backpropagate, and update parameters."""
    model.train()
    loss_sums = {}
    image_count = 0
    steps = 0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        total_loss = sum(losses.values())
        if not torch.isfinite(total_loss).item():
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}.")
        total_loss.backward()
        optimizer.step()

        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value.item() * len(images)
        image_count += len(images)
        steps += 1
        if print_every and (batch_index % print_every == 0 or batch_index == len(loader)):
            print(f"epoch={epoch} batch={batch_index}/{len(loader)} "
                  f"loss={total_loss.item():.4f} lr={optimizer.param_groups[0]['lr']}", flush=True)

    if image_count == 0 or image_count != len(loader.dataset):
        raise RuntimeError("Training must visit every loader sample, including the final partial batch.")
    mean_losses = {name: value / image_count for name, value in loss_sums.items()}
    return {"losses": mean_losses, "total_loss": sum(mean_losses.values()),
            "steps": steps, "images": image_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Continue from latest.pth after increasing EPOCHS above.")
    args = parser.parse_args()
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the environment or change DEVICE above.")
    if not (TORCH_HUB_DIR / "checkpoints" / Path(WEIGHTS.url).name).is_file():
        raise FileNotFoundError("The official pretrained weights must already exist in outputs/torch/hub/checkpoints/.")
    seed_everything(SEED)

    # 1. Read the existing fixed splits; the detector handles normalization and resizing.
    train_dataset = BDD100KDataset(DATA_ROOT, "train")
    val_dataset = BDD100KDataset(DATA_ROOT, "val")
    train_loader = make_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = make_dataloader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Load the pretrained detector and create its SGD optimizer.
    model = build_model(progress=False, image_size=IMAGE_SIZE).to(device)
    settings = {"seed": SEED, "batch_size": BATCH_SIZE, "val_batch_size": VAL_BATCH_SIZE,
                "device": DEVICE, "num_workers": 0, "transform": get_transform_settings(model.transform),
                "optimizer": {"lr": LEARNING_RATE, "momentum": MOMENTUM,
                                               "weight_decay": WEIGHT_DECAY}}
    optimizer = make_optimizer(model, settings["optimizer"])
    start_epoch, global_step, best_map, best_epoch = 1, 0, -1.0, 0
    history = []

    # Optional continuation uses the same run directory and its existing best model.
    if args.resume:
        resume_path = args.resume.resolve()
        output_dir = resume_path.parent
        if resume_path.name != "latest.pth" or not output_dir.is_relative_to(REPO_ROOT / "outputs"):
            raise ValueError("Resume from latest.pth inside this project's outputs directory.")
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved.get("kind") != "bdd100k_simple_training_v1":
            raise ValueError("This entry point resumes only checkpoints from the simplified trainer.")
        categories = train_dataset.coco.dataset["categories"]
        validate_resume_settings(saved, settings, categories)
        start_epoch = saved["epoch"] + 1
        if EPOCHS < start_epoch or saved["global_step"] != saved["epoch"] * len(train_loader):
            raise ValueError("Increase EPOCHS and keep the same training dataset on resume.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        global_step, best_map, best_epoch = saved["global_step"], saved["best_mAP"], saved["best_epoch"]
        history = saved["history"]
        if best_epoch == saved["epoch"]:
            save_checkpoint(output_dir / "best.pth", saved)
        elif not (output_dir / "best.pth").is_file():
            raise FileNotFoundError("Keep the run's earlier best.pth alongside latest.pth.")
        restore_rng_state(saved["rng_state"])
        del saved
    else:
        output_dir = REPO_ROOT / "outputs" / "training" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=False)
    write_history(output_dir / "history.csv", history)
    print(f"Run: {output_dir}\nTrain images={len(train_dataset)}, validation images={len(val_dataset)}", flush=True)

    # 3. Train a complete epoch, evaluate validation, and save the results.
    for epoch in range(start_epoch, EPOCHS + 1):
        training = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch)
        global_step += training["steps"]
        validation, predictions = evaluate(model, val_loader, val_dataset, device)
        metrics = validation["metrics"]
        score = metrics["bbox_mAP"]
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RuntimeError("Validation mAP is not evaluable; no checkpoint can be selected.")
        improved = score > best_map
        if improved:
            best_map, best_epoch = score, epoch
        history.append({"epoch": epoch, "global_step": global_step, **training["losses"],
                        "total_loss": training["total_loss"], "lr": optimizer.param_groups[0]["lr"],
                        "mAP": score, "AP50": metrics["bbox_AP50"], "AP75": metrics["bbox_AP75"]})
        checkpoint = {"kind": "bdd100k_simple_training_v1", "epoch": epoch, "global_step": global_step,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "best_mAP": best_map, "best_epoch": best_epoch, "settings": settings,
                      "categories": train_dataset.coco.dataset["categories"], "rng_state": capture_rng_state(),
                      "history": history}
        save_checkpoint(output_dir / "latest.pth", checkpoint)
        if improved:
            save_checkpoint(output_dir / "best.pth", checkpoint)
        write_history(output_dir / "history.csv", history)
        validation["epoch"] = epoch
        for name, value in (("validation.json", validation), ("predictions.json", predictions)):
            (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"epoch={epoch} mean_loss={training['total_loss']:.4f} "
              f"mAP={score:.4f} AP50={metrics['bbox_AP50']:.4f} AP75={metrics['bbox_AP75']:.4f} "
              f"best_epoch={best_epoch}", flush=True)


# Small support functions are kept below the training flow.
def validate_resume_settings(saved, settings, categories):
    if saved["settings"].get("transform") != settings["transform"]:
        raise ValueError("Checkpoint preprocessing differs or is missing. Start a fresh run without --resume.")
    if saved["settings"] != settings or saved["categories"] != categories:
        raise ValueError("Keep training settings and categories unchanged on resume; only EPOCHS may increase.")


def make_optimizer(model, settings):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], **settings)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["cpu"])
    if len(state["cuda"]) != torch.cuda.device_count():
        raise RuntimeError("Resume requires the same CUDA device count.")
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_history(path, history):
    if history:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    main()
