"""Check a small set of fixed train/validation samples without using a model."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Subset

from .dataset import BDD100KDataset, make_dataloader


REPO_ROOT = Path(__file__).resolve().parents[1]


def sample_indices(length: int, requested: int) -> list[int]:
    """Include the first and last image, with evenly spaced samples between."""
    count = min(length, requested)
    if count == 1:
        return [0]
    return [i * (length - 1) // (count - 1) for i in range(count)]


def save_preview(image, target, category_names, path):
    """Draw a copy of the original image and write only under local outputs."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], max(0, box[1] - 12)), category_names[label], fill="yellow")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def check_sample(dataset, index, preview_path=None):
    image, target = dataset[index]
    image_id = dataset.image_ids[index]
    record = dataset.coco.imgs[image_id]
    annotations = dataset.coco.imgToAnns[image_id]
    count = len(annotations)

    assert image.dtype == torch.float32, "Image must use float32."
    assert image.shape == (3, record["height"], record["width"]), "Wrong image shape."
    assert torch.isfinite(image).all(), "Image contains non-finite pixels."
    assert 0 <= image.min().item() <= image.max().item() <= 1, "Pixels must be in [0, 1]."
    expected_types = {
        "boxes": (torch.float32, (count, 4)),
        "labels": (torch.int64, (count,)),
        "image_id": (torch.int64, (1,)),
        "area": (torch.float32, (count,)),
        "iscrowd": (torch.int64, (count,)),
    }
    for key, (dtype, shape) in expected_types.items():
        assert key in target, f"Missing target field: {key}."
        assert target[key].dtype == dtype, f"Wrong dtype for {key}."
        assert target[key].shape == shape, f"Wrong shape for {key}."
    assert target["image_id"].item() == image_id, "Original COCO image ID was lost."

    # Python arithmetic on source xywh values is independent of tensor conversion.
    expected_boxes = []
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        expected_boxes.append([x, y, x + width, y + height])
    expected = {
        "boxes": torch.tensor(expected_boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor([a["category_id"] for a in annotations], dtype=torch.int64),
        "area": torch.tensor([a["area"] for a in annotations], dtype=torch.float32),
        "iscrowd": torch.tensor([a["iscrowd"] for a in annotations], dtype=torch.int64),
    }
    for key, value in expected.items():
        if key == "boxes":
            torch.testing.assert_close(target[key], value, rtol=1e-6, atol=1e-4)
        else:
            torch.testing.assert_close(target[key], value, rtol=0, atol=0)
    assert set(target["labels"].tolist()) <= set(dataset.category_names), "Unknown category ID."
    boxes = target["boxes"]
    assert torch.isfinite(boxes).all(), "Boxes contain non-finite coordinates."
    assert (boxes[:, 2:] > boxes[:, :2]).all(), "Boxes must have positive width and height."
    assert (boxes[:, :2] >= 0).all(), "A box starts outside the image."
    assert (boxes[:, 2] <= record["width"] + 1e-4).all(), "A box exceeds image width."
    assert (boxes[:, 3] <= record["height"] + 1e-4).all(), "A box exceeds image height."

    with Image.open(dataset.image_dir / record["file_name"]) as source:
        rgb = source.convert("RGB")
        pixels = np.array(rgb, dtype=np.uint8, copy=True)
        expected_image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        torch.testing.assert_close(image, expected_image, rtol=0, atol=0)
        if preview_path is not None:
            save_preview(rgb, target, dataset.category_names, preview_path)
    print(f"  index={index}, image_id={image_id}, file={record['file_name']}, boxes={count}: PASS")
    return target


def check_batch(images, targets, expected_ids, expected_targets):
    assert isinstance(images, list) and isinstance(targets, list), "Batch must contain two lists."
    assert len(images) == len(targets) == len(expected_ids), "Wrong batch length."
    for image, target, image_id in zip(images, targets, expected_ids):
        assert target["image_id"].item() == image_id, "Batch order or image ID changed."
        assert image.dtype == torch.float32 and image.ndim == 3, "Wrong batched image tensor."
        for key, expected in expected_targets[image_id].items():
            torch.testing.assert_close(target[key], expected, rtol=0, atol=0)


def check_split(data_root, split, samples, batch_size, num_workers, save_previews):
    dataset = BDD100KDataset(data_root, split)
    assert len(dataset) > 0, "Dataset is empty."
    indices = sample_indices(len(dataset), samples)
    print(f"{split}: {len(dataset)} images; checking {len(indices)} evenly spaced samples.")
    expected_targets = {}
    for position, index in enumerate(indices):
        image_id = dataset.image_ids[index]
        preview_path = None
        if save_previews and position < 2:
            preview_path = REPO_ROOT / "outputs" / "data_check" / f"{split}_{image_id}.jpg"
        expected_targets[image_id] = check_sample(dataset, index, preview_path)

    subset = Subset(dataset, indices)
    loader = make_dataloader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    selected_ids = [dataset.image_ids[index] for index in indices]
    offset = 0
    batch_sizes = []
    for images, targets in loader:
        expected_ids = selected_ids[offset:offset + batch_size]
        check_batch(images, targets, expected_ids, expected_targets)
        batch_sizes.append(len(images))
        offset += len(images)
    assert offset == len(indices), "DataLoader dropped or duplicated samples."

    # Even the default four samples/two-image batches exercise a partial batch.
    if batch_size > 1 and all(size == batch_size for size in batch_sizes):
        partial_loader = make_dataloader(
            Subset(dataset, indices[-1:]), batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )
        partial_batches = list(partial_loader)
        assert len(partial_batches) == 1, "Wrong number of partial batches."
        images, targets = partial_batches[0]
        check_batch(images, targets, selected_ids[-1:], expected_targets)
        print("  Additional one-sample final batch: PASS")
    elif batch_size > 1:
        assert batch_sizes[-1] < batch_size, "Final partial batch was lost."
        print(f"  Final partial batch ({batch_sizes[-1]} samples): PASS")

    object_counts = [len(expected_targets[image_id]["boxes"]) for image_id in selected_ids]
    print(f"  Batch sizes={batch_sizes}; per-image object counts={object_counts}: PASS")
    if len(set(object_counts)) > 1:
        print("  Variable object counts preserved across batch loading: PASS")
    else:
        print("  Selected samples have equal object counts; increase --samples-per-split to inspect variation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "bdd100k_subset_10k")
    parser.add_argument("--samples-per-split", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-previews", action="store_true")
    args = parser.parse_args()
    if args.samples_per_split < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("Samples and batch size must be positive; num-workers must be nonnegative.")
    for split in ("train", "val"):
        check_split(args.data_root, split, args.samples_per_split, args.batch_size,
                    args.num_workers, args.save_previews)
    print("Data loading checks passed. No model, training, or evaluation was run.")
    if args.save_previews:
        print(f"Previews: {REPO_ROOT / 'outputs' / 'data_check'}")


if __name__ == "__main__":
    main()
