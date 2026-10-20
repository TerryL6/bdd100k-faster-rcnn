"""Read the fixed, cleaned BDD100K COCO training and validation splits."""

import json
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
from torchvision.transforms.functional import pil_to_tensor


class BDD100KDataset(Dataset):
    """Return an RGB float32 image and a detection target with original COCO IDs."""

    def __init__(self, data_root: str | Path, split: str):
        if split not in ("train", "val"):
            raise ValueError("Only the fixed 'train' and 'val' splits are supported.")

        self.data_root = Path(data_root)
        self.split = split
        self.image_dir = self.data_root / "images" / split
        self.coco = COCO(str(self.data_root / "annotations" / f"{split}.json"))
        self.image_ids = self.coco.getImgIds()
        self.class_mapping = json.loads(
            (self.data_root / "class_mapping.json").read_text(encoding="utf-8")
        )
        expected_categories = [
            (entry["coco_id"], entry["name"]) for entry in self.class_mapping
        ]
        actual_categories = [
            (entry["id"], entry["name"]) for entry in self.coco.dataset["categories"]
        ]
        if actual_categories != expected_categories:
            raise ValueError("COCO categories do not match class_mapping.json in order.")
        self.category_names = dict(expected_categories)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        record = self.coco.imgs[image_id]
        with Image.open(self.image_dir / record["file_name"]) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"Image dimensions differ from COCO record: {image_id}")
            image_tensor = pil_to_tensor(image.convert("RGB")).to(torch.float32) / 255.0

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        boxes_xywh = torch.tensor(
            [annotation["bbox"] for annotation in annotations], dtype=torch.float32
        ).reshape(-1, 4)
        target = {
            "boxes": box_convert(boxes_xywh, in_fmt="xywh", out_fmt="xyxy"),
            "labels": torch.tensor(
                [annotation["category_id"] for annotation in annotations], dtype=torch.int64
            ),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(
                [annotation["area"] for annotation in annotations], dtype=torch.float32
            ),
            "iscrowd": torch.tensor(
                [annotation["iscrowd"] for annotation in annotations], dtype=torch.int64
            ),
        }
        return image_tensor, target


def collate_fn(batch):
    """Keep images and variable-length detection targets in separate lists."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a CPU loader without resizing, augmentation, or dropping samples."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
