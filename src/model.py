"""Build the pretrained Faster R-CNN baseline with the fixed BDD100K head."""

from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_HUB_DIR = REPO_ROOT / "outputs" / "torch" / "hub"
WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
NUM_CLASSES = 11  # Dataset category IDs 1-10, plus background 0.


def build_model(*, progress: bool = True) -> FasterRCNN:
    """Load official detection weights, then initialize a new 11-class head.

    Torch Hub's process-wide weight cache is placed inside ignored outputs/.
    The first call downloads COCO_V1 if it is not already cached there.
    Normalization and resizing remain inside the detector, using Torchvision
    defaults. The Dataset must supply original-size RGB tensors in [0, 1].
    """
    torch.hub.set_dir(str(TORCH_HUB_DIR))
    model = fasterrcnn_resnet50_fpn(
        weights=WEIGHTS,
        weights_backbone=None,
        progress=progress,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model
