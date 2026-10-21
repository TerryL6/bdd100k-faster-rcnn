"""Build the pretrained Faster R-CNN baseline with the fixed BDD100K head."""

from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign


REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_HUB_DIR = REPO_ROOT / "outputs" / "torch" / "hub"
WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
NUM_CLASSES = 11  # Dataset category IDs 1-10, plus background 0.


class SquarePadTransform(GeneralizedRCNNTransform):
    """Keep aspect ratio, then pad the normalized image on the right/bottom.

    Image sizes remain the resized content sizes, excluding padding. This lets
    Torchvision resize targets and restore predictions in original coordinates.
    Zero padding is applied after normalization, as in the default transform.
    """

    def __init__(
        self,
        image_size: int,
        image_mean: list[float],
        image_std: list[float],
        size_divisible: int = 32,
    ) -> None:
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size <= 0
            or image_size % size_divisible != 0
        ):
            raise ValueError("image_size must be a positive integer divisible by size_divisible")
        super().__init__(
            min_size=image_size,
            max_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            size_divisible=size_divisible,
        )
        self.image_size = image_size

    def batch_images(self, images: list[torch.Tensor], size_divisible: int = 32) -> torch.Tensor:
        """Use a fixed square canvas while retaining top-left image alignment."""
        batch = images[0].new_zeros((len(images), images[0].shape[0], self.image_size, self.image_size))
        for index, image in enumerate(images):
            height, width = image.shape[-2:]
            if height > self.image_size or width > self.image_size:
                raise ValueError("Resized image exceeds the square canvas")
            batch[index, :, :height, :width].copy_(image)
        return batch


class SquareCanvasRoIAlign(MultiScaleRoIAlign):
    """Infer FPN strides from the padded canvas, not the smaller content size.

    RPN/ROI clipping and transform postprocessing still use content sizes.
    This pooler has no learned parameters; all pooling defaults are preserved.
    """

    def __init__(self, image_size: int, pooler: MultiScaleRoIAlign) -> None:
        super().__init__(
            pooler.featmap_names, pooler.output_size, pooler.sampling_ratio,
            canonical_scale=pooler.canonical_scale, canonical_level=pooler.canonical_level,
        )
        self.image_size = image_size

    def forward(self, x, boxes, image_shapes):
        canvas_shapes = [(self.image_size, self.image_size)] * len(image_shapes)
        return super().forward(x, boxes, canvas_shapes)


def get_transform_settings(transform: GeneralizedRCNNTransform) -> dict:
    """Return JSON-compatible preprocessing settings for reports/checkpoints."""
    square = isinstance(transform, SquarePadTransform)
    return {
        "image_mean": list(transform.image_mean),
        "image_std": list(transform.image_std),
        "min_size": list(transform.min_size),
        "max_size": transform.max_size,
        "size_divisible": transform.size_divisible,
        "fixed_size": list(transform.fixed_size) if transform.fixed_size is not None else None,
        "resize_policy": "aspect_ratio_square_pad" if square else "torchvision_default",
        "canvas_size": [transform.image_size, transform.image_size] if square else None,
        "interpolation": "bilinear",
        "align_corners": False,
        "recompute_scale_factor": True,
        "allow_upscale": True,
        "padding_alignment": "top_left",
        "padding_sides": ["right", "bottom"],
        "padding_value": 0.0,
        "padding_value_space": "normalized",
        "box_offset_xy": [0, 0],
        "box_scale_reference": "actual_resized_content_size",
        "roi_pool_scale_reference": "canvas_size" if square else "resized_image_sizes",
    }


def build_model(*, progress: bool = True, image_size: int | None = None) -> FasterRCNN:
    """Load official detection weights, then initialize a new 11-class head.

    Torch Hub's process-wide weight cache is placed inside ignored outputs/.
    The first call downloads COCO_V1 if it is not already cached there.
    Normalization and resizing remain inside the detector. With image_size,
    keep aspect ratio and pad to that square size; None retains Torchvision's
    historical defaults. The Dataset supplies original-size RGB in [0, 1].
    """
    torch.hub.set_dir(str(TORCH_HUB_DIR))
    model = fasterrcnn_resnet50_fpn(
        weights=WEIGHTS,
        weights_backbone=None,
        progress=progress,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    if image_size is not None:
        model.transform = SquarePadTransform(
            image_size,
            image_mean=model.transform.image_mean,
            image_std=model.transform.image_std,
            size_divisible=model.transform.size_divisible,
        )
        model.roi_heads.box_roi_pool = SquareCanvasRoIAlign(image_size, model.roi_heads.box_roi_pool)
    return model
