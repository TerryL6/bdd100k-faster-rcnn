"""Synthetic CPU geometry checks without detector construction or training."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from src.model import (
    NUM_CLASSES, WEIGHTS, SquareCanvasRoIAlign, SquarePadTransform,
    build_model, get_transform_settings,
)


class SquarePadTransformTests(unittest.TestCase):
    def make_transform(self):
        return SquarePadTransform(
            640,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    def test_landscape_boxes_padding_and_original_targets(self):
        transform = self.make_transform()
        image = torch.ones((3, 720, 1280), dtype=torch.float32)
        boxes = torch.tensor([[128.0, 72.0, 1152.0, 648.0]])
        target = {"boxes": boxes.clone(), "labels": torch.tensor([3])}
        image_list, targets = transform([image], [target])

        self.assertEqual(tuple(image_list.tensors.shape), (1, 3, 640, 640))
        self.assertEqual(image_list.image_sizes, [(360, 640)])
        torch.testing.assert_close(targets[0]["boxes"], boxes * 0.5)
        torch.testing.assert_close(target["boxes"], boxes, rtol=0, atol=0)
        torch.testing.assert_close(image, torch.ones_like(image), rtol=0, atol=0)
        normalized = transform.normalize(image)[:, ::2, ::2]
        torch.testing.assert_close(image_list.tensors[0, :, :360, :640], normalized)
        self.assertEqual(torch.count_nonzero(image_list.tensors[0, :, 360:, :]).item(), 0)

    def test_portrait_empty_boxes_and_right_padding(self):
        transform = self.make_transform()
        image = torch.ones((3, 1280, 720), dtype=torch.float32)
        target = {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)}
        image_list, targets = transform([image], [target])

        self.assertEqual(tuple(image_list.tensors.shape), (1, 3, 640, 640))
        self.assertEqual(image_list.image_sizes, [(640, 360)])
        self.assertEqual(tuple(targets[0]["boxes"].shape), (0, 4))
        self.assertEqual(torch.count_nonzero(image_list.tensors[0, :, :, 360:]).item(), 0)
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))

    def test_mixed_batch_and_box_roundtrip_use_content_sizes(self):
        transform = self.make_transform().eval()
        original_sizes = [(701, 1279), (1291, 719)]
        images = [torch.zeros((3, height, width)) for height, width in original_sizes]
        targets = [
            {"boxes": torch.tensor([[17.5, 23.25, width - 9.5, height - 12.25]])}
            for height, width in original_sizes
        ]
        image_list, resized_targets = transform(images, targets)
        self.assertEqual(tuple(image_list.tensors.shape), (2, 3, 640, 640))

        for original_size, content_size, target, resized in zip(
            original_sizes, image_list.image_sizes, targets, resized_targets
        ):
            height, width = original_size
            resized_height, resized_width = content_size
            self.assertLessEqual(max(content_size), 640)
            self.assertLessEqual(abs(resized_height / resized_width - height / width), 0.005)
            scale = torch.tensor([resized_width / width, resized_height / height] * 2)
            torch.testing.assert_close(resized["boxes"], target["boxes"] * scale)

        predictions = [{"boxes": target["boxes"].clone()} for target in resized_targets]
        restored = transform.postprocess(predictions, image_list.image_sizes, original_sizes)
        for result, target in zip(restored, targets):
            torch.testing.assert_close(result["boxes"], target["boxes"], rtol=1e-6, atol=1e-4)

    def test_same_size_image_and_no_targets(self):
        transform = self.make_transform().eval()
        image = torch.zeros((3, 640, 640))
        image_list, targets = transform([image])
        self.assertIsNone(targets)
        self.assertEqual(image_list.image_sizes, [(640, 640)])
        torch.testing.assert_close(image_list.tensors[0], transform.normalize(image), rtol=0, atol=0)

    def test_training_and_validation_have_the_same_geometry(self):
        transform = self.make_transform()
        image = torch.zeros((3, 127, 229))
        target = {"boxes": torch.tensor([[1.25, 2.5, 221.75, 125.0]])}
        training_images, training_targets = transform([image], [target])
        transform.eval()
        validation_images, validation_targets = transform([image], [target])
        self.assertEqual(training_images.image_sizes, validation_images.image_sizes)
        self.assertEqual(tuple(training_images.tensors.shape), (1, 3, 640, 640))
        torch.testing.assert_close(training_images.tensors, validation_images.tensors, rtol=0, atol=0)
        torch.testing.assert_close(training_targets[0]["boxes"], validation_targets[0]["boxes"], rtol=0, atol=0)

    def test_settings_are_json_compatible_and_distinguish_default(self):
        square = get_transform_settings(self.make_transform())
        default = get_transform_settings(
            GeneralizedRCNNTransform(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        self.assertEqual(square["min_size"], [640])
        self.assertEqual(square["max_size"], 640)
        self.assertIsNone(square["fixed_size"])
        self.assertEqual(square["canvas_size"], [640, 640])
        self.assertEqual(square["resize_policy"], "aspect_ratio_square_pad")
        self.assertEqual(square["padding_alignment"], "top_left")
        self.assertEqual(square["padding_value"], 0.0)
        self.assertEqual(square["padding_value_space"], "normalized")
        self.assertEqual(square["box_offset_xy"], [0, 0])
        self.assertEqual(square["roi_pool_scale_reference"], "canvas_size")
        self.assertEqual(default["resize_policy"], "torchvision_default")
        self.assertIsNone(default["canvas_size"])
        self.assertEqual(default["min_size"], [800])
        self.assertEqual(default["max_size"], 1333)
        self.assertEqual(json.loads(json.dumps(square)), square)
        self.assertEqual(json.loads(json.dumps(default)), default)

    def test_invalid_size_is_rejected(self):
        for size in [0, -640, 639, 640.0, True, None]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                SquarePadTransform(size, [0.0] * 3, [1.0] * 3)


class SquareCanvasPoolTests(unittest.TestCase):
    def test_fpn_scales_and_pooled_coordinates_use_canvas_including_cached_calls(self):
        original = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2)
        pooler = SquareCanvasRoIAlign(640, original)
        features = {
            str(level): torch.arange(size * size, dtype=torch.float32).reshape(1, 1, size, size) / size
            for level, size in enumerate((160, 80, 40, 20))
        }
        boxes = [torch.tensor([[8., 12., 60., 80.], [32., 8., 332., 350.]])]
        reference = original(features, boxes, [(640, 640)])
        # Landscape content is shorter than the canvas: inferring from 360 would double every scale.
        for content_size in [(360, 640), (640, 360), (640, 640)]:
            actual = pooler(features, boxes, [content_size])
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            self.assertEqual(pooler.scales, [0.25, 0.125, 0.0625, 0.03125])
            self.assertEqual((pooler.map_levels.k_min, pooler.map_levels.k_max), (2, 5))
        self.assertEqual(pooler.state_dict(), {})

    def test_factory_keeps_official_weights_and_pooling_defaults_with_a_new_head(self):
        transform = GeneralizedRCNNTransform(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        pooler = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2)
        model = SimpleNamespace(
            transform=transform,
            roi_heads=SimpleNamespace(
                box_predictor=SimpleNamespace(cls_score=SimpleNamespace(in_features=8)),
                box_roi_pool=pooler,
                score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
            ),
        )
        # Mock the detector factory: no Faster R-CNN is constructed or weights loaded.
        with patch("src.model.fasterrcnn_resnet50_fpn", return_value=model) as factory, \
                patch("src.model.torch.hub.set_dir"):
            result = build_model(progress=False, image_size=640)
        factory.assert_called_once_with(weights=WEIGHTS, weights_backbone=None, progress=False)
        self.assertIs(result, model)
        self.assertEqual(result.roi_heads.box_predictor.cls_score.out_features, NUM_CLASSES)
        self.assertEqual(result.roi_heads.box_predictor.bbox_pred.out_features, NUM_CLASSES * 4)
        self.assertIsInstance(result.transform, SquarePadTransform)
        self.assertIsInstance(result.roi_heads.box_roi_pool, SquareCanvasRoIAlign)
        self.assertEqual(result.transform.image_mean, transform.image_mean)
        self.assertEqual(result.transform.image_std, transform.image_std)
        for name in ("featmap_names", "output_size", "sampling_ratio", "canonical_scale", "canonical_level"):
            self.assertEqual(getattr(result.roi_heads.box_roi_pool, name), getattr(pooler, name))
        self.assertEqual((result.roi_heads.score_thresh, result.roi_heads.nms_thresh,
                          result.roi_heads.detections_per_img), (0.05, 0.5, 100))


if __name__ == "__main__":
    unittest.main()
