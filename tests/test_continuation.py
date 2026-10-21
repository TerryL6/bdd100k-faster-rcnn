"""Exercise full continuation with synthetic CPU data, without a real detector."""

import copy
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from src import prepare_continuation, train
from src.dataset import make_dataloader


CATEGORIES = [{"id": 1, "name": "synthetic"}]
TRANSFORM = {"type": "synthetic_cpu_transform"}
OPTIMIZER = {"lr": 0.005, "momentum": 0.9, "weight_decay": 0.0005}
INITIAL_SCORES = [0.10, 0.15, 0.20, 0.25, 0.23]


class SyntheticDataset(Dataset):
    """Use five in-memory samples so the last training batch is incomplete."""

    def __init__(self, data_root=None, split="train"):
        if split not in ("train", "val"):
            raise AssertionError("Synthetic continuation must never request a test split.")
        self.split = split
        self.coco = SimpleNamespace(dataset={"categories": copy.deepcopy(CATEGORIES)})

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return torch.full((3, 1, 1), (index + 1) / 10), {
            "image_id": torch.tensor([101 + index], dtype=torch.int64)
        }


class TinyDetector(torch.nn.Module):
    """Make synthetic gradients depend on every host random generator."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(2))
        self.transform = torch.nn.Identity()
        self.image_order = []

    def forward(self, images, targets):
        self.image_order.extend(target["image_id"].item() for target in targets)
        jitter = random.random() + float(np.random.random()) + torch.rand(()).item()
        target = torch.stack([image.mean() for image in images]).mean() + jitter / 10
        loss = ((self.weight - target) ** 2).sum()
        return {"loss_classifier": loss, "loss_box_reg": loss * 0.5}


def synthetic_evaluation(score):
    """Consume RNG like a stochastic consumer, without inference or scoring."""
    random.random()
    np.random.random()
    torch.rand(())
    return {"metrics": {"bbox_mAP": score, "bbox_AP50": score + 0.1,
                        "bbox_AP75": score - 0.05}, "synthetic": True}, []


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, "device_count", return_value=0))
        self.original_rng = train.capture_rng_state()
        output_root = Path(__file__).resolve().parents[1] / "outputs"
        output_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_continuation_", dir=output_root)
        self.root = Path(self.temporary.name)
        self.original_dir = self.root / "outputs" / "training" / "original"
        self.branch_dir = self.root / "outputs" / "training" / "continuation"
        self.original_dir.mkdir(parents=True)
        self.branch_dir.mkdir(parents=True)
        self.settings = {"seed": 42, "batch_size": 2, "val_batch_size": 2,
                         "device": "cpu", "num_workers": 0,
                         "transform": copy.deepcopy(TRANSFORM), "optimizer": OPTIMIZER.copy()}
        self.continuation = {"parent_run": "outputs/training/original",
                             "source_checkpoint": "outputs/training/original/latest.pth",
                             "completed_epoch": 5, "global_step": 15}
        self.best_source = {"checkpoint": "outputs/training/original/best.pth",
                            "epoch": 4, "global_step": 12, "mAP": 0.25}
        self.latest, self.best = self.make_source_checkpoints()
        for directory in (self.original_dir, self.branch_dir):
            train.save_checkpoint(directory / "latest.pth", self.latest)
            train.save_checkpoint(directory / "best.pth", self.best)
        self.original_files = {name: (self.original_dir / name).read_bytes()
                               for name in ("latest.pth", "best.pth")}

    def tearDown(self):
        train.restore_rng_state(self.original_rng)
        self.stack.close()
        self.temporary.cleanup()

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(first, second)

    def training_components(self):
        model = TinyDetector()
        optimizer = train.make_optimizer(model, OPTIMIZER)
        loader = make_dataloader(SyntheticDataset(), batch_size=2, shuffle=True, num_workers=0)
        return model, optimizer, loader

    def make_source_checkpoints(self):
        train.seed_everything(42)
        model, optimizer, loader = self.training_components()
        history, best = [], None
        for epoch, score in enumerate(INITIAL_SCORES, start=1):
            with redirect_stdout(io.StringIO()):
                summary = train.train_one_epoch(model, loader, optimizer, "cpu", epoch=epoch,
                                                print_every=0)
            synthetic_evaluation(score)
            history.append({"epoch": epoch, "global_step": epoch * len(loader),
                            **summary["losses"], "total_loss": summary["total_loss"],
                            "lr": OPTIMIZER["lr"], "mAP": score,
                            "AP50": score + 0.1, "AP75": score - 0.05})
            best_row = max(history, key=lambda row: row["mAP"])
            checkpoint = {"kind": "bdd100k_simple_training_v1", "epoch": epoch,
                          "global_step": epoch * len(loader),
                          "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "best_mAP": best_row["mAP"], "best_epoch": best_row["epoch"],
                          "settings": self.settings, "categories": CATEGORIES,
                          "rng_state": train.capture_rng_state(), "history": history,
                          "continuation": self.continuation, "best_source": self.best_source}
            if epoch == 4:
                best = copy.deepcopy(checkpoint)
        return copy.deepcopy(checkpoint), best

    def expected_continuation(self, scores):
        train.seed_everything(999)
        model, optimizer, loader = self.training_components()
        model.load_state_dict(self.latest["model_state_dict"])
        optimizer.load_state_dict(copy.deepcopy(self.latest["optimizer_state_dict"]))
        train.restore_rng_state(self.latest["rng_state"])
        for epoch, score in enumerate(scores, start=6):
            with redirect_stdout(io.StringIO()):
                train.train_one_epoch(model, loader, optimizer, "cpu", epoch=epoch, print_every=0)
            synthetic_evaluation(score)
        return model, copy.deepcopy(optimizer.state_dict()), train.capture_rng_state()

    def run_main(self, scores):
        model = TinyDetector()
        cache = self.root / "cache" / "checkpoints" / "synthetic.pth"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"Synthetic presence check; never load pretrained weights.")
        results = iter(scores)
        observed_boundary = []
        original_train_one_epoch = train.train_one_epoch

        def observed_training(current_model, loader, optimizer, device, **kwargs):
            if not observed_boundary:
                observed_boundary.append({"model": copy.deepcopy(current_model.state_dict()),
                                          "optimizer": copy.deepcopy(optimizer.state_dict()),
                                          "rng": train.capture_rng_state(),
                                          "epoch": kwargs["epoch"]})
            return original_train_one_epoch(current_model, loader, optimizer, device,
                                            epoch=kwargs["epoch"], print_every=0)

        with ExitStack() as local:
            for name, value in (("REPO_ROOT", self.root), ("DEVICE", "cpu"), ("EPOCHS", 10),
                                ("TORCH_HUB_DIR", cache.parent.parent),
                                ("WEIGHTS", SimpleNamespace(url="https://invalid/synthetic.pth")),
                                ("DATA_ROOT", self.root / "unread_synthetic_data")):
                local.enter_context(patch.object(train, name, value))
            factory = local.enter_context(patch.object(train, "BDD100KDataset", side_effect=SyntheticDataset))
            local.enter_context(patch.object(train, "build_model", return_value=model))
            local.enter_context(patch.object(train, "get_transform_settings", return_value=TRANSFORM))
            evaluator = local.enter_context(patch.object(
                train, "evaluate", side_effect=lambda *args: synthetic_evaluation(next(results))))
            local.enter_context(patch.object(train, "train_one_epoch", side_effect=observed_training))
            local.enter_context(patch.object(sys, "argv", ["synthetic", "--resume",
                                                           str(self.branch_dir / "latest.pth")]))
            with redirect_stdout(io.StringIO()):
                train.main()
        self.assertEqual([call.args[1] for call in factory.call_args_list], ["train", "val"])
        self.assertEqual(evaluator.call_count, 5)
        self.assertEqual(observed_boundary[0]["epoch"], 6)
        self.assert_state_equal(observed_boundary[0]["model"], self.latest["model_state_dict"])
        self.assert_state_equal(observed_boundary[0]["optimizer"], self.latest["optimizer_state_dict"])
        self.assert_state_equal(observed_boundary[0]["rng"], self.latest["rng_state"])
        for name, content in self.original_files.items():
            self.assertEqual((self.original_dir / name).read_bytes(), content)
        return model, torch.load(self.branch_dir / "latest.pth", map_location="cpu", weights_only=True)

    def test_main_continues_epoch_six_to_ten_and_preserves_inherited_best(self):
        scores = [0.25, 0.24, 0.23, 0.22, 0.21]
        expected_model, expected_optimizer, expected_rng = self.expected_continuation(scores)
        inherited_best_bytes = (self.branch_dir / "best.pth").read_bytes()
        model, latest = self.run_main(scores)
        self.assertEqual(latest["epoch"], 10)
        self.assertEqual(latest["global_step"], 30)
        self.assertEqual([row["epoch"] for row in latest["history"]], list(range(1, 11)))
        self.assertEqual([row["global_step"] for row in latest["history"]], list(range(3, 31, 3)))
        self.assertEqual(latest["history"][:5], self.latest["history"])
        self.assertEqual(latest["best_epoch"], 4)
        self.assertEqual(latest["best_mAP"], 0.25)
        self.assertEqual(latest["continuation"], self.continuation)
        self.assertEqual(latest["best_source"], self.best_source)
        self.assertEqual((self.branch_dir / "best.pth").read_bytes(), inherited_best_bytes)
        self.assert_state_equal(latest["model_state_dict"], expected_model.state_dict())
        self.assert_state_equal(latest["optimizer_state_dict"], expected_optimizer)
        self.assert_state_equal(latest["rng_state"], expected_rng)
        self.assertEqual(model.image_order, expected_model.image_order)
        report = json.loads((self.branch_dir / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["epoch"], 10)
        self.assertEqual(report["best_model"], self.best_source)
        history_csv = (self.branch_dir / "history.csv").read_text(encoding="utf-8")
        self.assertEqual(len(history_csv.splitlines()), 11)

    def test_strict_improvement_updates_best_checkpoint_and_provenance(self):
        _, latest = self.run_main([0.25, 0.24, 0.30, 0.30, 0.29])
        best = torch.load(self.branch_dir / "best.pth", map_location="cpu", weights_only=True)
        expected_source = {"checkpoint": "outputs/training/continuation/best.pth",
                           "epoch": 8, "global_step": 24, "mAP": 0.30}
        self.assertEqual(best["epoch"], 8)
        self.assertEqual(best["global_step"], 24)
        self.assertEqual(best["best_source"], expected_source)
        self.assertEqual(latest["best_source"], expected_source)
        self.assertEqual(latest["best_epoch"], 8)
        self.assertEqual(latest["best_mAP"], 0.30)
        train.validate_best_checkpoint(latest, best, 3)
        report = json.loads((self.branch_dir / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["best_model"], expected_source)

    def test_missing_rng_or_momentum_cannot_silently_resume(self):
        cases = []
        no_rng = copy.deepcopy(self.latest)
        del no_rng["rng_state"]
        cases.append((no_rng, "complete"))
        partial_rng = copy.deepcopy(self.latest)
        del partial_rng["rng_state"]["numpy"]
        cases.append((partial_rng, "Python, NumPy"))
        no_momentum = copy.deepcopy(self.latest)
        first_parameter = no_momentum["optimizer_state_dict"]["param_groups"][0]["params"][0]
        del no_momentum["optimizer_state_dict"]["state"][first_parameter]["momentum_buffer"]
        cases.append((no_momentum, "momentum"))
        for saved, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                train.validate_resume_state(saved, 3)

    def test_mismatched_earlier_best_is_rejected(self):
        train.validate_best_checkpoint(self.latest, self.best, 3)
        for field, value in (("best_source", {"checkpoint": "unrelated.pth"}),
                             ("continuation", {"parent_run": "another_run"})):
            invalid_best = copy.deepcopy(self.best)
            invalid_best[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not match"):
                train.validate_best_checkpoint(self.latest, invalid_best, 3)
        invalid_latest = copy.deepcopy(self.latest)
        invalid_latest["history"][0]["total_loss"] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            train.validate_best_checkpoint(invalid_latest, self.best, 3)

    def test_legacy_upgrade_preserves_state_and_does_not_consume_host_rng(self):
        legacy = copy.deepcopy(self.latest)
        legacy["settings"] = copy.deepcopy(prepare_continuation.LEGACY_SETTINGS)
        legacy["global_step"] = 17500
        for row in legacy["history"]:
            row["global_step"] = row["epoch"] * 3500
        # This byte vector only exercises serialized CUDA metadata; no CUDA API is used.
        legacy["rng_state"]["cuda"] = [torch.arange(16, dtype=torch.uint8)]
        del legacy["continuation"]
        del legacy["best_source"]
        before = copy.deepcopy(legacy)
        host_rng = train.capture_rng_state()
        settings = {**legacy["settings"], "num_workers": 0, "transform": TRANSFORM}
        upgraded = prepare_continuation.upgraded_checkpoint(
            legacy, settings, self.continuation, self.best_source)
        self.assert_state_equal(legacy, before)
        self.assert_state_equal(train.capture_rng_state(), host_rng)
        self.assertEqual(upgraded["settings"], settings)
        self.assertEqual(upgraded["continuation"], self.continuation)
        self.assertEqual(upgraded["best_source"], self.best_source)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "history"):
            self.assertIs(upgraded[key], legacy[key])
        missing_cuda = copy.deepcopy(legacy)
        missing_cuda["rng_state"]["cuda"] = []
        with self.assertRaisesRegex(ValueError, "single-device CUDA RNG"):
            prepare_continuation.upgraded_checkpoint(
                missing_cuda, settings, self.continuation, self.best_source)
        missing_rng = copy.deepcopy(legacy)
        del missing_rng["rng_state"]
        with self.assertRaisesRegex(ValueError, "complete"):
            prepare_continuation.upgraded_checkpoint(
                missing_rng, settings, self.continuation, self.best_source)

    def test_legacy_settings_require_saved_transform_and_inference_evidence(self):
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        transform = train.get_transform_settings(GeneralizedRCNNTransform(
            800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        fields = ("image_mean", "image_std", "min_size", "max_size", "size_divisible", "fixed_size")
        report = {"settings": {"model": {
            "transform": {key: transform[key] for key in fields},
            "roi_heads": {"score_threshold": 0.05, "nms_iou_threshold": 0.5,
                          "detections_per_image": 100}}}}
        settings = prepare_continuation.compatible_settings(report)
        self.assertEqual(settings["transform"], transform)
        self.assertEqual(settings["num_workers"], 0)
        wrong_transform = copy.deepcopy(report)
        wrong_transform["settings"]["model"]["transform"]["min_size"] = [640]
        with self.assertRaisesRegex(ValueError, "native 800/1333"):
            prepare_continuation.compatible_settings(wrong_transform)
        wrong_threshold = copy.deepcopy(report)
        wrong_threshold["settings"]["model"]["roi_heads"]["score_threshold"] = 0.1
        with self.assertRaisesRegex(ValueError, "inference thresholds"):
            prepare_continuation.compatible_settings(wrong_threshold)

    def test_preparation_rejects_other_source_before_reading_artifacts(self):
        with patch.object(prepare_continuation, "sha256") as fingerprint:
            with self.assertRaisesRegex(ValueError, "restricted"):
                prepare_continuation.prepare_continuation(self.original_dir, self.branch_dir)
        fingerprint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
