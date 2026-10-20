"""Check the training loop and epoch-boundary resume using tiny synthetic data."""

import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from src import train
from src.dataset import make_dataloader


class SyntheticDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return torch.full((3, 1, 1), (index + 1) / 10), {
            "image_id": torch.tensor([101 + index], dtype=torch.int64)
        }


class TinyDetector(torch.nn.Module):
    """Use the detector loss interface and all three host random generators."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(2))
        self.image_order = []
        self.batch_sizes = []

    def forward(self, images, targets):
        self.image_order.extend(target["image_id"].item() for target in targets)
        self.batch_sizes.append(len(images))
        jitter = random.random() + float(np.random.random()) + torch.rand(()).item()
        target = torch.stack([image.mean() for image in images]).mean() + jitter / 10
        loss = ((self.weight - target) ** 2).sum()
        return {"loss_classifier": loss, "loss_box_reg": loss * 0.5}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.original_rng = train.capture_rng_state()
        output_root = Path(__file__).resolve().parents[1] / "outputs"
        output_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_training_", dir=output_root)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        train.restore_rng_state(self.original_rng)
        self.temporary.cleanup()

    def build_training(self):
        model = TinyDetector()
        optimizer = train.make_optimizer(
            model, {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
        )
        loader = make_dataloader(SyntheticDataset(), batch_size=2, shuffle=True)
        return model, optimizer, loader

    def run_epoch(self, model, optimizer, loader, epoch):
        with redirect_stdout(io.StringIO()):
            summary = train.train_one_epoch(
                model, loader, optimizer, torch.device("cpu"), epoch=epoch, print_every=100
            )
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["images"], 5)
        self.assertTrue(np.isfinite(summary["total_loss"]))
        self.assertEqual(set(summary["losses"]), {"loss_classifier", "loss_box_reg"})
        return summary["steps"]

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_state_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_epoch_resume_matches_uninterrupted_training(self):
        train.seed_everything(17)
        full_model, full_optimizer, full_loader = self.build_training()
        initial_weight = full_model.weight.detach().clone()
        full_model.eval()
        full_steps = self.run_epoch(full_model, full_optimizer, full_loader, 1)
        full_steps += self.run_epoch(full_model, full_optimizer, full_loader, 2)
        full_rng = train.capture_rng_state()
        self.assertTrue(full_model.training)
        self.assertFalse(torch.equal(full_model.weight, initial_weight))
        self.assertEqual(full_model.batch_sizes, [2, 2, 1, 2, 2, 1])

        train.seed_everything(17)
        first_model, first_optimizer, first_loader = self.build_training()
        steps = self.run_epoch(first_model, first_optimizer, first_loader, 1)
        checkpoint_path = self.root / "latest.pth"
        train.save_checkpoint(checkpoint_path, {
            "model_state_dict": first_model.state_dict(),
            "optimizer_state_dict": first_optimizer.state_dict(),
            "rng_state": train.capture_rng_state(), "epoch": 1, "global_step": steps,
        })

        train.seed_everything(99)
        resumed_model, resumed_optimizer, resumed_loader = self.build_training()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        resumed_model.load_state_dict(checkpoint["model_state_dict"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Construction consumes random values; restore before the next loader iterator.
        train.restore_rng_state(checkpoint["rng_state"])
        steps = checkpoint["global_step"] + self.run_epoch(
            resumed_model, resumed_optimizer, resumed_loader, checkpoint["epoch"] + 1
        )

        self.assertEqual(full_steps, 6)
        self.assertEqual(steps, full_steps)
        self.assertEqual(full_model.image_order, first_model.image_order + resumed_model.image_order)
        self.assertEqual(sorted(resumed_model.image_order), [101, 102, 103, 104, 105])
        self.assert_state_equal(full_model.state_dict(), resumed_model.state_dict())
        self.assert_state_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
        self.assert_state_equal(full_rng, train.capture_rng_state())

    def test_nonfinite_loss_stops_before_optimizer_update(self):
        model, optimizer, loader = self.build_training()
        initial = model.weight.detach().clone()
        with patch.object(model, "forward", return_value={"loss": model.weight.sum() * float("nan")}), \
                patch.object(optimizer, "step", wraps=optimizer.step) as update:
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                train.train_one_epoch(model, loader, optimizer, "cpu", epoch=1)
            update.assert_not_called()
        self.assertTrue(torch.equal(model.weight, initial))

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        destination = self.root / "latest.pth"
        train.save_checkpoint(destination, {"step": 1})
        original = destination.read_bytes()

        def fail_during_save(payload, path):
            Path(path).write_bytes(b"partial checkpoint")
            raise OSError("Synthetic disk write failure")

        with patch.object(train.torch, "save", side_effect=fail_during_save):
            with self.assertRaisesRegex(OSError, "Synthetic disk write failure"):
                train.save_checkpoint(destination, {"step": 2})
        self.assertEqual(destination.read_bytes(), original)
        self.assertFalse(destination.with_suffix(".pth.tmp").exists())
        self.assertEqual(torch.load(destination, weights_only=True)["step"], 1)


if __name__ == "__main__":
    unittest.main()
