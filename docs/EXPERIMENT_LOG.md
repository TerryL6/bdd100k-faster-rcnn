# Experiment Log

Last updated: 2026-10-19.

## Experiment Status

No model training or evaluation experiments have been run in this repository. No checkpoints or model metrics exist. Project initialization and dataset preparation reports are not model experiments.

## Recording Rules

- Record every completed training experiment with its actual configuration, metrics, and observations. Record failed or interrupted attempts accurately when applicable.
- Use English throughout. Never invent results or replace missing/undefined metrics with zero.
- Use validation to compare runs and choose model settings. Record final test evaluation only after model and hyperparameters are fixed.
- Keep checkpoints and generated results under ignored `outputs/`. Record their relative paths here without committing the artifacts.
- Update `docs/CURRENT_STATUS.md` after important stages and record lasting decisions in `docs/PROJECT_CONTEXT.md` when necessary.

## Template for a Future Run

Copy this template for a real run; it is not an experiment entry.

```text
Run ID:
Status: Planned / Running / Completed / Failed / Interrupted
Start and end time (with timezone):
Objective or hypothesis:
Code commit and uncommitted changes:
Configuration path and actual settings:
Dataset root, fixed split, and annotation/subset files:
Category mapping and image ID handling:
Hardware and device:
Python, PyTorch, Torchvision, CUDA, and evaluator versions:
Random seed and determinism settings:
Model variant, pretrained weights, and detection head:
Transforms, resize settings, and augmentation:
Optimizer, learning rate, weight decay, momentum, and scheduler:
Batch size, epoch count, and any early stopping rule:
Training duration and completed epochs:
Checkpoint path and validation-based selection criterion:
Evaluation settings (IoU thresholds, max detections, and any score filtering):
Validation results (mAP, AP50, AP75, size AP, and recall):
Per-class AP (mark missing ground-truth classes Not evaluable):
Condition results (val_all, clear_daytime, night, dawn_dusk, rainy, overcast):
Artifacts (checkpoint, predictions, metrics, and plots):
Observations, failures, limitations, and next decision:
Final test evaluation, if authorized after settings are frozen: Not evaluated
```
