# BDD100K Faster R-CNN Baseline

Train and evaluate Torchvision Faster R-CNN ResNet50-FPN on the supplied fixed BDD100K subset. Full baseline training and complete-validation performance have not been measured. Small functionality checks are not baseline results.

## Start Here

Use the existing `.venv` from the repository root. Edit the settings at the top of `src/train.py`, then run:

```powershell
& .\.venv\Scripts\python.exe -m src.train
```

This command starts a full training epoch over 7,000 training images and evaluates all 1,000 validation images. It has **not** been run. Full execution still requires the user's confirmation.

| Setting | Initial value |
|---|---:|
| `EPOCHS` | 1 |
| `BATCH_SIZE` / `VAL_BATCH_SIZE` | 2 / 2 |
| `LEARNING_RATE` | 0.005 |
| `MOMENTUM` | 0.9 |
| `WEIGHT_DECAY` | 0.0005 |
| `SEED` | 42 |
| `DEVICE` | `cuda:0` |
| `PRINT_EVERY` | 50 batches |

These are provisional starting values, not tuned hyperparameters. One epoch is not assumed sufficient. SGD uses constant learning rate, float32, no scheduler/warmup/AMP/augmentation, and zero loader workers. The existing official weights must already be cached under `outputs/torch/hub/checkpoints/`; the trainer does not download missing weights.

## Read the Code

Training and model inference are in four files, with scoring delegated to the supplied team script:

| File | Responsibility |
|---|---|
| [src/dataset.py](src/dataset.py) | Read the fixed train/val COCO data; detection collation and DataLoader |
| [src/model.py](src/model.py) | Load official COCO_V1 Faster R-CNN weights and replace the head with 11 classes |
| [src/train.py](src/train.py) | Settings, `train_one_epoch()`, the epoch loop, and small save/resume helpers |
| [src/evaluate.py](src/evaluate.py) | Validation inference and original-ID prediction export; delegates scoring to the team script |
| [evaluate_team.py](evaluate_team.py) | Shared COCO bbox scoring and fixed validation-condition evaluation |

In `train_one_epoch()`, read the batch loop: move images/targets to the GPU, clear gradients, call `model(images, targets)`, sum the four losses, call `backward()`, and call `optimizer.step()`. `model.train()` selects training behavior; it does not update weights. The detector computes classification, box-regression, RPN objectness, and RPN box-regression losses internally.

In `main()`, read the three numbered sections: create loaders, create the model/SGD, then repeat training -> validation -> checkpoint saving. Small RNG/checkpoint/CSV helpers are below this flow. There is no external training config, per-batch log file, environment snapshot, source/data hashing, or report status machine. Non-finite total loss stops with the epoch/batch position; ordinary Python/PyTorch errors keep their traceback. Terminal progress prints periodically and each epoch prints mean loss and validation metrics.

## Results and Resume

A new run uses `outputs/training/<timestamp>/`:

- `latest.pth`: last completed training-and-validation epoch.
- `best.pth`: highest complete-validation bbox mAP; earlier epochs win ties.
- `history.csv`: one row per epoch with individual/total mean losses, LR, global step, mAP/AP50/AP75. Loss means are image-weighted batch means.
- `validation.json` and `predictions.json`: latest validation report and COCO predictions, overwritten after each completed validation. The report contains the epoch, all 12 COCO summary metrics, per-category AP/AP50/AP75 and counts, evaluator settings and model transforms/inference settings.

Increase `EPOCHS` at the top of `src/train.py`, keep the other settings unchanged, and resume from the run's latest checkpoint:

```powershell
& .\.venv\Scripts\python.exe -m src.train --resume outputs/training/<run>/latest.pth
```

Continuation stays in the same output directory. Checkpoints store model/SGD state, completed epoch/global step, best mAP/epoch, training settings, categories, compact epoch history, and Python/NumPy/Torch CPU/CUDA RNG. RNG is restored before constructing the next shuffled iterator. Latest is saved atomically before best; if that boundary was itself best, resume can rebuild best from latest. Keep an earlier `best.pth` beside latest when their epochs differ.

Only checkpoints produced by this simplified trainer are accepted. Earlier implementation/check checkpoints remain untouched and are not trained baselines. Resume requires unchanged data/settings and workers 0. An incomplete epoch is replayed; the first epoch has no recovery point until validation completes. Batch-level/distributed/multi-worker resume and bitwise agreement across hardware/versions are unsupported. The trainer does not compare environment or data hashes, and deterministic algorithms are not forced. RNG restoration is not a guarantee that every CUDA operation is deterministic.

## Dataset and Evaluation Rules

Use the supplied `data/bdd100k_subset_10k/README.md`, `class_mapping.json`, cleaned `annotations/train.json` / `val.json`, and matching original images. Splits are fixed: 7,000 train, 1,000 validation, 2,000 reserved test. Dataset files are never rewritten or repartitioned. `bike` and `motor` keep their supplied names; foreground category IDs are 1-10 and background is 0.

Images remain original-size RGB float32 tensors in `[0, 1]`. Normalization and resize stay inside Faster R-CNN: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, min size 800, max size 1333, and padding divisor 32. Model/body trainability is unchanged.

Validation uses `eval()` and `torch.inference_mode()`, preserves original image/category IDs, and converts predicted boxes from original-pixel xyxy to COCO xywh only in memory. All scores are computed by the supplied `evaluate_team.py`, using standard category-aware COCOeval: bbox IoUs 0.50:0.05:0.95, 101 recall points, maxDets `[1, 10, 100]`, default all/small/medium/large areas, metric scale 0-1. Model defaults remain score threshold 0.05, NMS 0.5 and 100 detections/image, with no additional score filter. Empty predictions are supported. Missing-GT categories, including validation's railway `train`, use `Not evaluable` and are excluded from the COCO mean. Full validation coverage is required by default; explicit sampled evaluation is marked as a functionality check.

The team script preserves its original mAP/AP50/per-class AP calculation and adds AP75, size/recall and per-class AP50/AP75. Undefined values are JSON `null` in its output, meaning `Not evaluable`; the training report renders them as that string. Its fixed condition subsets are clear_daytime, night, dawn_dusk, rainy and overcast. Differences are `clear_daytime mAP - subset mAP`, descriptive comparisons rather than causal effects.

For a completed run's exported predictions, the team's CLI can score full validation and the five fixed subsets without loading images or running the detector again:

```powershell
& .\.venv\Scripts\python.exe evaluate_team.py --root data/bdd100k_subset_10k --predictions outputs/training/<run>/predictions.json --split val --output outputs/training/<run>/team_validation.json
```

This command has not been run on the full validation set. The trainer's `predictions.json` belongs to the most recently evaluated epoch, as identified in `validation.json`; it is not necessarily predictions from `best.pth`. Evaluate the desired checkpoint first before claiming results for it. The shared CLI emits JSON and CSV. It applies no extra confidence filter or box clipping; model-side candidate export rules must be agreed across the group. Current Faster R-CNN defaults remain unchanged and recorded.

The shared script includes a test option for future final evaluation, but training never invokes it. Test data must remain unused for selection/tuning until model and parameters are fixed. No full condition or test evaluation has been run. No prediction visualization is generated.

## Development Checks and Records

The older `src/check_data.py`, `src/check_model.py`, `src/check_training.py`, and `configs/training_check.json` remain independent development checks. Normal training does not depend on them. Earlier check artifacts are preserved. Synthetic tests use the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [Current Status](docs/CURRENT_STATUS.md) for actual verification, [Experiment Log](docs/EXPERIMENT_LOG.md) for measured process checks, [Project Context](docs/PROJECT_CONTEXT.md) for lasting decisions, [Environment](docs/ENVIRONMENT.md) for the verified `.venv`, and [AGENTS.md](AGENTS.md) for working rules. Reuse the environment; no dependency installation is needed.

`data/`, `outputs/`, `.venv/`, weights, checkpoints and caches are ignored and must stay untracked. Every commit and push requires explicit user confirmation.
