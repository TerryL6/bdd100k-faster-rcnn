# BDD100K Faster R-CNN Baseline

Train and evaluate Torchvision Faster R-CNN ResNet50-FPN on the supplied fixed BDD100K subset. Normal training has returned to Torchvision's default aspect-preserving 800/1333 input rule and native batch padding/ROI pooling. A fresh ten-epoch run with that restored rule is prepared, not started. The user completed three prior independent experiments, all preserved below. Synthetic checks and unexecuted plans are separate from measured results.

## Start Here

Use the existing `.venv` and local RTX 5070. Current settings at the top of `src/train.py` prepare a fresh ten-epoch run with the default input rule. When the user chooses to start it manually from the repository root, the command is:

```powershell
& .\.venv\Scripts\python.exe -m src.train
```

Without `--resume`, this creates a separate timestamped output directory, initializes from cached official COCO_V1 weights and a new 11-class head, and targets full 7,000-image training plus 1,000-image validation every epoch. It restores none of the three old checkpoints. The restored-input run has not started; its metrics and elapsed duration are `Not measured`. The assistant only displays training commands unless explicitly instructed to execute training.

| Setting | Current value |
|---|---:|
| `EPOCHS` | 10 |
| `IMAGE_SIZE` | `None` (native transform/pooler) |
| Resize / padding | Short edge target 800, long edge limit 1333; batch padding to multiples of 32 |
| `BATCH_SIZE` / `VAL_BATCH_SIZE` | 2 / 2 |
| `LEARNING_RATE` | 0.005 |
| `MOMENTUM` | 0.9 |
| `WEIGHT_DECAY` | 0.0005 |
| `SEED` | 42 |
| `DEVICE` | `cuda:0` |
| `PRINT_EVERY` | 50 batches |

These remain provisional settings, not a selected final baseline. Ten epochs are an observation window, not a convergence claim. Only the input rule is restored; this does not adopt an entire official training recipe. SGD remains constant-LR/float32 with no scheduler/warmup/AMP/augmentation and zero workers. The official weights must already be cached under `outputs/torch/hub/checkpoints/`; the trainer refuses to start if they are missing. All three experiments remain separate and preserved. Exact historical durations are `Not measured` because the trainer does not log them.

The input decision was made after inspecting the lower 640 result. Main comparisons may retain each detector's documented official input rule while sharing data, classes, evaluator and checkpoint-selection rules. This compares configured detection systems, not architecture alone or equal compute budgets. The 640 run remains a supplementary result. Its ten-epoch budget differs from the old five-epoch budget, so their best-score gap is not a strict resolution-only ablation. Official defaults are a justified starting point, not a guarantee of the best BDD100K score.

## Completed Experiments

The first two runs used aspect-preserving short-edge target 800, long-edge limit 1333 and batch padding to multiples of 32. The third used aspect-preserving resize and right/bottom padding to a fixed 640 x 640 canvas.

| Run under `outputs/training/` | Completed epochs / updates | Best validation mAP | Best / latest epoch |
|---|---:|---:|---:|
| `20261020_155352_056862` | 1 / 3,500 | 0.20033579847015223 | 1 / 1 |
| `20261020_182545_580059` | 5 / 17,500 | 0.25544452083836444 | 4 / 5 |
| `20261021_002448_944703` | 10 / 35,000 | 0.17773109411680846 | 4 / 10 |

Five-epoch best: `best.pth` is epoch 4 / step 14,000, AP50 0.5170826995660548 and AP75 0.21488694482832407. `latest.pth`, `validation.json` and `predictions.json` are epoch 5 / step 17,500, mAP 0.23822375438048712. The latest detailed metrics/predictions must not be presented as the best model's output. Intermediate detailed reports were overwritten; epoch 4 per-class/size/recall results are not separately available. Saved histories, reports, prediction coverage and CPU-only checkpoint metadata were inspected without new inference or scoring. See [Experiment Log](docs/EXPERIMENT_LOG.md) for every epoch.

Ten-epoch 640 best: `best.pth` is epoch 4 / step 14,000, mAP 0.17773109411680846, AP50 0.3619035810493331 and AP75 0.15257382817657714. Latest checkpoint and JSON files are epoch 10 / step 35,000, mAP 0.17702375663177553; the prediction file contains 64,422 detections over all 1,000 validation IDs. Its epoch-4 detailed metrics/predictions were not retained separately. The best-score difference from the old five-epoch run is -7.7713426721555985 percentage points. Training loss kept decreasing and the last two validation scores increased; sufficient convergence and a causal explanation are unverified. Saved artifacts were inspected without running training, inference or scoring again.

## Read the Code

Training and model inference are in four files, with scoring delegated to the supplied team script:

| File | Responsibility |
|---|---|
| [src/dataset.py](src/dataset.py) | Read the fixed train/val COCO data; detection collation and DataLoader |
| [src/model.py](src/model.py) | Official COCO_V1 weights, new 11-class head and default preprocessing; optional historical square helpers |
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

For a future continuation of a newly created **matching default-input run**, increase `EPOCHS`, keep other settings unchanged, and use that run's latest checkpoint:

```powershell
& .\.venv\Scripts\python.exe -m src.train --resume outputs/training/<run>/latest.pth
```

Continuation stays in the same output directory. Checkpoints store model/SGD state, completed epoch/global step, best mAP/epoch, training settings, categories, compact epoch history, and Python/NumPy/Torch CPU/CUDA RNG. RNG is restored before constructing the next shuffled iterator. Latest is saved atomically before best; if that boundary was itself best, resume can rebuild best from latest. Keep an earlier `best.pth` beside latest when their epochs differ.

Only compatible checkpoints produced by this simplified trainer are accepted. Settings include preprocessing identity and workers 0. The 640 checkpoint is rejected because its transform differs; the older one-/five-epoch checkpoints still lack transform metadata and are rejected rather than silently inferred or migrated. Start the prepared run without `--resume`. Earlier artifacts remain untouched. Resume requires unchanged data/settings and workers 0. An incomplete epoch is replayed; the first epoch has no recovery point until validation completes. Batch-level/distributed/multi-worker resume and bitwise agreement across hardware/versions are unsupported. The trainer does not compare environment or data hashes, and deterministic algorithms are not forced.

## Dataset and Evaluation Rules

Use the supplied `data/bdd100k_subset_10k/README.md`, `class_mapping.json`, cleaned `annotations/train.json` / `val.json`, and matching original images. Splits are fixed: 7,000 train, 1,000 validation, 2,000 reserved test. Dataset files are never rewritten or repartitioned. `bike` and `motor` keep their supplied names; foreground category IDs are 1-10 and background is 0.

Dataset images remain original-size RGB float32 tensors in `[0, 1]`, with original pixel xyxy target boxes. Both training and validation use the same model-internal preprocessing:

1. Normalize once with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
2. Use native aspect-preserving bilinear resize: short edge target 800, long edge limit 1333, `align_corners=False`, `recompute_scale_factor=True`. Actual integer content dimensions determine box scaling. There is no direct stretching or Dataset resize.
3. Use native batch padding to multiples of 32, with content top-left aligned and right/bottom fill value 0 in normalized space. There is no fixed square canvas. A synthetic 720 x 1280 input produces 749 x 1333 content padded to 768 x 1344 in the pinned runtime.
4. Retain native `MultiScaleRoIAlign`, clipping and prediction postprocess. Predictions return to original pixel xyxy before COCO xywh export with original image/category IDs. Left/top box offsets remain zero.

All changes are in memory; source images/annotations remain untouched. Reports and checkpoints record the actual transform identity. Normal training passes `IMAGE_SIZE=None` and retains the factory transform and pooler. Optional `SquarePadTransform` / `SquareCanvasRoIAlign` and their tests remain for reproducing the historical 640 experiment; they are not active in normal training. Backbone freezing, normalization and inference thresholds are unchanged.

Teammates' nominal 640 settings do not require Faster R-CNN to use the same target. YOLO letterboxing and RT-DETRv2 direct Resize already differ. Record each model's effective input rule, training recipe and budget alongside results; a common target size alone would not ensure identical information or compute budgets.

Validation uses `eval()` and `torch.inference_mode()`, preserves original image/category IDs, and converts predicted boxes from original-pixel xyxy to COCO xywh only in memory. All scores are computed by the supplied `evaluate_team.py`, using standard category-aware COCOeval: bbox IoUs 0.50:0.05:0.95, 101 recall points, maxDets `[1, 10, 100]`, default all/small/medium/large areas, metric scale 0-1. Model defaults remain score threshold 0.05, NMS 0.5 and 100 detections/image, with no additional score filter. Empty predictions are supported. Missing-GT categories, including validation's railway `train`, use `Not evaluable` and are excluded from the COCO mean. Full validation coverage is required by default; explicit sampled evaluation is marked as a functionality check.

The team script preserves its original mAP/AP50/per-class AP calculation and adds AP75, size/recall and per-class AP50/AP75. Undefined values are JSON `null` in its output, meaning `Not evaluable`; the training report renders them as that string. Its fixed condition subsets are clear_daytime, night, dawn_dusk, rainy and overcast. Differences are `clear_daytime mAP - subset mAP`, descriptive comparisons rather than causal effects.

For a completed run's exported predictions, the team's CLI can score full validation and the five fixed subsets without loading images or running the detector again:

```powershell
& .\.venv\Scripts\python.exe evaluate_team.py --root data/bdd100k_subset_10k --predictions outputs/training/<run>/predictions.json --split val --output outputs/training/<run>/team_validation.json
```

The standalone full-validation/condition CLI above has not been run; per-epoch full-validation scoring has completed inside training for all three completed experiments. The trainer's `predictions.json` belongs to the latest evaluated epoch, as identified in `validation.json`, and is not necessarily from `best.pth`. The shared CLI emits JSON and CSV. It applies no extra confidence filter or box clipping; model-side candidate export rules must be agreed across the group. Current Faster R-CNN thresholds remain unchanged and recorded. Condition analysis is outside this preparation stage.

The shared script includes a test option for future final evaluation, but training never invokes it. Test data must remain unused for selection/tuning until model and parameters are fixed. No full condition or test evaluation has been run. No prediction visualization is generated.

## Development Checks and Records

The older `src/check_data.py`, `src/check_model.py`, `src/check_training.py`, and `configs/training_check.json` remain independent development checks. Normal training does not depend on them. Earlier check artifacts are preserved. Synthetic tests use the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The historical 640 preparation passed 32 synthetic tests, and the user completed that experiment. The current default-input restoration passed **34 synthetic tests**, compilation and training CLI help, including current-setting native transform/pooler identity, default geometry/box restoration and rejection of historical checkpoint settings. No real Faster R-CNN was constructed, trained or inferred for verification; no real-data scoring, benchmark or test access occurred. The prepared restored-input run has not started, and its performance is unmeasured.

See [Current Status](docs/CURRENT_STATUS.md), [Experiment Log](docs/EXPERIMENT_LOG.md), [Project Context](docs/PROJECT_CONTEXT.md), [Environment](docs/ENVIRONMENT.md), and [AGENTS.md](AGENTS.md). Reuse the environment; no dependency installation is needed.

`data/`, `outputs/`, `.venv/`, weights, checkpoints and caches are ignored and must stay untracked. Every commit and push requires explicit user confirmation.
