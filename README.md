# BDD100K Faster R-CNN Baseline

Train and evaluate Torchvision Faster R-CNN ResNet50-FPN on the supplied fixed BDD100K subset. Normal training uses Torchvision's default aspect-preserving 800/1333 input rule and native batch padding/ROI pooling. The user completed four independent experiments and one continuation branch from the old five-epoch run through epoch 10. The inherited epoch-4 mAP 0.25544452083836444 remains the highest observed validation result. All runs are preserved; final baseline selection is pending. Synthetic checks and historical preparation plans are separate from measured results.

## Start Here

The old five-epoch run's epoch-4 best checkpoint has now completed one separately authorized full-validation pass. See the [readable validation summary](outputs/evaluation/20261021_143617_089705_epoch4_best/validation_summary.md); the same separate directory contains raw JSON/CSV, predictions and source/integrity records. The pass covered all 1,000 validation images and exported 69,740 predictions. Its mAP/AP50/AP75 exactly reproduce the original 0.25544452083836444 / 0.5170826995660548 / 0.21488694482832407. Previously missing epoch-4 per-category/size/recall details are now available in this separate directory. This is checkpoint evaluation, with zero training updates and no new baseline or recipe selection.

The user has **completed** `outputs/training/20261021_legacy5_to_epoch10/`, continuing `20261020_182545_580059/latest.pth` from epoch 5 / step 17,500 through epochs 6-10. The branch added five full epochs and 17,500 updates; its epoch-10 / step-35,000 latest state includes the parent's first five epochs, so it is not a fifth independent initialization or ten additional epochs. Each new epoch trained all 7,000 images and validated all 1,000 images. The retained best remains the parent's epoch 4 / step 14,000, mAP 0.25544452083836444. Latest `validation.json` / `predictions.json` belong to epoch 10, mAP 0.24465600637621113; they are not the best model's detailed results.

The best new epoch was 7, mAP 0.24794138977720387, below the inherited best by 0.7503131061160573 percentage points. Training loss continued to decrease while validation fluctuated without sustained improvement. This supports a validation-plateau interpretation under the unchanged constant LR 0.005; it does not establish strict optimization convergence or the model's performance ceiling. No further training recipe or final baseline is selected. Exact training duration remains `Not measured`. The continuation review inspected saved artifacts only; the assistant has not executed training.

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
| `num_workers` | 0 |
| `PRINT_EVERY` | 50 batches |

These remain provisional settings, not a selected final baseline. Ten epochs are an observation window, not a convergence claim. Only the input rule is restored; this does not adopt an entire official training recipe. SGD remains constant-LR/float32 with no scheduler/warmup/AMP/augmentation and zero workers. The official weights must already be cached under `outputs/torch/hub/checkpoints/`; the trainer refuses to start if they are missing. All four experiments remain separate and preserved. Exact historical durations are `Not measured` because the trainer does not log them.

The input decision was made after inspecting the lower 640 result. Main comparisons may retain each detector's documented official input rule while sharing data, classes, evaluator and checkpoint-selection rules. This compares configured detection systems, not architecture alone or equal compute budgets. The 640 run remains a supplementary result. Its ten-epoch budget differs from the old five-epoch budget, so their best-score gap is not a strict resolution-only ablation. Official defaults are a justified starting point, not a guarantee of the best BDD100K score.

## Completed Experiments

The first two and fourth runs used aspect-preserving short-edge target 800, long-edge limit 1333 and batch padding to multiples of 32. The third used aspect-preserving resize and right/bottom padding to a fixed 640 x 640 canvas.

| Run under `outputs/training/` | Completed epochs / updates | Best validation mAP | Best / latest epoch |
|---|---:|---:|---:|
| `20261020_155352_056862` | 1 / 3,500 | 0.20033579847015223 | 1 / 1 |
| `20261020_182545_580059` | 5 / 17,500 | 0.25544452083836444 | 4 / 5 |
| `20261021_002448_944703` | 10 / 35,000 | 0.17773109411680846 | 4 / 10 |
| `20261021_022856_226381` | 10 / 35,000 | 0.2519508348597597 | 7 / 10 |
| `20261021_legacy5_to_epoch10` (continuation) | 10 / 35,000 total; 5 / 17,500 new | 0.25544452083836444 (inherited) | 4 / 10 |

The continuation row inherits the five-epoch parent's first five history rows and best checkpoint; it is not another independent experiment. Across the four independent runs and five newly continued epochs, 31 complete validation cycles were executed, without counting inherited history twice.

Five-epoch best: `best.pth` is epoch 4 / step 14,000, AP50 0.5170826995660548 and AP75 0.21488694482832407. The original training directory's `latest.pth`, `validation.json` and `predictions.json` remain epoch 5 / step 17,500, mAP 0.23822375438048712. Those latest files must not be presented as best-model output. Intermediate detailed reports were overwritten during training; the separate epoch-4 evaluation above now supplies new per-class/size/recall results and predictions without replacing original files. See [Experiment Log](docs/EXPERIMENT_LOG.md) for every epoch and the new validation pass.

Ten-epoch 640 best: `best.pth` is epoch 4 / step 14,000, mAP 0.17773109411680846, AP50 0.3619035810493331 and AP75 0.15257382817657714. Latest checkpoint and JSON files are epoch 10 / step 35,000, mAP 0.17702375663177553; the prediction file contains 64,422 detections over all 1,000 validation IDs. Its epoch-4 detailed metrics/predictions were not retained separately. The best-score difference from the old five-epoch run is -7.7713426721555985 percentage points. Training loss kept decreasing and the last two validation scores increased; sufficient convergence and a causal explanation are unverified. Saved artifacts were inspected without running training, inference or scoring again.

Restored-input ten-epoch best: `best.pth` is epoch 7 / step 24,500, mAP 0.2519508348597597, AP50 0.5155850158288147 and AP75 0.2030129302252958. Latest checkpoint and JSON files are epoch 10 / step 35,000, mAP 0.24015201202561187; predictions contain 61,137 detections over all 1,000 original validation IDs. Epoch-7 detailed metrics/predictions were not retained separately. The best is 0.34936859786047214 percentage points below the old five-epoch best. Later validation fluctuated while training loss decreased; simply extending this unchanged constant learning rate has not shown sustained validation improvement. The gap's cause and statistical significance are not established. No new training setting or final baseline was selected during this artifact review.

Completed old-five-epoch continuation: epoch-6-to-10 mAP values are 0.24640704578621095, 0.24794138977720387, 0.24324549105981796, 0.22546045840518608 and 0.24465600637621113. Its inherited `best.pth` is still epoch 4 / step 14,000, AP50 0.5170826995660548 and AP75 0.21488694482832407, with the parent source/hash preserved. The branch's latest checkpoint/report are epoch 10 / step 35,000, AP50 0.4986412390330815 and AP75 0.20920142467848557; 60,067 predictions cover all 1,000 original validation IDs. Training loss fell from the parent's epoch-5 0.8152584325985185 to epoch-10 0.7575437616668641. No new epoch exceeded the inherited best. The completed continuation details and bounded interpretation are in [Experiment Log](docs/EXPERIMENT_LOG.md).

## Read the Code

Training and model inference are in four files, with scoring delegated to the supplied team script:

| File | Responsibility |
|---|---|
| [src/dataset.py](src/dataset.py) | Read the fixed train/val COCO data; detection collation and DataLoader |
| [src/model.py](src/model.py) | Official COCO_V1 weights, new 11-class head and default preprocessing; optional historical square helpers |
| [src/train.py](src/train.py) | Settings, `train_one_epoch()`, the epoch loop, and small save/resume helpers |
| [src/prepare_continuation.py](src/prepare_continuation.py) | Explicit CPU-only preparation of the audited legacy five-epoch branch; never trains |
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

Continuation stays in the checkpoint's directory. The explicitly prepared old-five-epoch branch uses its new directory and never writes into the parent. Checkpoints store model/SGD state, completed epoch/global step, best mAP/epoch, training settings, categories, compact epoch history, and Python/NumPy/Torch CPU/CUDA RNG. RNG is restored before constructing the next shuffled iterator. Latest is saved atomically before best; if that boundary was itself best, resume can rebuild best from latest. An earlier sibling `best.pth` must match the restored history, categories, settings and best identity.

Strict resume checks remain active: settings include preprocessing identity and workers 0, with complete history/counter/best, SGD momentum and RNG checks. Directly resuming original legacy checkpoints still fails for missing metadata; the 640 transform also remains incompatible. `src.prepare_continuation` is an explicit, separate CPU-only conversion restricted to the audited five-epoch source and its five artifact SHA-256 fingerprints. Saved epoch-5 transform fields and historical source/installed Torchvision establish only the missing `settings.transform` and `settings.num_workers`; the original model, optimizer, history, counters and RNG values are preserved. No automatic migration or global validation bypass is added.

At preparation time the branch contained inherited epoch-5 `latest.pth`, epoch-4 `best.pth`, five-row `history.csv`, `continuation.json` and `checkpoint_verification.json`. The last two remain historical preparation records; their Prepared status does not describe the now-completed training. Current latest/history/report show epoch 10 / step 35,000 / ten history rows, with new validation/prediction JSON from epoch 10. The unchanged inherited best still identifies the parent's `best.pth` and hash because no new complete-validation mAP exceeded 0.25544452083836444. Both checkpoints retain `continuation` and `best_source`; validation `epoch` identifies the actual evaluated model independently of `best_model`. Parent JSON files remain epoch 5. The separately stored epoch-4 validation pass supplies best-model details without changing either training directory.

Resume requires unchanged data/settings and workers 0. An incomplete epoch is replayed; the first epoch has no recovery point until validation completes. Batch-level/distributed/multi-worker resume and bitwise agreement across hardware/versions are unsupported. The original run did not record executed source/runtime/data hashes; evidenced metadata reconstruction cannot prove exact execution identity or bitwise CUDA trajectory. The trainer does not compare environment or data hashes, and deterministic algorithms are not forced.

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

The standalone full-validation/condition CLI above has not been run; per-epoch full-validation scoring completed inside training for all four independent experiments and the five added continuation epochs. The trainer's `predictions.json` belongs to the latest evaluated epoch, as identified in `validation.json`, and is not necessarily from `best.pth`. The shared CLI emits JSON and CSV. It applies no extra confidence filter or box clipping; model-side candidate export rules must be agreed across the group. Current Faster R-CNN thresholds remain unchanged and recorded. Condition analysis remains pending.

The shared script includes a test option for future final evaluation, but training never invokes it. Test data must remain unused for selection/tuning until model and parameters are fixed. No full condition or test evaluation has been run. No prediction visualization is generated.

## Development Checks and Records

The older `src/check_data.py`, `src/check_model.py`, `src/check_training.py`, and `configs/training_check.json` remain independent development checks. Normal training does not depend on them. Earlier check artifacts are preserved. Synthetic tests use the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The historical 640 preparation passed 32 synthetic tests; default-input restoration passed 34. Both subsequent user-run experiments are recorded above, and their documentation-only artifact review did not rerun that historical suite. The current continuation preparation passed **41 synthetic CPU tests** with CUDA hidden, compilation of `src` / `tests` and both CLI help commands. Independent CPU `weights_only=True` disk roundtrips preserved every original checkpoint key/state value except the two evidenced settings additions and new lineage fields: each checkpoint retained 295 finite model tensors, 72 finite nonzero momentum buffers and all four RNG families including one saved CUDA state. All twenty original artifacts across the four completed experiments retained identical SHA-256, sizes and modification times. Train/val annotations and mapping confirm 7,000 / 1,000 images and unchanged categories; no source image was decoded and no new dataset hash audit is claimed. Cached official weights exist (167,502,836 bytes) but were not loaded or downloaded. No real Faster R-CNN was constructed/trained/inferred, no real-data scoring/benchmark/test access occurred, and CUDA restoration/bitwise trajectory was not executed.

See [Current Status](docs/CURRENT_STATUS.md), [Experiment Log](docs/EXPERIMENT_LOG.md), [Project Context](docs/PROJECT_CONTEXT.md), [Environment](docs/ENVIRONMENT.md), and [AGENTS.md](AGENTS.md). Reuse the environment; no dependency installation is needed.

The completed-continuation review reads saved CSV/JSON and CPU checkpoint state only. Latest/best history, counters, settings, momentum, lineage and best consistency passed inspection; all five parent artifact hashes still match the preparation manifest and the branch's inherited best hash still matches its prepared copy. No detector, inference, scoring, benchmark or test access was run, and the historical 41-test suite was not rerun for documentation changes.

The subsequent epoch-4 validation used the existing model/evaluator on `cuda:0`, batch size 2, workers 0 and seed 42, with unchanged native 800/1333 preprocessing and thresholds. It executed exactly one val pass, no training/condition/test evaluation. All 295 model-state tensors stayed unchanged and no parameter gradients were created. Preservation checks passed for 1,037 protected source/artifact/cache/validation files. The existing project source/tests were unchanged and the historical test suite was not rerun. The measured 48.30145260000427 seconds belongs only to this new evaluation, not historical training duration.

`data/`, `outputs/`, `.venv/`, weights, checkpoints and caches are ignored and must stay untracked. Every commit and push requires explicit user confirmation.
