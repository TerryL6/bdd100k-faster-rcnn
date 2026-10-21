# Project Context

Last updated: 2026-10-21.

## Project and Responsibility

**Project:** Improving Robustness of Object Detection under Diverse Driving Conditions.

This is a university Deep Learning group project. The repository owner's responsibility is training and evaluating the Faster R-CNN baseline. Group members have already prepared the shared dataset. Dataset preparation is outside this repository's current implementation scope.

The goal is to establish a reproducible baseline and analyze object detection performance under different driving conditions. Hyperparameters and model choices must be determined using validation results before final test evaluation.

## Authoritative Local Dataset

Dataset root: `data/bdd100k_subset_10k/`.

The supplied `README.md`, `class_mapping.json`, and cleaned COCO annotations define the data contract. Their existing content must remain untouched. Cleaning, sampling, label conversion, and split creation are already complete; do not repeat them.

```text
bdd100k_subset_10k/
|-- README.md
|-- images/{train,val,test}/
|-- annotations/
|   |-- train.json
|   |-- val.json
|   |-- test.json
|   `-- subsets/val_*.json
|-- labels/{train,val,test}/
|-- labels_original/{train,val,test}/
|-- subsets/val_*.txt
|-- manifests/{train,val,test}.txt
|-- metadata_clean.csv
|-- metadata.csv
|-- class_mapping.json
|-- data.yaml
|-- sampling_config.json
|-- reports/
|-- preview/
`-- preview_converted/
```

Brace expressions and wildcards above summarize existing paths; they are not literal directory names.

### Fixed Splits

| Split | Images | Valid detection boxes | Intended use |
|---|---:|---:|---|
| train | 7,000 | 128,949 | Fine-tuning |
| val | 1,000 | 18,465 | Model selection, tuning, and condition analysis |
| test | 2,000 | 36,951 | Final evaluation after choices are fixed |
| Total | 10,000 | 184,365 | Fixed prepared dataset |

The dataset README reports a sampling seed of 18794 and sampling approximately proportional to weather by time of day within source splits. This describes the existing preparation process; it does not select a future training seed or authorize resampling.

### COCO Annotation Contract

- Use `annotations/train.json`, `annotations/val.json`, and eventually `annotations/test.json` with their matching `images/<split>/` directories.
- Each JSON contains `images`, `annotations`, and `categories`.
- Image `file_name` contains the JPG basename, not a path relative to the dataset root. Join it to the appropriate split image directory.
- Image records include numeric COCO IDs, dimensions, weather, time of day, and scene attributes.
- Annotation `bbox` values use absolute pixel coordinates in `[x_min, y_min, width, height]` format. Annotations include `area` and `iscrowd`; the supplied detection annotations have `iscrowd=0`.
- Preserve all supplied image IDs. Use annotation image IDs, `metadata_clean.csv`'s `coco_image_id`, or `reports/image_id_map.csv` for prediction export. Never substitute batch indices or newly assigned IDs.
- The future model adapter must handle the model's required box representation in memory; COCO prediction export must use COCO box format and original IDs. Do not rewrite source annotations.

### Fixed Categories

| Name | YOLO ID | COCO ID |
|---|---:|---:|
| person | 0 | 1 |
| rider | 1 | 2 |
| car | 2 | 3 |
| truck | 3 | 4 |
| bus | 4 | 5 |
| train | 5 | 6 |
| motor | 6 | 7 |
| bike | 7 | 8 |
| traffic light | 8 | 9 |
| traffic sign | 9 | 10 |

`train` as a category means a railway train; `train/` as a directory means the training split. Preserve `motor` and `bike` exactly. Pretrained COCO model category IDs do not define this dataset's mapping.

### Validation Condition Subsets

Each subset has a filename list at `subsets/<name>.txt` and COCO ground truth at `annotations/subsets/<name>.json`. All reference images in `images/val/`. Their image IDs and category IDs match the full validation annotations.

| Name | Selection | Images | Valid boxes |
|---|---|---:|---:|
| val_all | All validation images | 1,000 | 18,465 |
| val_clear_daytime | `weather=clear` and `timeofday=daytime` | 176 | 3,241 |
| val_night | `timeofday=night` | 394 | 6,537 |
| val_dawn_dusk | `timeofday=dawn/dusk` | 77 | 1,499 |
| val_rainy | `weather=rainy` | 74 | 1,311 |
| val_overcast | `weather=overcast` | 124 | 2,604 |

Subsets overlap, so their counts cannot be added together. For example, an image can be both rainy and night. Images with undefined attributes remain in `val_all`. The dataset README reports only two foggy validation images, with no separate foggy evaluation group.

### Supporting Files and Limitations

- `labels/` and `data.yaml` support YOLO consumers; Faster R-CNN will use the cleaned COCO annotations. No changes to `data.yaml` are needed for this project.
- `labels_original/` preserves original JSON annotations for provenance and may include non-detection tasks or boxes removed during cleaning. Do not use it to recreate training labels.
- `metadata_clean.csv` describes the 10,000 prepared images and valid objects. Prefer it over the pre-cleaning `metadata.csv` for analysis.
- `manifests/` records fixed image IDs without `.jpg`. `sampling_config.json` records preparation provenance and source-machine paths; leave it unchanged.
- `reports/condition_counts.csv` and `reports/class_counts_clean.csv` support interpretation of condition and category distributions.
- The dataset README and `reports/conversion_summary.json` report 44 removed invalid boxes and 184,365 valid boxes. The README also reports image decode checks and original-file hash checks from preparation. These are preparation-stage findings, not newly executed model experiments or a new full image-integrity audit.
- No near-duplicate or video-source leakage audit was reported in the dataset README. Environment attributes may be noisy.
- Validation has zero `train` category objects. Its AP must be labeled `Not evaluable`, not zero; apply the same rule to any category without ground truth in a particular evaluated subset.
- Condition subsets differ in size and class distribution. Interpret metric differences alongside sample and object counts.
- `reports/evaluation_validation.json` is an evaluator validation summary, not a trained model result or the shared evaluation program itself.
- `preview/` and `preview_converted/` contain annotated visualizations, not input images for training.

## Model and Evaluation Plan

- Frameworks: PyTorch and Torchvision.
- Baseline: Torchvision `fasterrcnn_resnet50_fpn` initialized with explicit `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, then a new `FastRCNNPredictor` for 11 outputs (foreground IDs 1-10, background 0). Three user-run experiments are preserved: one and five epochs at 800/1333, then ten epochs with a fixed 640 canvas. The old five-epoch best mAP is 0.25544452083836444; the 640 best is 0.17773109411680846. On 2026-10-21 the user chose to restore the default 800/1333 input rule for the main Faster R-CNN work. A fresh ten-epoch run is prepared, not started; no final baseline is selected. Never use the pretrained COCO category mapping for this dataset.
- Keep the implementation small and use existing framework functionality where practical.
- Fine-tune on training only. Use validation for checkpoint selection and justified hyperparameter changes.
- The group supplied root-level `evaluate_team.py` on 2026-10-20. Use it as the shared bbox scoring standard; `src/evaluate.py` delegates to its `evaluate()` rather than duplicating metric computation. Preserve the original COCO AP protocol and document compatibility fixes/extensions.
- Plan to report bounding-box COCO mAP at IoU 0.50:0.95, AP50, AP75, size-based AP, recall, and per-class AP where evaluable, along with full and condition-specific validation results.
- Preserve evaluation settings, prediction export conventions, configurations, seeds, dependency versions, and checkpoint identity so runs can be compared fairly.
- Freeze model and hyperparameters before final test evaluation. Never tune or choose a model based on test performance.
- Record genuine configurations, results, observations, and limitations in `docs/EXPERIMENT_LOG.md` after each training experiment. Keep generated artifacts in ignored `outputs/`.

## Environment Decision

The 2026-10-19 inspection found Windows 11 x64, an RTX 5070 with approximately 12 GB VRAM, driver 617.42, and standard CPython 3.13.15. The existing global Torch 2.14.0+cu130 and Torchvision 0.29.0+cu130 passed CPU/CUDA tensor and detection-operator checks, including NMS and ROIAlign.

That working pair is now independently installed in a clean local `.venv`, with NumPy 2.5.3, Pillow 12.3.0, and pycocotools 2.0.11. Global site-packages are not inherited, and the existing global packages were not changed. Isolated dependency, CPU/CUDA operator, in-memory image conversion, and synthetic COCOeval checks passed. No BDD100K images or model predictions were used for these checks.

`requirements.txt` pins direct dependencies; `requirements-lock.txt` captures all 15 resolved installed distribution versions, including packaging tools. [Environment Setup and Verification](ENVIRONMENT.md) records official compatibility evidence, installation order, interpreter usage, completed checks, and their limits. Use the project's `.venv` interpreter for subsequent work.

## Current Scope and Open Decisions

Repository initialization, isolated environment setup/verification, and the training/validation Dataset and DataLoader stage are complete. `src/dataset.py` reads only the fixed cleaned COCO `train` and `val` splits, keeps their original image/category IDs and category order, and checks categories against `class_mapping.json`. Images are original-size RGB float32 tensors in `[0, 1]`; boxes become float32 `xyxy` only in memory. Targets retain source `area` and `iscrowd`, along with int64 labels and a one-element int64 image ID. No source boxes are clipped, filtered, or rounded.

Detection collation returns image and target lists so object counts may differ within a batch. `make_dataloader` keeps the final partial batch and exposes batch size, shuffling, and worker count. `src/check_data.py` provides a small CPU-only training/validation smoke check with optional previews in ignored `outputs/data_check/`. Its sample selection and batch defaults are checking settings, not new dataset definitions or training choices. The source dataset remains unchanged. See the root README for commands and the current status for actual verification scope.

The separately authorized model-integration stage is complete. `src/model.py` loads the official COCO_V1 detection weights before replacing the head; Torch Hub's process-wide cache is set to `outputs/torch/hub/`, with weights under its `checkpoints/` directory. The exact source URL and downloaded-file SHA-256 are recorded in current status and the local check report. No additional ImageNet backbone weights are needed.

`src/check_model.py` reuses the existing Dataset and DataLoader for one two-image training-mode loss/backward check and one two-image validation inference check on CUDA. It verifies finite individual/total losses, finite gradients with a nonzero backbone/head signal, public prediction shapes/types/ranges, and coordinate restoration from internal resized boxes. Every model parameter and buffer is compared exactly before/after; no optimizer is constructed or stepped. The head remains untrained. These are integration checks, not training experiments or performance measurements. No test data is loaded and no AP is calculated.

Images remain original-size RGB float32 in `[0, 1]` until entering the detector. Historical integration checks and the first two completed runs used mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, `min_size=(800,)`, `max_size=1333`, aspect-preserving bilinear interpolation (`align_corners=False`), and batch padding to multiples of 32. Their checked `720 x 1280` images became `749 x 1333`, padded to `768 x 1344`. Those historical results must remain associated with that preprocessing. Do not interpret weight metadata's `min_size=(1, 1)` as the detector resize setting.

The historical 640 experiment called `build_model(image_size=640)` and used `SquarePadTransform` for a fixed square canvas. Its single normalization, aspect-preserving bilinear resize, actual-content box scaling, top-left alignment and right/bottom normalized-zero padding are preserved in code and records for reproduction. Its 720 x 1280 inputs became 360 x 640 content with 280 bottom-padding pixels. These helpers are no longer active in normal training. Dataset resizing, direct stretching and duplicate normalization remain forbidden.

Current normal training passes `IMAGE_SIZE=None` to `build_model`, retaining the factory `GeneralizedRCNNTransform` and native `MultiScaleRoIAlign` without either square adapter. Mean/std remain unchanged; the short edge targets 800 with a 1333 long-edge limit, bilinear interpolation and native padding to multiples of 32. A synthetic 720 x 1280 image produces 749 x 1333 content and 768 x 1344 padded input in the pinned runtime. Unpadded content sizes drive clipping and native postprocess back to original pixel xyxy, then the adapter exports COCO xywh with original IDs. Reports/checkpoints retain actual preprocessing identity. The optional historical `SquareCanvasRoIAlign` remains available only with an explicit square size.

Main comparisons can use each detector's documented official input rule while sharing fixed splits/classes, evaluator, condition definitions and validation-based checkpoint selection. YOLO letterboxing and RT-DETRv2 direct Resize already differ despite nominal 640 targets. Record all effective input rules, training recipes and budgets; this compares configured systems, not architecture alone or equal compute budgets. The return to 800/1333 was chosen after viewing the lower 640 result and must be reported transparently. Preserve the 640 run as supplementary evidence; different five-/ten-epoch budgets mean the existing comparison is not a strict resolution-only ablation. Official defaults are a starting point, not a guarantee of the best BDD100K setting. Normalization, backbone trainability and inference thresholds remain unchanged.

Retain the pretrained factory defaults for this integration stage: FrozenBatchNorm2d, trainable backbone body stages `layer2`/`layer3`/`layer4`, inference score threshold 0.05, NMS IoU 0.5, and maximum 100 detections per image. These settings have not been tuned. The check uses seed 0, batch size 2, no shuffling, no workers, float32, and no AMP; these are checking settings rather than selected training hyperparameters, and deterministic algorithms are not enforced.

The separately authorized bounded training/checkpoint stage is also complete. `src/check_training.py` reuses the original model and loader without changing `src/check_model.py`. It performs real SGD updates, checks finite losses/gradients/parameters, verifies parameter changes and frozen-parameter preservation, saves model/optimizer/RNG/config/data-position state, then rebuilds and restores before the final update. Defaults in `configs/training_check.json` are seed 0, three two-image steps, SGD learning rate 0.001, momentum 0.9, and weight decay 0.0005. Use only the first six fixed training images in annotation order, no shuffling or workers, float32, no AMP/scheduler/augmentation. These are process-check settings, not a tuned baseline configuration.

Store all check runs under unique `outputs/training_check/<timestamp>/` directories. Save `state_dict` tensors and basic Python values, load with `weights_only=True`, and restore CPU/CUDA RNG after model and DataLoader iterator construction. Model state, SGD momentum/hyperparameters/devices, category mapping, transform settings, configuration, batch position, and RNG restoration were checked, followed by one successful continued update. This covers the current sequential single-worker check only; arbitrary shuffled/multi-worker resume and bitwise agreement with an uninterrupted trajectory have not been verified. Checkpoints contain a few real updates, not a trained or validation-selected baseline. `build_model()` still creates a fresh official-weight model with a new predictor on each call and does not load these checkpoints automatically.

The user requested a readability refactor after the initial epoch-driver implementation. The primary implementation consists of `src/train.py` and `src/evaluate.py`, alongside Dataset/model modules that were unchanged during that historical refactor. The historical 640 preparation changed the model transform/pooler while leaving Dataset unchanged; normal training now selects the native transform/pooler again. `training.py` and `evaluation.py` have been merged into the primary files; the external baseline JSON configuration has been removed. The original bounded check keeps its independent SGD constructor. Older code/experiment records below or in status describe the earlier implementation, not the current command interface.

Current settings prepare a fresh default-input ten-epoch run: `IMAGE_SIZE=None`, seed 42, train/val batch size 2, SGD LR 0.005, momentum 0.9, weight decay 0.0005, CUDA device 0 and print interval 50. Loaders use workers 0, training shuffles and retains partial batches; float32, no scheduler/warmup/AMP/augmentation. Only the input rule is restored, not an entire official training recipe. The prepared run has not started; metrics and duration are `Not measured`. A future user-run command without `--resume` creates a separate directory from cached COCO_V1 and a new 11-class head. Do not restore any old checkpoint automatically. Both missing legacy transform metadata and explicit 640 transform settings remain incompatible under the existing resume guard. All completed artifacts and saving/selection rules are preserved. The assistant must not execute or click training unless explicitly instructed to execute training. No cloud resources or paid tasks are authorized.

`train_one_epoch()` visibly contains image/target transfer, zero_grad, model forward, finite-total-loss check, backward, and optimizer step. `main()` creates loaders/model/SGD, optionally restores a boundary, and repeats full-epoch training, complete validation, and saving. Support helpers are below this flow. Keep basic terminal progress and native error tracebacks. Remove detailed per-step logging, custom report lifecycles, code/data hashes, environment snapshots and strict environment matching. Generated artifacts remain under a unique `outputs/training/<timestamp>/`: latest/best checkpoints, one-row-per-epoch `history.csv`, and latest `validation.json`/`predictions.json`. JSON validation/prediction files are overwritten each epoch; their report contains the evaluated epoch. Checkpoint/history state is the recovery authority if auxiliary-file writing is interrupted.

`src/evaluate.py` handles eval/inference mode, original-ID output, in-memory xyxy-to-xywh conversion and complete coverage, then delegates scoring to root-level `evaluate_team.py`. Record all 12 summary values and per-class AP/AP50/AP75 plus GT counts, evaluator IoU/recall/area/maxDets settings and actual detector settings. Use `Not evaluable` for absent-GT/other undefined metrics, excluded from the COCO mean. Keep score/NMS/100-detection defaults with no extra score filter. Passing explicit image IDs is reserved for sampled functionality checks and marks the report as non-performance. The adapter has no separate CLI. The supplied team script scores exported COCO predictions and provides its own CLI for full validation and fixed condition subsets. Its test option is reserved for final evaluation after choices are fixed; it is not invoked by training. Condition analysis and final test remain unexecuted.

Save latest atomically after a full training-and-validation epoch; update best only on improved complete-validation bbox mAP, retaining the earlier epoch on ties. Future continuation of the same new experiment reuses its directory via `--resume <run>/latest.pth` after increasing EPOCHS. Checkpoints contain model/SGD, epoch/global step, best mAP/epoch, settings/categories, compact history and Python/NumPy/Torch CPU/all-CUDA RNG. Settings now include workers and full preprocessing identity; reject missing/different transforms, including both historical 800/1333 experiments. Restore RNG after constructors, before the next shuffled iterator. If latest itself was best, it can reconstruct best after an interrupted best write; otherwise keep the existing earlier best. The format remains `bdd100k_simple_training_v1`, subject to the new compatibility check; historical artifacts are untouched. Resume expects unchanged data/code/environment/settings and workers 0; settings/categories/step count are checked, but content hashes and environment identity are not. Partial epochs replay; batch/distributed/multi-worker and cross-hardware/version bitwise continuation are unsupported. Deterministic algorithms are not forced.

The historical refactor passed nine synthetic tests and a bounded real-data GPU check. The user-launched one-epoch run measured full-validation mAP 0.20033579847015223. The independent five-epoch run completed 17,500 updates with mAP values 0.1834592123086974, 0.22445889409023706, 0.23896501467662432, 0.25544452083836444 and 0.23822375438048712. Its best checkpoint is epoch 4 / step 14,000 (AP50 0.5170826995660548, AP75 0.21488694482832407); latest checkpoint and validation/prediction JSON belong to epoch 5 / step 17,500. Epoch 4 per-class/size/recall metrics and predictions were not retained separately; never relabel epoch 5 data as best results. CPU-only checkpoint metadata, histories and saved reports were checked without new model computation. See `docs/EXPERIMENT_LOG.md` for actual results and limitations.

The historical 640 preparation passed 32 synthetic tests, and the user completed ten epochs / 35,000 updates in `20261021_002448_944703`. Its epoch-4 best mAP/AP50/AP75 are 0.17773109411680846 / 0.3619035810493331 / 0.15257382817657714; latest/JSON are epoch 10 (mAP 0.17702375663177553). Epoch-4 details/predictions were not retained separately. The best is 7.7713426721555985 percentage points below the old five-epoch best. Several later epochs without a new best support a validation-plateau interpretation; they do not identify a cause or prove strict convergence. Exact start/end/duration remain unmeasured. The current restoration stage passed 34 synthetic tests, compilation and training CLI help, without real detector construction/training/inference or scoring. Future default-input results are unmeasured. Condition analysis for this Faster R-CNN remains unexecuted and test is `Not evaluated`. Preserve all source data, environment and artifacts; no cloud migration or report PDF edits are included.

Every user-facing chat response must use Simplified Chinese. All new or edited project-owned content must use English. Git commits and pushes require explicit user confirmation.

The shared evaluator preserves original `images`, `objects`, `mAP`, `AP50`, `per_class`, five condition subsets and `clear_daytime mAP - subset mAP` definition. Additive AP75/size/recall/per-class AP50/AP75 fields do not change existing metric values. Shared JSON uses null for non-evaluable metrics and excludes absent classes from its mean; the training adapter renders null as `Not evaluable`. Only model-specific inference/output adaptation remains in `src/evaluate.py`. Keep the existing model score 0.05/NMS 0.5/max 100 unchanged and recorded; the shared evaluator adds no score filter. The group must agree prediction-export rules across models. Matching a scoring script alone does not establish matching model-side filtering. Compatibility fixes isolate pycocotools input mutation, remove external runtime-path injection and use explicit UTF-8 without changing source dataset files.

Shared evaluator integration was verified with 18 synthetic tests and archived predictions for two validation images; all 12 metrics matched the previous COCOeval report within 1e-14. No training, new image inference, complete validation, real condition analysis or test evaluation was performed. Evidence is under `outputs/team_evaluator_check/20261020_152251_464791/`. Earlier experiment/check records retain their actual provisional-evaluator history.
