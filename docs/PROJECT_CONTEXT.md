# Project Context

Last updated: 2026-10-19.

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
- Baseline: pretrained Faster R-CNN ResNet50-FPN, fine-tuned for the ten supplied foreground categories. Adapt the prediction head and handle the model's background convention during implementation while preserving source category IDs.
- Keep the implementation small and use existing framework functionality where practical.
- Fine-tune on training only. Use validation for checkpoint selection and justified hyperparameter changes.
- Use standard COCOeval provisionally because the group's shared evaluation script has not been supplied. Reconcile settings and outputs when it arrives.
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

No model construction, training/evaluation code, pretrained weights, or model training runs have been added. Model integration and its verification require a later authorized stage.

The training seed, optimizer, learning rate, scheduler, training batch size, epoch count, model resize/normalization, augmentation, and checkpoint selection details have not been chosen or verified. Resolve these in later authorized stages instead of presenting loader check defaults as completed decisions. Existing GPU operator checks do not prove full-model training compatibility or memory capacity for a particular batch size.

Every user-facing chat response must use Simplified Chinese. All new or edited project-owned content must use English. Git commits and pushes require explicit user confirmation.
