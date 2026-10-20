# Improving Robustness of Object Detection under Diverse Driving Conditions

Faster R-CNN baseline for a university Deep Learning group project using a prepared BDD100K 10k subset. This repository covers baseline training and evaluation; other group members maintain the dataset preparation and shared evaluation tooling.

**Current stage:** Dataset/DataLoader, pretrained Faster R-CNN integration, and a bounded training/checkpoint check are implemented. Three two-image GPU batches completed real SGD updates, including one step after checkpoint restoration. Full baseline training and AP evaluation have not been performed; model performance remains unmeasured.

## Project Layout

```text
bdd100k-faster-rcnn/
|-- AGENTS.md                  # Working rules and session continuity
|-- README.md                  # Project overview and usage status
|-- docs/
|   |-- PROJECT_CONTEXT.md     # Dataset contract and technical decisions
|   |-- CURRENT_STATUS.md      # Verified progress, issues, and next steps
|   |-- EXPERIMENT_LOG.md      # Future experiment configurations and results
|   `-- ENVIRONMENT.md         # Setup commands and verified environment checks
|-- requirements.txt           # Direct dependency pins
|-- requirements-lock.txt      # Resolved versions from the tested environment
|-- .venv/                     # Local Python environment; ignored by Git
|-- src/
|   |-- __init__.py            # Python package
|   |-- dataset.py             # COCO Dataset, detection collation, and DataLoader
|   |-- check_data.py          # Training/validation data-loading smoke check
|   |-- model.py               # Official pretrained Faster R-CNN with an 11-class head
|   |-- check_model.py         # GPU loss/backward/inference check; no optimizer step
|   `-- check_training.py      # Bounded SGD updates and checkpoint reload check
|-- configs/
|   `-- training_check.json    # Small training-check settings; not a tuned baseline
|-- data/                      # Local dataset; ignored by Git
|   `-- bdd100k_subset_10k/
|-- outputs/                   # Local checks, weight cache, and future artifacts; ignored
`-- .gitignore
```

The ignored `.venv/`, `data/`, and `outputs/` directories are local and will not appear in a fresh clone. Recreate the environment using the documented setup commands only when it does not already exist, and keep the supplied dataset at the path shown above. Check commands create their own output directories. `training_check.json` contains checking settings; a formal baseline training configuration remains to be defined.

## Dataset

Read the supplied `data/bdd100k_subset_10k/README.md` before working with the data. It has already been cleaned, sampled, converted, and split; do not repeat preprocessing or change the files, splits, or category mapping.

| Split | Images | Valid boxes | Image directory | COCO annotations |
|---|---:|---:|---|---|
| Training | 7,000 | 128,949 | `images/train/` | `annotations/train.json` |
| Validation | 1,000 | 18,465 | `images/val/` | `annotations/val.json` |
| Test | 2,000 | 36,951 | `images/test/` | `annotations/test.json` |

Paths in this table are relative to `data/bdd100k_subset_10k/`. There are ten foreground categories, with COCO IDs 1 through 10. The full mapping, annotation conventions, and six validation condition subsets are documented in [Project Context](docs/PROJECT_CONTEXT.md).

## Planned Training and Evaluation

- Use PyTorch and Torchvision with a pretrained Faster R-CNN ResNet50-FPN model.
- Fine-tune on training images and select the model and hyperparameters using validation results.
- Provisionally evaluate with standard COCOeval, reporting COCO mAP, AP50, AP75, and appropriate recall, size, and per-class metrics.
- Compare validation performance on clear daytime, night, dawn/dusk, rainy, and overcast subsets, alongside the full validation set.
- Use the test set for final evaluation only after the model and hyperparameters are fixed.

The group's shared evaluation script has not been supplied. Existing dataset reports are data preparation or evaluator validation records, not baseline model results.

## Setup and Running Status

The project's isolated `.venv` uses Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, NumPy 2.5.3, Pillow 12.3.0, and pycocotools 2.0.11. Global packages were not changed. CPU/CUDA tensor operations, NMS, ROIAlign, a small CUDA convolution, image conversion, and synthetic COCOeval checks passed on the RTX 5070.

See [Environment Setup and Verification](docs/ENVIRONMENT.md) for installation order, verification scope, and how to select the project interpreter. [requirements.txt](requirements.txt) pins direct dependencies; [requirements-lock.txt](requirements-lock.txt) records the 15 installed package versions, including transitive dependencies and packaging tools. Follow the two-step Torch-index installation order when recreating the environment.

The bounded training check below verifies a few real parameter updates and checkpoint save/reload. There is no full-epoch training driver or evaluation script yet. These checks do not establish long-run training stability, a suitable baseline configuration, or evaluation correctness.

Before continuing in a new session, read [Working Rules](AGENTS.md), [Project Context](docs/PROJECT_CONTEXT.md), and [Current Status](docs/CURRENT_STATUS.md). Consult [Experiment Log](docs/EXPERIMENT_LOG.md) for actual run history.

## Data Loading and Checks

Use the existing environment from the repository root:

```powershell
& .\.venv\Scripts\python.exe -m src.check_data --save-previews
```

The check reads four evenly spaced images from each fixed training/validation split. It compares RGB pixels and each target against source records, checks batch loading, and optionally saves at most two annotated images per split under `outputs/data_check/`. The selected samples are only a smoke check; they do not redefine the dataset splits or perform a full image-integrity audit. No model is constructed, no weights are downloaded, and no test data or model metrics are used.

To check Windows worker processes and a final partial batch:

```powershell
& .\.venv\Scripts\python.exe -m src.check_data --num-workers 2 --batch-size 3
```

Options include `--data-root <path>`, `--samples-per-split 4`, `--batch-size 2`, `--num-workers 0`, and `--save-previews`. The dataset root defaults to this repository's `data/bdd100k_subset_10k/`; keep local path overrides outside `data/`. The small batch/worker defaults are for checking, not selected training hyperparameters.

`BDD100KDataset` reads `annotations/<split>.json` with pycocotools and the matching `images/<split>/` files with Pillow. Only `train` and `val` are accepted at this stage. Images become RGB `float32` tensors in `[0, 1]`, shaped `[3, H, W]`, without resizing, normalization by channel statistics, or augmentation. Targets are dictionaries:

| Field | Shape | Type | Meaning |
|---|---|---|---|
| `boxes` | `[N, 4]` | `float32` | Absolute pixel `xyxy`, converted from COCO `xywh` in memory |
| `labels` | `[N]` | `int64` | Original COCO category IDs 1-10; 0 is reserved for background |
| `image_id` | `[1]` | `int64` | Original COCO image ID, never the dataset index |
| `area` | `[N]` | `float32` | Supplied COCO area, in square pixels |
| `iscrowd` | `[N]` | `int64` | Supplied COCO crowd flag |

The class mapping is checked against `class_mapping.json`, including category order. Fractional coordinates and empty-target shapes are preserved. Boxes are not rounded, clipped, filtered, or rewritten. The source JSON stays unchanged.

`collate_fn` returns `(list_of_images, list_of_targets)`, retaining each image's own object count. `make_dataloader` uses this function, accepts explicit batch size/shuffling/worker options, and keeps the final partial batch. A simple loading example is:

```python
from src.dataset import BDD100KDataset, make_dataloader

dataset = BDD100KDataset("data/bdd100k_subset_10k", "train")
loader = make_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0)
images, targets = next(iter(loader))
```

For custom scripts with `num_workers > 0` on Windows, create and iterate loaders inside a function called under `if __name__ == "__main__":`. The supplied check commands already have this guard. Data loading stays on CPU; `check_model` explicitly transfers its image and target lists to the GPU.

## Model Integration and GPU Check

From the repository root, use the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m src.check_model
```

CUDA is required. The command uses the first and last image of each fixed training/validation split, in one two-image batch per split, with `shuffle=False`, `num_workers=0`, float32, no AMP, and check seed 0. These are smoke-check settings, not selected training hyperparameters. Override only the dataset location with `--data-root <path>` when needed; keep configuration outside `data/`.

`src.model.build_model()` loads the complete official `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` detection weights, then replaces the predictor with a new `FastRCNNPredictor` for 11 classes. Dataset IDs 1-10 remain unchanged and background is 0; no pretrained COCO category mapping is reused. The new head is randomly initialized and has not been fine-tuned. The pretrained factory's default backbone behavior is retained: `layer2`, `layer3`, and `layer4` are trainable, while the earlier body stages are frozen. This has not been selected through training experiments.

The first check downloads approximately 160 MiB from the [official PyTorch weight URL](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) into `outputs/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`; subsequent checks reuse this cache. The factory sets Torch Hub's process-wide cache directory inside ignored `outputs/`. The report records the exact weight enum, source URL, size, and SHA-256. No extra ImageNet backbone download is needed. See the [official model interface](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) and [head-replacement tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

The existing loader returns `images` as a list of RGB `[3, H, W]` float32 tensors in `[0, 1]`, and `targets` as a matching list of dictionaries. The check moves both lists' tensors to CUDA. In training mode, `model(images, targets)` returns four scalar losses: classifier, box regression, RPN objectness, and RPN box regression. The check verifies each loss and their sum, calls `backward()` once, checks finite/nonzero gradients, and clears them. It creates no optimizer and performs no parameter update.

In `model.eval()` with `torch.inference_mode()`, `model(images)` returns a list of predictions, one per image: `boxes` float32 `[N, 4]` in original input pixel `xyxy`, `labels` int64 `[N]` using IDs 1-10, and `scores` float32 `[N]`. Validation targets are loaded for sample metadata but are not passed to inference or used for AP. The check verifies shapes, types, device, finite values, ranges, and the exact coordinate scaling from internal resized boxes to public outputs. Empty predictions are valid. It also compares every model parameter and buffer against its pre-forward value exactly.

Keep normalization and resizing inside the model; do not call a second normalization/resize transform in the Dataset. Retained Torchvision defaults are mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, `min_size=(800,)`, `max_size=1333`, bilinear resize with aspect ratio preserved, and zero padding to multiples of 32. On the checked `720 x 1280` inputs, actual resized dimensions are `749 x 1333`, and the padded batch shape is `[2, 3, 768, 1344]`. Inference defaults remain score threshold 0.05, NMS IoU 0.5, and at most 100 detections per image; none were tuned.

The command overwrites `outputs/model_check/report.json` with the current attempt's status and, on success, configuration and results. The verified run produced finite total loss `3.7880015373`, 72 finite/nonzero gradient tensors, and correctly shaped predictions for both validation images. These losses and predictions do not measure model performance. AP is `Not measured`; test evaluation is `Not evaluated`. No trained checkpoint or prediction file is saved. See [Current Status](docs/CURRENT_STATUS.md) for full verification evidence and remaining work.

## Bounded Training and Checkpoint Check

Use the existing environment from the repository root:

```powershell
& .\.venv\Scripts\python.exe -m src.check_training
```

This command performs real parameter updates. Default settings are in `configs/training_check.json`: seed 0, batch size 2, three steps, and SGD with learning rate 0.001, momentum 0.9, and weight decay 0.0005. Only the first six annotation-order training images are used, with `shuffle=False`, `num_workers=0`, float32, no AMP, no scheduler, and no augmentation. Model resizing and normalization stay unchanged. These settings were chosen to verify the process, not selected using validation performance.

Each step clears gradients, computes and checks the four losses, runs backward, verifies gradients, and calls `optimizer.step()`. The check confirms that backbone and replacement-head parameters actually change while frozen parameters stay unchanged. After step 2, it saves a checkpoint, rebuilds the model and optimizer, restores model/SGD momentum/CPU and CUDA RNG state, and completes step 3 on the next two images. It compares the saved/loaded states exactly and verifies the final checkpoint against its live model and optimizer.

Each invocation creates a new `outputs/training_check/<timestamp>/` directory with `config.json`, per-step `steps.jsonl`, a `report.json` containing running/passed/failed/interrupted status, `checkpoint_before_resume.pth`, and `checkpoint_final.pth`. Existing model-check reports and cached official weights are preserved. The verified default run completed three updates with losses approximately `4.053978`, `3.588855`, and `3.706026`. Different batches contain different images; these values do not establish convergence or model performance.

Optional `--config <path>` and `--data-root <path>` keep local settings outside `data/`. The check is bounded to 2-10 steps and batch sizes 1-2, with exactly one step after checkpoint restoration. It does not implement arbitrary shuffled/multi-worker resume, compare an uninterrupted training trajectory, evaluate validation/test data, select a checkpoint, or produce prediction images. The checkpoints contain only a few updates and are not a trained baseline. A fresh `build_model()` still initializes from official weights and a new head; it does not automatically load either check checkpoint.

Checkpoint handling uses PyTorch `state_dict` APIs and `torch.load(..., weights_only=True)`; see the [official save/load tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). See [Experiment Log](docs/EXPERIMENT_LOG.md) and [Current Status](docs/CURRENT_STATUS.md) for the actual run, artifacts, and remaining work.

## Git Hygiene

`data/`, `outputs/`, model weights, Python caches, virtual environments, local secrets, and temporary files are ignored. Verify locally from the repository root:

```powershell
git status --short --untracked-files=all
git check-ignore -v data/bdd100k_subset_10k/ outputs/
git ls-files -- data outputs .venv
```

The final command should print nothing. Do not force-add ignored files. Every commit and push requires explicit user confirmation.
