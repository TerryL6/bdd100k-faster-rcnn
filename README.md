# Improving Robustness of Object Detection under Diverse Driving Conditions

Faster R-CNN baseline for a university Deep Learning group project using a prepared BDD100K 10k subset. This repository covers baseline training and evaluation; other group members maintain the dataset preparation and shared evaluation tooling.

**Current stage:** fixed training/validation Dataset and DataLoader implemented and checked with a small sample. No model, training, or evaluation implementation exists yet, and no model results have been measured.

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
|   `-- check_data.py          # Training/validation data-loading smoke check
|-- configs/                   # Reserved for future experiment configuration
|-- data/                      # Local dataset; ignored by Git
|   `-- bdd100k_subset_10k/
|-- outputs/                   # Local check previews and future run artifacts; ignored
`-- .gitignore
```

`configs/` still contains only a `.gitkeep` placeholder. The ignored `.venv/`, `data/`, and `outputs/` directories are local and will not appear in a fresh clone. Recreate the environment using the documented setup commands only when it does not already exist, and keep the supplied dataset at the path shown above. The check command creates its output directory when previews are requested.

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

There is no training script, evaluation script, or runnable experiment configuration yet. The successful operator checks do not establish that the complete detection/evaluation pipeline works.

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

For custom scripts with `num_workers > 0` on Windows, create and iterate loaders inside a function called under `if __name__ == "__main__":`. The supplied check command already has this guard. Data loading stays on CPU; GPU transfer and all model/training choices belong to later stages.

## Git Hygiene

`data/`, `outputs/`, model weights, Python caches, virtual environments, local secrets, and temporary files are ignored. Verify locally from the repository root:

```powershell
git status --short --untracked-files=all
git check-ignore -v data/bdd100k_subset_10k/ outputs/
git ls-files -- data outputs .venv
```

The final command should print nothing. Do not force-add ignored files. Every commit and push requires explicit user confirmation.
