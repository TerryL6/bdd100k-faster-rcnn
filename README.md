# Improving Robustness of Object Detection under Diverse Driving Conditions

Faster R-CNN baseline for a university Deep Learning group project using a prepared BDD100K 10k subset. This repository covers baseline training and evaluation; other group members maintain the dataset preparation and shared evaluation tooling.

**Current stage:** isolated environment installation and synthetic functional checks complete. No training or evaluation implementation exists yet, and no model results have been measured.

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
|-- src/                       # Reserved for future Python implementation
|-- configs/                   # Reserved for future experiment configuration
|-- data/                      # Local dataset; ignored by Git
|   `-- bdd100k_subset_10k/
|-- outputs/                   # Future run artifacts; ignored by Git
`-- .gitignore
```

`src/` and `configs/` contain only `.gitkeep` placeholders. The ignored `.venv/`, `data/`, and `outputs/` directories are local and will not appear in a fresh clone. Recreate the environment using the documented setup commands, keep the supplied dataset at the path shown above, and create `outputs/` locally if needed.

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

The recommended next stage is to implement and verify a minimal COCO dataset adapter and batch loading for the fixed training/validation splits, preserving source IDs and converting box formats in memory. Model and training implementation follow that stage.

## Git Hygiene

`data/`, `outputs/`, model weights, Python caches, virtual environments, local secrets, and temporary files are ignored. Verify locally from the repository root:

```powershell
git status --short --untracked-files=all
git check-ignore -v data/bdd100k_subset_10k/ outputs/
git ls-files -- data outputs .venv
```

The final command should print nothing. Do not force-add ignored files. Every commit and push requires explicit user confirmation.
