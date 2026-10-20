# Current Status

Last updated: 2026-10-19.

## Current Stage

Project initialization, isolated environment setup, and the Dataset/DataLoader stage are complete. Fixed training/validation sample loading and Windows worker processes have been checked with the existing `.venv`. No detector construction, training implementation, or model evaluation implementation exists yet.

## Completed Initialization Work

- Inspected the initial repository: branch `main`, existing commit `b958867` (`Initial commit`), and a single tracked root `README.md`.
- Created `.gitignore` before inspecting dataset contents. Verified that `data/` and `outputs/` match ignore rules and that no files in either directory are tracked.
- Read the supplied dataset README, class mapping, conversion summary, and condition count report without changing the dataset.
- Created `AGENTS.md` with language, session continuity, dataset integrity, validation/test separation, and Git confirmation rules.
- Expanded the root README and created project context, current status, and experiment log documents in English.
- Created `src/` and `configs/` with empty `.gitkeep` placeholders, plus an empty ignored local `outputs/` directory.
- Completed read-only checks of actual JPG counts and COCO records: train 7,000 images / 128,949 boxes, val 1,000 / 18,465, and test 2,000 / 36,951.
- Verified that all referenced images exist; split manifests match the COCO image sets; image IDs, annotation IDs, and image filenames are globally unique; category references, positive box dimensions, box areas, and `iscrowd=0` are consistent.
- Verified all ten categories against `class_mapping.json` and confirmed zero railway-train annotations in validation.
- Verified all six validation subset JSON/TXT pairs, exact weather/time membership, retained IDs/categories/boxes, and complete annotations for selected images. Counts match the dataset README.
- Confirmed that the required files/directories exist, all project-owned text is English, and no trailing whitespace was found. Independent documentation review found no actionable issues.
- Final Git checks passed: ignore rules cover dataset files, output artifacts, model weights, caches, environments, local secrets, and temporary files; no data/output files are tracked; no changes are staged; `git diff --check` passes.

## Completed Environment Inspection and Planning

- Inspected Windows 11 x64, an AMD Ryzen 7 9850X3D CPU, 31.19 GiB system memory, an RTX 5070 with approximately 12 GB VRAM, and NVIDIA driver 617.42.
- At inspection time, identified standard CPython 3.13.15 and existing global Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, NumPy 2.5.3, and Pillow 12.3.0. The project `.venv` had not yet been created, and global `pycocotools` was absent.
- Verified global CPU/CUDA matrix multiplication, NMS, ROIAlign, and a small CUDA convolution with synthetic tensors. CUDA is available, the compiled architectures include `sm_120`, and the Torch runtime is CUDA 13.0. Global `pip check` passed.
- Confirmed official Windows wheel availability for the selected Torch pair, NumPy, Pillow, and pycocotools 2.0.11. The standard interpreter accepts the COCO package's `cp312-abi3-win_amd64` tag.
- Added `requirements.txt` with direct dependency pins and `docs/ENVIRONMENT.md` with host observations, verification limits, official references, and proposed isolated setup commands.
- Selected a clean local `.venv` using the existing Python. Install the Torch pair from its official CUDA 13.0 index first, then install the remaining requirements from PyPI. No packages were installed, upgraded, removed, or downloaded for installation during this inspection.
- Updated the root README and project context with the selected environment plan. No training hyperparameters have been selected.
- Final checks passed: all project-owned text remains English, direct dependency pins parse and match the four existing packages, local documentation links resolve, and no environment/code/output artifacts were created. Independent read-only review found no actionable issues. Git ignore, tracked-artifact, staging, and whitespace checks passed.

## Completed Isolated Environment Setup

- Created `.venv` using existing CPython 3.13.15 with `include-system-site-packages = false`. Confirmed the interpreter and selected package locations are inside this environment.
- Installed Torch 2.14.0+cu130 and Torchvision 0.29.0+cu130 from the official CUDA 13.0 wheel index, then installed the exact remaining direct dependencies from PyPI: NumPy 2.5.3, Pillow 12.3.0, and pycocotools 2.0.11. Project pip is 26.2.1.
- `pip check` passed inside `.venv`. Added `requirements-lock.txt` containing all 15 installed distribution versions from `pip freeze --all`, including transitive dependencies and packaging tools.
- Verified CPU/CUDA matrix multiplication, NMS, ROIAlign, a small CUDA convolution, and an in-memory Pillow/NumPy/Torch image roundtrip in `.venv`. CUDA is available on RTX 5070; runtime 13.0 and `sm_120` support were confirmed.
- Verified COCO/COCOeval imports and pure in-memory synthetic bbox evaluations: exact detections produce AP approximately 1 and disjoint detections produce AP 0. These are evaluator functionality checks, not BDD100K model results; no experiment entry was added.
- Rechecked global package versions: existing Torch, Torchvision, NumPy, Pillow, and pip remain unchanged; global pycocotools is still absent.
- Measured environment file lengths at approximately 3.05 GiB, excluding shared pip cache and filesystem allocation overhead. No cache cleanup or global package removal was performed.
- Updated environment documentation, the root README, and project context. No dataset image access, source-data edits, pretrained-weight downloads, detector construction, or training occurred during this stage.
- Final consistency and independent read-only review passed: direct pins and the full version snapshot match `.venv`, local documentation links resolve, project-owned text is English, no source/output artifacts were added, Git whitespace checks pass, no changes are staged, and data/output/environment directories are ignored and untracked.

## Repository and Setup Publication

- Branch: `main`; remote: `https://github.com/TerryL6/bdd100k-faster-rcnn.git`.
- The user explicitly confirmed one setup milestone commit and push to `origin/main` on 2026-10-19.
- Setup milestone commit message: `Initialize project structure and verified Python environment`.
- The reviewed milestone contains 11 small documentation, dependency, ignore-rule, and placeholder files. Before publication, local and remote `main` both pointed to `b958867` (`Initial commit`).
- The dataset, local outputs directory, and `.venv` are ignored. No dataset files were deleted, moved, or modified.
- This document accompanies the setup milestone. Read `git log -1`, `git status --short --branch`, and the remote branch ref for its actual commit identity and publication state; the file does not predict a successful push or embed its own commit hash.
- Further commits and pushes still require explicit user confirmation.

## Not Started

- Training/evaluation configuration and model integration.
- Faster R-CNN construction, pretrained-weight download, training, checkpoint creation, or inference.
- Validation metrics, condition analysis, hyperparameter tuning, or final test evaluation.

## Known Issues and Limitations

- The group's shared evaluation script has not been supplied; standard COCOeval is the provisional plan.
- Isolated GPU operators, synthetic COCOeval, and sampled CPU data loading pass, but the detector pipeline, checkpoint handling, and training memory requirements remain unverified. The version snapshot has no artifact hashes and targets the verified Windows/Python/CUDA combination.
- Validation has no railway-train objects; that class's AP is not evaluable. Condition subsets overlap and have different sample and class distributions.
- Data-loading checks decoded eight distinct training/validation images, not the full dataset. No full image-decoding, content-hash, or near-duplicate/video-leakage audit was performed. Environment-stage checks used synthetic inputs; the current stage used real samples without model evaluation.
- There are no baseline model results. Dataset preparation reports must not be described as model performance.

## Decisions to Preserve

- Use the supplied cleaned COCO labels and fixed splits/category IDs without modifying the dataset.
- The Dataset accepts only `train` and `val` in this stage, checks category order against `class_mapping.json`, returns original-size RGB float32 images in `[0, 1]`, and converts boxes to float32 pixel `xyxy` in memory. Keep original IDs, area, and crowd flags; do not round, clip, filter, or remap source boxes/categories.
- Detection collation returns two lists; keep partial final batches. Loader check defaults do not establish training hyperparameters. Use the module entry point with its main guard for Windows multiprocessing.
- Plan a PyTorch/Torchvision pretrained Faster R-CNN ResNet50-FPN baseline.
- Use the verified Python 3.13 `.venv` and Torch 2.14.0+cu130 / Torchvision 0.29.0+cu130 pair for project execution; preserve the setup order and dependency pins documented in `docs/ENVIRONMENT.md`, `requirements.txt`, and `requirements-lock.txt`.
- Use validation for selection and tuning; reserve test performance for final evaluation after choices are fixed.
- Keep data and generated artifacts local and ignored. Require explicit user confirmation for every commit and push.
- Use Simplified Chinese in chat and English in all project-owned content.

## Completed Dataset and DataLoader Stage

- Rechecked the actual working tree and required project/dataset documentation before editing. The starting tree was clean, and both local `main` and the local `origin/main` ref pointed to setup commit `690523b7fea3a687d50802dee878a755f7036494`.
- Reused `.venv/Scripts/python.exe`; did not create an environment, install packages, or change dependency pins.
- Added `src/__init__.py`, `src/dataset.py`, and `src/check_data.py`; removed the now-unneeded empty `src/.gitkeep`. The `configs/.gitkeep` placeholder remains.
- `BDD100KDataset` reads the fixed cleaned training/validation COCO annotations using pycocotools and RGB images using Pillow/Torchvision. It returns `boxes`, `labels`, `image_id`, `area`, and `iscrowd` with the documented types/shapes and preserves source IDs and category mapping.
- Added the top-level detection `collate_fn` and `make_dataloader`, keeping per-image variable-length targets and the final partial batch. No resizing, augmentation, model preprocessing, or GPU transfer was added.
- Ran `python -m src.check_data --save-previews` with the existing environment. Four evenly spaced images per split passed exact RGB tensor comparisons, dimensions/types/range checks, independent source-JSON box conversion comparisons, category/image IDs, area/crowd preservation, and batch checks. Training IDs/counts: `2001:12`, `4334:6`, `6667:32`, `9000:9`; validation IDs/counts: `9001:17`, `9334:25`, `9667:17`, `10000:6`. These eight distinct images contain 124 boxes; repeated reads are not additional unique samples.
- Default batches were `[2, 2]` per split; an additional one-sample loader verified partial-batch retention. Ran `python -m src.check_data --num-workers 2 --batch-size 3` successfully on Windows; each split produced `[3, 1]` batches with the same image IDs and targets.
- Separately simulated an empty annotation list in memory with `unittest.mock.patch.object`, while reading the first training image. Verified `(0, 4)` boxes, empty one-dimensional labels/area/iscrowd, and collation with a nonempty target. This was a synthetic edge-case check, not an empty real dataset sample. Confirmed that source COCO box values in memory stayed unchanged and that unsupported split names, including `test`, are rejected before data access.
- Saved and visually inspected `train_2001.jpg`, `train_4334.jpg`, `val_9001.jpg`, and `val_9334.jpg` under ignored `outputs/data_check/`. Boxes align with the sampled objects; these are annotation previews, not model detections.
- Compared before/after file paths, sizes, and modification times for all 30,071 dataset files; they were unchanged. SHA-256 matched for the dataset README, class mapping, training/validation annotation JSON, and the eight sampled source images (12 files). Local evidence is in `outputs/data_check/source_snapshot_before.json` and `source_integrity_check.json`. This is not a full-dataset content-hash audit.
- The CLI help and `python -m compileall -q src` passed. Independent read-only implementation review found no actionable issues. Updated the root README with usage, target fields, and Windows worker guidance, and recorded lasting loader decisions in project context.
- No source dataset files, split/category/subset definitions, or existing environment files were modified. No test images or test annotations were loaded, no weights were downloaded, no model was constructed, and no training or model evaluation was performed. Model metrics remain `Not measured`; test evaluation remains `Not evaluated`. No experiment entry was added because this stage was not a model experiment.
- Implementation-stage Git checks confirmed `data/`, `outputs/`, `.venv/`, and Python caches remained ignored and untracked, with no changes staged. No commit or push occurred during implementation; publication was authorized separately afterward as recorded below.

## Dataset and DataLoader Milestone Publication

- On 2026-10-19, the user explicitly authorized one commit and push of the completed Dataset/DataLoader milestone to the existing `origin/main`.
- Before publication, local `main` and the live remote `main` both pointed to setup commit `690523b7fea3a687d50802dee878a755f7036494`.
- Commit message: `Add COCO dataset and detection data loading checks`.
- The milestone includes the Dataset/DataLoader module, sample-check command, Python package entry point, removal of the empty source placeholder, and related README/context/status/environment documentation updates. It does not include dataset files, the local environment, previews, caches, or other generated outputs.
- This document accompanies the milestone. Use `git log -1`, `git status --short --branch`, and the live remote branch ref to verify the resulting commit identity and publication state; this file does not predict a successful push or embed its own commit hash.
- Further development, commits, and pushes require their own applicable user authorization. The user will start the model-integration stage in a new Codex chat using a separate prompt; no model code or weights have been added by this publication step.

## Running the Data Check

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m src.check_data --save-previews
& .\.venv\Scripts\python.exe -m src.check_data --num-workers 2 --batch-size 3
```

The first command checks four samples per split and saves at most two previews per split; omit `--save-previews` for a check without preview writes. The second command checks worker spawning and the final partial batch. See the root README for remaining options and a loading example. Neither command trains or evaluates a model.

## Recommended Next Stage

After explicit user authorization, implement and verify minimal Faster R-CNN model integration against the loader contract. Pretrained-weight downloads, training configuration, the training loop, validation evaluation, and test evaluation have not been authorized by this stage. Preserve the fixed dataset and choose training settings only during subsequent authorized work.
