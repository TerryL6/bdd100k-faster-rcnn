# Current Status

Last updated: 2026-10-19.

## Current Stage

Project initialization, environment inspection, isolated dependency installation, and synthetic functional verification are complete. Final documentation review and repository checks passed. No dataset adapter, training implementation, or model evaluation implementation exists yet.

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

- Dataset adapter, data loading code, or training/evaluation configuration.
- Faster R-CNN construction, pretrained-weight download, training, checkpoint creation, or inference.
- Validation metrics, condition analysis, hyperparameter tuning, or final test evaluation.

## Known Issues and Limitations

- The group's shared evaluation script has not been supplied; standard COCOeval is the provisional plan.
- Isolated GPU operators and synthetic COCOeval pass, but the complete dataset adapter, detector pipeline, checkpoint handling, and training memory requirements remain unverified. The version snapshot has no artifact hashes and targets the verified Windows/Python/CUDA combination.
- Validation has no railway-train objects; that class's AP is not evaluable. Condition subsets overlap and have different sample and class distributions.
- The dataset reports preparation-stage integrity checks, but project work has not repeated image decoding, preprocessing, or a near-duplicate/video-leakage audit. Environment work used synthetic inputs and did not access dataset images or perform model evaluation.
- There are no baseline model results. Dataset preparation reports must not be described as model performance.

## Decisions to Preserve

- Use the supplied cleaned COCO labels and fixed splits/category IDs without modifying the dataset.
- Plan a PyTorch/Torchvision pretrained Faster R-CNN ResNet50-FPN baseline.
- Use the verified Python 3.13 `.venv` and Torch 2.14.0+cu130 / Torchvision 0.29.0+cu130 pair for project execution; preserve the setup order and dependency pins documented in `docs/ENVIRONMENT.md`, `requirements.txt`, and `requirements-lock.txt`.
- Use validation for selection and tuning; reserve test performance for final evaluation after choices are fixed.
- Keep data and generated artifacts local and ignored. Require explicit user confirmation for every commit and push.
- Use Simplified Chinese in chat and English in all project-owned content.

## Recommended Next Stage

Read the supplied dataset README and implement a minimal COCO dataset adapter plus batch collation. Verify training/validation sample loading, image/target pairing, category and image IDs, and in-memory conversion from COCO xywh to model xyxy boxes. Keep source files unchanged and do not train or evaluate on the test set. Model construction and full training remain separate later stages.
