# Current Status

Last updated: 2026-10-20.

## Current Stage

The supplied `evaluate_team.py` is now the shared scoring standard. `src/evaluate.py` performs inference/output adaptation and directly delegates metrics to that script. Its original mAP/AP50/per-class protocol is preserved with compatibility fixes and additive metrics. Eighteen tests and an archived two-image prediction comparison passed. Full baseline training and complete-validation performance remain unmeasured.

The temporary inference visualization requested for inspection on 2026-10-20 was subsequently removed at the user's request, including its helper and separate preview/prediction exports. The original model, smoke-check command, and successful summary report are retained unchanged. This stage did not add visualization.

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

## Remaining Work

- Review and explicitly authorize the provisional single-epoch configuration before full execution; these are not tuned hyperparameters.
- Full baseline fine-tuning, validation-based checkpoint selection, and full-dataset inference/evaluation.
- Validation metrics, condition analysis, hyperparameter tuning, or final test evaluation.

## Known Issues and Limitations

- The shared `evaluate_team.py` is now integrated. Prediction-export filtering across different models still needs a group-wide agreement; no full validation or condition analysis has run.
- Isolated operators, sampled loading, detector integration, bounded SGD, and synthetic/real epoch-boundary continuation checks pass. Shuffled single-worker epoch restoration matches an uninterrupted trajectory in the synthetic CPU test. Long-run stability, full-dataset capacity/evaluation, multi-worker resume, arbitrary batch resume, and cross-hardware/version bitwise agreement remain unverified or unsupported. The version snapshot has no artifact hashes and targets the verified Windows/Python/CUDA combination.
- Validation has no railway-train objects; that class's AP is not evaluable. Condition subsets overlap and have different sample and class distributions.
- Earlier data-loading checks decoded eight distinct training/validation images; later model/training checks also used bounded samples. No full image-decoding, content-hash, or near-duplicate/video-leakage audit was performed. Environment-stage checks used synthetic inputs. The current implementation check performed sampled inference and COCOeval only, without full-validation evaluation.
- There are no baseline model performance results. `build_model()` creates a new untrained predictor; check checkpoints contain only bounded updates. Check losses/predictions/sampled AP and dataset preparation reports must not be described as formal model performance. Deterministic algorithms are not enforced. Synthetic CPU continuation matched an uninterrupted trajectory; GPU numerical trajectory equality has not been tested.

## Decisions to Preserve

- Use the supplied cleaned COCO labels and fixed splits/category IDs without modifying the dataset.
- The Dataset accepts only `train` and `val` in this stage, checks category order against `class_mapping.json`, returns original-size RGB float32 images in `[0, 1]`, and converts boxes to float32 pixel `xyxy` in memory. Keep original IDs, area, and crowd flags; do not round, clip, filter, or remap source boxes/categories.
- Detection collation returns two lists; keep partial final batches. Loader check defaults do not establish training hyperparameters. Use the module entry point with its main guard for Windows multiprocessing.
- Use Torchvision `fasterrcnn_resnet50_fpn` with explicit `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, loaded before replacing the head with `FastRCNNPredictor` for IDs 0-10. Store Torch Hub weights under ignored `outputs/torch/hub/`. Leave normalization and resizing inside the detector; see the actual settings below.
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
- Further development, commits, and pushes require their own applicable user authorization. At that publication step, the model-integration stage was deferred to a separate prompt; no model code or weights were added by publication itself. The later authorized stage is recorded below.

## Completed Faster R-CNN Model Integration Stage

- Re-read the required project/dataset documents and inspected actual code, Git status, and history. The starting tree was clean; local `main` and the local `origin/main` ref pointed to `d26a7fbf` (`Add COCO dataset and detection data loading checks`). No remote publication was attempted in this stage.
- Reused the existing `.venv` with Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, and CUDA runtime 13.0 on NVIDIA GeForce RTX 5070. No environment creation, dependency installation, or dependency-pin changes occurred. `pip check` passed, and all 15 package names/versions still match `requirements-lock.txt`.
- Added `src/model.py`: `build_model()` loads the official detection weights first, then replaces the original predictor with an untrained 11-class `FastRCNNPredictor` (classification outputs 11; box-regression outputs 44). Foreground IDs remain 1-10 and background is 0. No pretrained COCO category IDs are reused.
- Added `src/check_model.py`, reusing `BDD100KDataset` and `make_dataloader`. It runs exactly one training-mode forward/backward batch and one validation inference batch; there is no training loop or optimizer. It writes the latest attempt's status and successful results to ignored `outputs/model_check/report.json`, replacing any previous report at that path.
- The actual GPU check completed successfully on 2026-10-19 at 23:42:42 EDT (`2026-10-19T23:42:42.320262-04:00`). Check settings: seed 0, batch size 2, `shuffle=False`, `num_workers=0`, float32, no AMP, and deterministic algorithms disabled. The check uses each split's first/last annotation-order image, not a new data partition. These are checking settings, not selected training hyperparameters.
- Training sample IDs/object counts: `2001:12` and `9000:9`, at indices 0 and 6999. Images were `[3, 720, 1280]`. Image tensors and all target tensors entered CUDA; targets retained their original IDs, categories, and values after the forward pass.
- All four losses were finite differentiable scalars, and their sum was finite. Actual measured values from this smoke check are listed below; they are not performance metrics.

| Smoke-check value | Actual value |
|---|---:|
| `loss_classifier` | 2.728191375732422 |
| `loss_box_reg` | 0.9853302240371704 |
| `loss_objectness` | 0.05408589169383049 |
| `loss_rpn_box_reg` | 0.020394030958414078 |
| Total loss | 3.788001537322998 |
| Finite gradient tensors | 72 |
| Nonzero gradient tensors | 72 |
| Optimizer steps | 0 |
| Peak PyTorch CUDA allocated memory (MiB) | 2230.14453125 |

- Called `backward()` once. Every trainable parameter had a finite gradient, with nonzero gradients in both backbone and replacement predictor. Frozen parameters were not required to have gradients. Gradients were cleared afterward. The peak memory value is PyTorch allocated memory for this check, not total device usage or proof of future training-batch capacity.
- Validation sample IDs/object counts: `9001:17` and `10000:6`, at indices 0 and 999, both `[3, 720, 1280]`. Ran `model.eval()` together with `torch.inference_mode()`, passing images only. Validation targets were used for sample metadata, not passed to the detector or used to calculate AP.
- Each validation image returned 100 detections: `boxes` float32 `[100, 4]`, `labels` int64 `[100]`, `scores` float32 `[100]`, all on CUDA with no gradients. Checked finite values, positive box sizes, original image bounds, labels 1-10, scores in `[0, 1]`, and matching counts. The 100 limit is the unchanged model default, not a selected threshold or evidence of quality. Empty predictions remain a valid supported shape.
- Temporary hooks captured actual resized sizes and RoIHead boxes before postprocessing. Independently scaling those boxes with the actual original/resized H/W ratios matched final `boxes` (`rtol=1e-6`, `atol=1e-4`). Both outputs were nonempty, so original-pixel-coordinate restoration had actual coverage. Hooks were removed after the check.
- Compared every `state_dict` tensor exactly against its pre-forward CPU snapshot after backward and inference. All parameters and persistent buffers remained identical. No `optimizer.step()`, checkpoint creation, AP calculation, hyperparameter tuning, or test image/annotation access occurred.
- CLI help and `python -m compileall -q src` passed. Independent read-only API/implementation review found no actionable correctness or scope issues. No second GPU run was needed after the successful check.
- Compared before/after paths, sizes, and modification times for all 30,071 dataset files: no added, missing, or changed files. SHA-256 matched for eight files: dataset README, class mapping, training/validation JSON, and the four accessed images. Evidence is in ignored `outputs/model_check/source_snapshot_before.json` and `source_integrity_check.json`. Test files were included only in metadata enumeration; their contents were not read. This is not a full-dataset content-hash audit.
- Updated README, project context, environment-stage wording, and the experiment-log status to distinguish integration checks from training experiments. No training experiment entry was added. Formal training, validation/condition metrics, checkpoint selection, and final test evaluation remain unfinished.
- Final checks passed: project-owned text is English, local Markdown links resolve, `git diff --check` reports no whitespace errors, and independent documentation review found no substantive inconsistencies with the actual report. Git status contains only two new source files and five modified documents; nothing is staged. Dataset, `.venv`, outputs, downloaded weights, and Python caches remain ignored and untracked. No commit or push occurred.

### Downloaded Weight Identity

- Exact enum: `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` (the original ResNet50-FPN detector, not the V2 architecture).
- Source: [official PyTorch model file](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth). Interface references: [Torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) and [official predictor-replacement tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Installed package source was also inspected.
- Actual cache: `outputs/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` (167,502,836 bytes).
- SHA-256: `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. The official downloader checked the filename hash prefix on initial download. The full digest was recorded independently after loading.
- `build_model()` sets Torch Hub's process-wide cache directory before constructing the model. Subsequent calls reuse the local cache. `weights_backbone=None` avoids any separate ImageNet weight download. No trained project checkpoint existed at the end of the model-integration stage; later bounded training-check checkpoints are recorded below.

### Actual Model Transform and Defaults

- Dataset preprocessing is unchanged: original-size RGB float32 `[0, 1]`, with target pixel `xyxy` boxes. No duplicate channel normalization, resizing, or augmentation was added.
- Built-in mean: `[0.485, 0.456, 0.406]`; std: `[0.229, 0.224, 0.225]`.
- Built-in `min_size=(800,)`, `max_size=1333`, aspect-preserving bilinear resize with `align_corners=False`, `fixed_size=None`, and zero padding to multiples of 32.
- Both training and inference batches resized `720 x 1280` inputs to actual `749 x 1333` images under the long-edge constraint, then padded the batch to `[2, 3, 768, 1344]`. The 749 value is the actual runtime result, not a rounded expected 750. Target resizing and output coordinate restoration occur only inside the model.
- Retained pretrained-factory behavior: FrozenBatchNorm2d, trainable backbone body stages `layer2`/`layer3`/`layer4`; earlier body stages frozen. Inference score threshold 0.05, NMS IoU 0.5, maximum 100 detections per image. None was tuned using predictions or validation results.

## Completed Bounded Training and Checkpoint Stage (2026-10-20)

- User authorization covered the first proposed next step only: a minimal training loop with a few real updates and checkpoint save/reload verification. Full baseline training, AP evaluation, tuning, and test evaluation were not included. No visualization was added.
- Inspected the actual working tree and required context. Preserved the existing model-integration changes (two new source modules and five modified documents), original source modules, model-check report, and cached weights. Added `src/check_training.py` and `configs/training_check.json`; removed the now-unneeded empty `configs/.gitkeep` placeholder.
- Reused the existing `.venv` and cached `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, with no environment/dependency changes or further weight downloads. The original Dataset, model factory, no-update model checker, and its successful report remained byte-for-byte unchanged.
- The single GPU run `20261020_012930_550530` passed. Start/end: `2026-10-20T01:29:30.550530-04:00` to `2026-10-20T01:29:41.581063-04:00` (EDT); elapsed approximately 11.03 seconds. Environment: Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, CUDA 13.0, RTX 5070.
- Actual checking settings: seed 0, batch size 2, three SGD steps, learning rate 0.001, momentum 0.9, weight decay 0.0005; `shuffle=False`, `num_workers=0`, float32, no AMP, scheduler, or augmentation. Model defaults and category IDs were unchanged. These are process-check settings, not validation-selected baseline hyperparameters.
- Used the first six annotation-order training images (indices 0-5) only. This is a bounded access selection, not a new dataset split. The source split, labels, IDs, category mapping, and condition subsets remain unchanged.

| Step | Training image IDs | Object counts | Total loss | Finite / nonzero gradient tensors | Changed parameter tensors | After reload |
|---|---|---|---:|---:|---:|---|
| 1 | 2001, 2002 | 12, 17 | 4.053978443145752 | 72 / 72 | 72 | No |
| 2 | 2003, 2004 | 25, 10 | 3.5888547897338867 | 72 / 72 | 72 | No |
| 3 | 2005, 2006 | 20, 28 | 3.706026077270508 | 72 / 72 | 72 | Yes |

- All four losses and each total were finite; gradients, updated parameters, and SGD momentum buffers were finite at every step. Backbone and new predictor parameters actually changed at each update; all frozen parameters stayed exactly unchanged. Exactly three `optimizer.step()` calls completed, and zero full epochs completed.
- Saved `checkpoint_before_resume.pth` after step 2, then rebuilt the model and SGD optimizer from scratch. Loaded with `map_location='cpu', weights_only=True`, restored model/optimizer state, and compared every saved/loaded model tensor, optimizer tensor/hyperparameter, model definition/transform, category mapping, configuration, step, and selected image position exactly. Verified momentum buffers were on each parameter's device.
- Rebuilt the remaining sequential DataLoader iterator before restoring CPU/CUDA RNG. Verified RNG restoration exactly, then continued step 3 using IDs 2005/2006. The complete observed ID sequence matched 2001-2006 without repeats or skipped images. Saved the final checkpoint and verified another disk roundtrip against its live model and optimizer.
- The resume conclusion is limited to exact state restoration and successful continuation in this sequential single-worker check. No shuffled/multi-worker/general epoch resume or numerical comparison with an uninterrupted trajectory was performed. Checkpoints include model/optimizer states, CPU/CUDA RNG, step/data position, selected IDs, configuration, model/transform/category metadata, and package versions. No scheduler state exists because no scheduler was used.
- Artifacts: `outputs/training_check/20261020_012930_550530/{config.json,steps.jsonl,report.json,checkpoint_before_resume.pth,checkpoint_final.pth}`. The before-resume/final checkpoints are 330,439,684 / 330,436,364 bytes. Peak PyTorch CUDA allocated memory was 2391.22021484375 MiB; this is not total device memory usage or a capacity guarantee for longer training.
- Independent read-only implementation review found no blocking correctness or scope issues. CLI help and compilation passed. Source-integrity evidence in `outputs/training_check_integrity/` confirms all 30,071 dataset file paths/sizes/modification times were unchanged, all nine selected data-file hashes matched, and the four preserved source/report files matched. Test/validation contents were not read by this stage; only their file metadata was included in the integrity inventory. This is not a full-dataset content-hash audit.
- Recorded this real-update check in `docs/EXPERIMENT_LOG.md`. Different batches contain different images; nonmonotonic losses do not prove either convergence or failure. Validation/condition/class performance is `Not measured`; test evaluation is `Not evaluated`. No checkpoint was selected by performance.
- Final checks passed: report/config/step logs and source hashes agree; oversized step/batch configurations are rejected before model construction; compilation, CLI help, English text, local documentation links, and Git whitespace checks pass. Independent final review found no substantive inconsistency. Dataset, `.venv`, outputs, weights, checkpoints, and Python caches remain ignored and untracked; nothing is staged. The removed prediction visualization remains absent. No commit or push occurred.

## Running the Training Check

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m src.check_training
& .\.venv\Scripts\python.exe -m src.check_training --help
```

This command performs real SGD updates and creates a new timestamped directory under `outputs/training_check/`. Optional `--config <path>` and `--data-root <path>` keep local settings outside the dataset. It is bounded to 2-10 steps and batch sizes 1-2, always with one step after checkpoint restoration. It does not implement full baseline training, validation/test evaluation, or visualization. The original `src.check_model` command below still performs no parameter updates.

## Running the Model Check

From the repository root in PowerShell, with the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m src.check_model
& .\.venv\Scripts\python.exe -m src.check_model --help
& .\.venv\Scripts\python.exe -m compileall -q src
```

The check requires CUDA and downloads/reuses only the pinned official detection weights under `outputs/`. Optional `--data-root <path>` changes the local dataset location without editing dataset configuration. The command performs one backward check with no optimizer step and validation inference without AP; it does not fine-tune or use test data. The new head's losses and predictions do not represent model performance. The latest report is local at `outputs/model_check/report.json` and is overwritten by the next attempt.

## Running the Data Check

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m src.check_data --save-previews
& .\.venv\Scripts\python.exe -m src.check_data --num-workers 2 --batch-size 3
```

The first command checks four samples per split and saves at most two previews per split; omit `--save-previews` for a check without preview writes. The second command checks worker spawning and the final partial batch. See the root README for remaining options and a loading example. Neither command trains or evaluates a model.

## Model Integration and Training Check Milestone Publication

- On 2026-10-20, the user requested publication of the completed work and explicitly confirmed one commit and push to the existing `origin/main`. Publication preparation found local `main` and the live remote `main` at `d26a7fbf478216679d82d3c6e59d7c850661b44a`.
- The prepared milestone contains the model factory, original no-update model check, bounded training/checkpoint check and configuration, related README/context/status/environment/experiment documentation, and removal of the empty configuration placeholder. No formal full-epoch training or evaluation driver is included.
- Commit message: `Add Faster R-CNN integration and bounded training checks`.
- Existing successful reports and source hashes/configuration agree; compilation and Git whitespace checks pass. Independent read-only publication review found no blocking issues. Dataset, environment, weights, check checkpoints, local reports, and caches remain ignored and untracked.
- This document accompanies the authorized milestone; it does not predict a successful commit/push or embed its own commit hash. Use Git history, status, and the live remote ref to verify actual publication state. Further commits and pushes require new explicit confirmation under `AGENTS.md`.
- Formal training implementation and execution remain deferred at the user's request; publishing this milestone does not authorize starting them.

## Initial Epoch-Training Implementation (Superseded by Readability Refactor)

The user authorized implementation of full-epoch training, complete validation COCO evaluation, validation-mAP selection, and epoch-boundary resume on 2026-10-20. Execution was limited to synthetic and small sampled functional checks. Full training and full validation must wait for the user's next confirmation. The starting working tree was clean at `b8f8239d`, matching the local `origin/main` ref. No commit or push occurred.

Added `src/training.py`, `src/train.py`, `src/evaluation.py`, `src/evaluate.py`, `configs/baseline_1epoch.json`, and two synthetic unittest modules under `tests/`. The original bounded check now imports the same SGD constructor; its update/checkpoint behavior is preserved. `src/dataset.py`, `src/model.py`, and `src/check_model.py` remain unchanged. No visualization was added. The proposed review configuration uses seed 42, one epoch, train/val batch size 2, workers 0, SGD learning rate 0.005, momentum 0.9, and weight decay 0.0005. It uses float32, constant learning rate, no AMP/scheduler/warmup/augmentation, and unchanged built-in detector transforms. These choices are not validation-selected hyperparameters.

- Training visits the complete train loader, retains partial batches, records each loss/total/LR/gradient norm/progress and fails on non-finite losses/gradients/updated parameters. Per-epoch losses are image-weighted batch means. Full mode uses the unchanged fixed 7,000/1,000 train/val splits; there is no test data path.
- Evaluation uses `eval()`/`torch.inference_mode()`, exact original image/category IDs, in-memory xyxy-to-xywh conversion, and provisional standard category-aware bbox COCOeval. It records all 12 summary metrics, AP/AP50/AP75 and counts per category, IoU/recall/area/maxDets settings, and actual detector settings. No extra score filtering is added. Missing-GT categories and other undefined COCO metrics use `Not evaluable`; full validation's category 6 is not zero-filled. Empty detections are supported. Full coverage is required in formal mode.
- Each attempt creates a new output run with config, status/code/environment/data identity, per-step logs, epoch training/metric/prediction/COCO summary files, latest and best checkpoints. Latest advances after successful complete training plus validation; best maximizes validation bbox mAP, retaining earlier ties. Both use atomic replacement/copy. No test-based selection occurs.
- Boundary checkpoints include model/SGD/config/epoch/global-step/RNG state, current/best mAP, best epoch, categories/image IDs/annotation hashes, model settings, runtime and code identity. Python/NumPy/Torch CPU/all CUDA RNG restore before the next epoch iterator. Resume creates a new run and permits only an increased total epoch target. The earlier best is validated/copied; latest can recover its own best when a best-copy write was interrupted. Keep historical sibling best with latest when their epochs differ. Missing/inconsistent historical best is rejected.
- Synthetic suite: **14 tests passed** using the existing `.venv`, with no real dataset access. Tests compare two uninterrupted CPU epochs against save/reload continuation for shuffle IDs, all model/SGD/RNG states and global steps; check NaN rejection before update, failed atomic-save preservation, best recovery/inheritance/mismatch, worker restriction, perfect/empty/no-GT COCO cases, ID/category/box validation, missing/duplicate coverage, eval/inference mode and source-object preservation. The first integrated suite exposed missing synthetic code-provenance fixture metadata after that field was added; the fixture was corrected and all 14 passed. This was not a real training failure.
- Bounded GPU integration: `outputs/training_functional_check/20261020_141800_458919/` completed a single three-image sample pass (2 updates), then `20261020_141839_217256/` restored that boundary and completed sample pass 2 (global step 4). Each pass used training IDs 2001-2003 and validation IDs 9001-9002. Batch sizes were `[2, 1]`. Latest/best loaded with `weights_only=True`, finite model tensors, counters, sampled evaluation scope, and zero completed full training epochs were verified. Both sampled evaluations labeled the missing railway class `Not evaluable`. Sampled AP only tests plumbing and is not baseline performance.
- Train orders were `[2003, 2002, 2001]`, then `[2002, 2001, 2003]` after continuation. Total losses were approximately 3.876161, 2.713760, 2.115657, and 2.238419. These are different mini-batch process checks and do not establish convergence. The actual configuration, losses, timings and limitations are recorded in `docs/EXPERIMENT_LOG.md` and local reports.
- Integrity evidence in `outputs/epoch_implementation_check/` confirms unchanged SHA-256, sizes, and modification times for the dataset README, class mapping, train/val annotation JSON and five sampled images (9 files). No source dataset file was written or any test annotation/image opened. This is not a full-dataset integrity audit. Existing `.venv` and official cached weights were reused without installation or downloads.
- Compilation, CLI help, synthetic tests, Git whitespace, project-owned English text, and local documentation links passed. Independent read-only code review found no actionable correctness issue. Git checks confirm dataset, `.venv`, outputs, cached weights/checkpoints and caches are ignored and untracked, with no staged changes. No commit or push occurred.

## Next Authorized Execution Decision

Review the settings at the top of `src/train.py` and the README training command, then obtain the user's explicit confirmation before the first complete train epoch and complete validation. Long-run stability, full validation performance, condition subsets, tuning and final test evaluation remain unfinished. Baseline mAP/AP50/AP75/per-class results are `Not measured`; test is `Not evaluated`. The group evaluator was supplied and integrated in the later stage recorded below. Current resume supports completed epoch boundaries with workers 0 and unchanged data/code/settings/environment; batch-level, multi-worker/distributed, and cross-environment bitwise continuation are unsupported. The simplified trainer does not compare data hashes or environment snapshots. Further commits and pushes require separate explicit confirmation.

## Completed Readability Refactor

- The user authorized the proposed simplification on 2026-10-20. No full training/evaluation, commit, push, dependency installation, or new weight download was performed.
- The primary implementation now has two files: `src/train.py` (203 lines) and `src/evaluate.py` (220 lines). `training.py` and `evaluation.py` were merged into them and removed; the external `configs/baseline_1epoch.json` was removed. The old bounded check has its original independent optimizer constructor, keeping normal training independent of all check scripts. Dataset/model/original no-update check remain unchanged.
- Training settings are constants at the top of `train.py`. `train_one_epoch()` contains the core parameter-update loop; `main()` has three numbered setup/train/evaluate/save sections. Small RNG/checkpoint/CSV helpers follow the main flow. No per-batch JSONL, source/data hash collection, environment snapshot, report lifecycle or cross-run best inheritance remains.
- Retained basic safeguards: finite-total-loss error with epoch/batch position; normal Python/PyTorch tracebacks; periodic terminal progress; final partial-batch retention; full val coverage; original IDs; empty predictions; undefined category metrics as `Not evaluable`; latest/best checkpoint selection by full-validation mAP only. The previous every-parameter/gradient scans were intentionally removed from ordinary training; the refactor check independently verified finite saved/restored tensors.
- New outputs: `outputs/training/<timestamp>/{latest.pth,best.pth,history.csv,validation.json,predictions.json}`. History has one row per epoch with individual/total loss, LR, global step and mAP/AP50/AP75. Loss summaries are image-weighted batch means. JSONs contain the latest evaluated epoch and overwrite their previous versions; all 12/per-class metrics/settings remain in validation JSON. The checkpoint's epoch/history is authoritative for recovery if auxiliary-file writes are interrupted.
- Resume uses the same run directory, model/optimizer/epoch/global step/best/settings/categories/history/RNG state, and only `--resume <run>/latest.pth`. Increase `EPOCHS` in the source first; keep other settings/data/code/environment unchanged. RNG is restored before iterator creation. The simplified entry accepts its own `bdd100k_simple_training_v1` checkpoints only. Earlier implementation/check files are preserved but unsupported for new resume. There is no separate evaluation CLI or functional-check flag. Explicit sample IDs in the evaluation function label development-only results correctly.
- Verification: **9 synthetic tests passed** (6 evaluator, 3 training). They cover mode/inference/ID/conversion/coverage/empty/no-GT metrics and preservation; actual tiny-model updates, partial batches, shuffled continuous-versus-restored model/SGD/RNG/global-step equality; NaN rejection before update; and preserving latest on failed atomic save. CLI help, compilation, local documentation links, English/whitespace/JSON checks passed. Independent read-only code review found no blocking correctness issue.
- Bounded real GPU check: `outputs/simplification_check/20261020_144730_143567/` contains its standalone development script, sample checkpoint/validation report and report. It used only training IDs 2001-2003 (batch sizes 2,1) and validation IDs 9001-9002. The initial sample pass completed 2 updates, evaluated those 2 val images, saved and restored model/SGD/RNG exactly, then completed 2 continued updates. Mean sample-pass losses were 3.4875135086476803 and 2.1554824989289045; these are process observations, not convergence or baseline results. Latest loaded model values were finite and SGD momentum devices correct. Missing train-class AP/AP50/AP75 remained `Not evaluable`. Zero full training epochs/full-validation runs were performed.
- The real check verified unchanged hashes/sizes/mtimes for the dataset README, mapping, train/val annotation JSON and five images (9 source files). No test image/annotation was accessed; this is not a whole-dataset integrity audit. Existing output files were preserved.
- Final Git checks confirm ignored data, outputs, `.venv`, cached weights/checkpoints and caches remain untracked. No staged changes, commit or push. README/context/environment/experiment records describe the simplified interface; historical entries retain their actual old paths/settings.

## Run After User Confirmation

Edit the top of `src/train.py`, then use the existing environment:

```powershell
& .\.venv\Scripts\python.exe -m src.train
```

Provisional settings remain one epoch, batch size 2, SGD LR 0.005, momentum 0.9, weight decay 0.0005 and seed 42. They are not tuned and one epoch is not assumed sufficient. The command traverses all 7,000 training images and evaluates all 1,000 validation images. No such complete run has been started. Baseline performance is `Not measured`; condition analysis is pending and test is `Not evaluated`.

## Completed Shared Team Evaluator Integration

- The user supplied root-level `evaluate_team.py` (referred to as Evaluate_T) and authorized adopting it as the group standard after review. The actual starting tree contained the prior uncommitted training/refactor changes plus this new script; they were preserved. Integration started at `b8f8239d`; no commit or push occurred during that implementation stage.
- Independent review of the original script confirmed mAP/AP50/per-class AP match standard COCOeval, including ordinary/empty/no-GT predictions and unsorted category IDs. The protocol remains bbox IoU 0.50:0.05:0.95, 101 recall points, maxDets 1/10/100, default area ranges, original image/category IDs and 0-1 scores. No score filter, NMS, clipping or remapping is added by the evaluator.
- Necessary fixes to the supplied file: remove project-external `.runtime_deps` injection; copy GT/predictions before pycocotools mutates them; support missing `info`, empty predictions/GT; use explicit UTF-8; expand compressed code into readable functions. Keep original CLI flags, `images`/`objects`/`mAP`/`AP50`/`per_class` outputs, null semantics, subset definitions and drop direction. Additional fields expose AP75, size AP/recall, per-class AP50/AP75/counts, settings/library version and COCO summary without changing existing scores.
- `src/evaluate.py` is now a 151-line inference/report adapter, with no independent COCOeval implementation. It keeps eval/inference mode, original-ID and xyxy-to-xywh conversion, complete-val coverage and explicit sampled-check labeling. All metrics come from `evaluate_team.evaluate`. The team's undefined JSON values stay null; the training report displays those values as `Not evaluable`, never zero. Training/best selection still uses complete-validation mAP. Dataset/model/train code and transforms were not changed.
- The team's `evaluate_validation()` and CLI can score an existing full-validation prediction file globally and for clear_daytime/night/dawn_dusk/rainy/overcast. They filter prediction rows by original subset IDs and use complete subset GT, including images without predictions. Subsets overlap. `mAP_drop_vs_clear_daytime` remains clear_daytime mAP minus the group mAP, a descriptive comparison. The original explicit test option is retained for later final evaluation but was not invoked.
- Verification: **18 tests passed** (8 new team-evaluator tests, 7 adapter tests, 3 unchanged training tests). Checks cover original formulas versus direct COCOeval, non-perfect detections, unsorted categories, low-score candidates, empty predictions/GT, absent classes, source-object preservation, invalid inputs, no extra box clipping/rejection, overlapping synthetic subsets/drop direction, UTF-8 JSON/CSV CLI, shared-backend delegation, inference/coverage/ID conversion and existing resume helpers. All were synthetic and used the existing `.venv`.
- Additional integration check used only archived predictions for validation IDs 9001/9002 (200 detections), from `outputs/training_functional_check/20261020_141800_458919/epoch_0001/`. All 12 shared/adapter metrics matched the previously saved local COCOeval results within 1e-14. The shared railway-class result was null and local display `Not evaluable`. GT/prediction objects stayed unchanged; hashes/sizes/mtimes of dataset README/mapping/val annotations and both archived JSON files stayed unchanged. Evidence: `outputs/team_evaluator_check/20261020_152251_464791/`. Only val annotation content was loaded and the evaluation GT was restricted to two images in memory; no new image inference, optimizer update, full validation, actual condition subset or test access occurred.
- Current Faster R-CNN score threshold 0.05, NMS 0.5 and 100 detections/image remain unchanged and recorded. The group still needs consistent prediction-export rules across its different models; a shared evaluator does not itself undo model-side filtering.
- README/context now document the shared standard and CLI. `predictions.json` in a training run is from the latest evaluated epoch, not necessarily best; its identity is in `validation.json`. Do not claim the file is best-checkpoint predictions without running the desired checkpoint's inference.
- CLI help, compilation, English/JSON/document links and Git whitespace checks passed. Data, outputs, `.venv`, weights/checkpoints and caches remain ignored and untracked; no staged changes. No dependencies were installed or official weights downloaded. Full baseline metrics remain `Not measured`; test remains `Not evaluated`. No new training experiment entry was created because this stage performed evaluator checks only.

## Training and Shared-Evaluation Milestone Publication

- On 2026-10-20 the user explicitly authorized one commit and push of the completed stage to the existing GitHub repository. The user expressly prohibited starting training during publication.
- Publication preparation verified local `main` and the live remote `main` at `b8f8239d0010246e010fcbfcc23f35e26893f36b`. The milestone comprises readable `src/train.py`, the inference adapter, shared team evaluator, three synthetic test modules and five updated README/context/status/environment/experiment documents (11 files). Existing Dataset/model/check scripts, supplied data and dependencies are unchanged.
- Eighteen synthetic tests, compilation and both CLI help commands passed. An independent read-only review found no publication blocker. Git whitespace, English/local-document-link/JSON, ignore-rule and tracked-artifact checks passed. No data/output/environment/weight/checkpoint files are included.
- Commit message: `Add readable training and shared team evaluation`. This document accompanies the authorized milestone; use Git history/status and the live remote ref for its actual identity and publication result. It does not embed its own future commit hash or predict push success. Further commits and pushes require separate user confirmation.
- Full training, complete validation, real condition analysis and test evaluation remain unexecuted. There is no formal `outputs/training/` run or baseline score. Publication does not start training.

## Next Chat: First Complete Baseline Run

The user requested a copyable handoff prompt for a separate chat. In that chat, first read AGENTS/PROJECT_CONTEXT/CURRENT_STATUS/ENVIRONMENT/EXPERIMENT_LOG/README and inspect actual Git/files/output/process state. Use the existing `.venv` and cached official weights. Current settings are one epoch, train/val batch size 2, SGD LR 0.005, momentum 0.9, weight decay 0.0005, seed 42, workers 0 and CUDA device 0, with unchanged transforms, float32, no AMP/scheduler/augmentation. These are provisional values, not optimized settings.

When the user explicitly starts that separate training task, run one complete pass over the fixed 7,000 training images (3,500 optimizer updates), followed by all 1,000 validation images scored through `evaluate_team.py`. Do not start another run if one is active, change the dataset, access test data, add epochs, retune settings, reinstall dependencies or publish new commits without applicable authorization. On success report actual losses/metrics/elapsed time/artifact path and verify latest/best at epoch 1/global step 3500. On failure record the actual error and partial progress; never claim completion. Checkpoints first appear after the complete training-and-validation boundary; no arbitrary batch resume exists. Record the actual completed/failed/interrupted experiment in EXPERIMENT_LOG and update CURRENT_STATUS in English, with Simplified Chinese user-facing reports.
