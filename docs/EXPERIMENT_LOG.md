# Experiment Log

Last updated: 2026-10-20.

## Experiment Status

No full baseline training or full-validation performance evaluation has been run. The original bounded training/checkpoint check completed three real SGD updates. The later epoch-driver functional check completed four additional updates across two three-image sample passes, with a disk restore between them and two-image validation plumbing checks. A later readability-refactor check completed four additional updates with the same three-train/two-val-image scope. These are not a trained baseline or validation-selected model. Project initialization and dataset preparation reports are not model experiments.

The 2026-10-19 model-integration smoke check loaded official COCO_V1 weights, replaced the head, checked one training-mode forward/backward batch without an optimizer step, and ran validation inference without AP. This is not a training experiment. Its configuration, losses, checks, and limitations are recorded in [Current Status](CURRENT_STATUS.md) and ignored `outputs/model_check/report.json`; no experiment entry is warranted. Model performance is `Not measured`, and test evaluation is `Not evaluated`.

## Recording Rules

- Record every completed training experiment with its actual configuration, metrics, and observations. Record failed or interrupted attempts accurately when applicable.
- Use English throughout. Never invent results or replace missing/undefined metrics with zero.
- Use validation to compare runs and choose model settings. Record final test evaluation only after model and hyperparameters are fixed.
- Keep checkpoints and generated results under ignored `outputs/`. Record their relative paths here without committing the artifacts.
- Update `docs/CURRENT_STATUS.md` after important stages and record lasting decisions in `docs/PROJECT_CONTEXT.md` when necessary.

## Bounded Training/Checkpoint Check: 20261020_012930_550530

- **Status:** Completed; training-process check, not a full baseline experiment.
- **Start/end:** 2026-10-20 01:29:30.550530 to 01:29:41.581063 EDT (UTC-04:00); elapsed 11.0297621 seconds.
- **Objective:** Verify real parameter updates, finite losses/gradients, frozen-parameter preservation, model/optimizer/RNG checkpoint save/reload, and one continued update after restoration. No performance hypothesis was tested.
- **Code:** HEAD `d26a7fbf478216679d82d3c6e59d7c850661b44a` plus uncommitted model-integration and training-check changes. `src/check_training.py` SHA-256: `2b9561db8b63e5d12297c312cea63f29914245e898f5731988c8223dfcc1fbf5`. The complete observed Git status and source hashes are in the run report. No commit or push occurred.
- **Configuration:** `configs/training_check.json`, copied to the run's `config.json`; seed 0, three steps, batch size 2, SGD learning rate 0.001, momentum 0.9, weight decay 0.0005, default dampening 0, Nesterov disabled, no scheduler, no AMP, float32. These checking choices were not tuned against validation results. Deterministic algorithms were disabled; bitwise training-trajectory reproducibility was not tested.
- **Data:** Fixed cleaned `data/bdd100k_subset_10k/annotations/train.json` and matching `images/train/`. Used annotation-order indices 0-5, with no shuffling and zero workers; image IDs 2001-2006. No validation/test image or annotation content was read. Original category IDs 1-10 and image IDs were preserved; background remains 0. No files or partitions were changed.
- **Hardware/environment:** RTX 5070, Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, CUDA runtime 13.0. Existing isolated `.venv` and official cached weights were reused without installing dependencies or downloading weights.
- **Model:** `fasterrcnn_resnet50_fpn`, `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, then a newly initialized 11-class predictor. Official weight URL: `https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`; identity recorded in [Current Status](CURRENT_STATUS.md). The predictor and trainable backbone parameters received real updates. Backbone body `layer2`/`layer3`/`layer4` remain trainable; earlier body stages remain frozen, with FrozenBatchNorm2d.
- **Transforms:** Model-internal mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, `min_size=(800,)`, `max_size=1333`, padding divisor 32, no fixed size, no Dataset resize/normalization duplication, and no augmentation. No transform or detector threshold was tuned.

| Step | Image IDs | Objects | Classifier loss | Box loss | RPN objectness loss | RPN box loss | Total loss |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 2001, 2002 | 12, 17 | 2.6997947692871094 | 0.997239887714386 | 0.06843404471874237 | 0.28850966691970825 | 4.053978443145752 |
| 2 | 2003, 2004 | 25, 10 | 2.435701608657837 | 0.9851020574569702 | 0.05786222219467163 | 0.1101890504360199 | 3.5888547897338867 |
| 3 (restored) | 2005, 2006 | 20, 28 | 2.1695380210876465 | 0.8970697522163391 | 0.4663691520690918 | 0.173049196600914 | 3.706026077270508 |

- **Update results:** All individual/total losses, gradients, updated parameters, and momentum buffers were finite. Each step had 72 finite/nonzero gradient tensors and 72 changed parameter tensors, including backbone and predictor changes. Frozen parameters stayed exactly unchanged. Three optimizer steps and zero full epochs completed; no early stopping or checkpoint-selection policy was used. Peak PyTorch CUDA allocated memory: 2391.22021484375 MiB, not total GPU usage.
- **Checkpoint results:** Saved after step 2, rebuilt the model/SGD/DataLoader iterator, loaded with `weights_only=True`, and checked model tensors, optimizer momentum/hyperparameters/devices, configuration, model/transform/category metadata, CPU/CUDA RNG, step, and data cursor exactly. Restored cursor 4 continued with IDs 2005/2006 as step 3, without duplicated/skipped images. The final checkpoint also matched the live model/optimizer after a disk roundtrip. This does not establish general shuffled/multi-worker resume or equality to an uninterrupted training trajectory.
- **Artifacts:** `outputs/training_check/20261020_012930_550530/` contains `config.json`, `steps.jsonl`, `report.json`, `checkpoint_before_resume.pth` (330,439,684 bytes), and `checkpoint_final.pth` (330,436,364 bytes). Checkpoint SHA-256 values respectively: `9ba39810a2fdca6ae2ce7b5cf4625caa458edf1026d813c05fca4fd2403c0b4c`, `0b1a407947eb5cf6d7b0df8681228385dc22a4f9817422f139cc3ada372d9410`. No predictions, plots, or evaluation files were generated.
- **Integrity:** `outputs/training_check_integrity/source_integrity_check.json` confirms all 30,071 dataset-file paths/sizes/modification times, nine selected data hashes, and the four preserved model/loader/check/report hashes stayed unchanged. Validation/test file metadata was inventoried without reading their contents. This is not a full-dataset content-hash audit.
- **Performance:** Validation mAP/AP50/AP75/size AP/recall, per-class AP, and condition results: `Not measured` because no evaluation was performed. Test evaluation: `Not evaluated`. No checkpoint was selected by validation or test performance.
- **Observations/next decision:** The training and basic checkpoint operations work for the tested few batches. Different images/target counts explain why these losses are not a controlled before/after comparison; the sequence does not establish convergence or detection quality. Formal baseline settings, full-epoch training, validation evaluation, and checkpoint selection need a later authorized stage. The original no-update model check is still separate and unchanged; temporary prediction visualization remains removed.

## Epoch-Driver Functional Check and Continuation: 2026-10-20

- **Status/purpose:** Completed implementation checks, not a formal baseline experiment. Zero complete 7,000-image training epochs and zero complete 1,000-image validation evaluations.
- **Run identities:** `outputs/training_functional_check/20261020_141800_458919/` (initial) and `outputs/training_functional_check/20261020_141839_217256/` (continuation). The continuation records the source latest checkpoint path and SHA-256.
- **Time:** Initial 2026-10-20 14:18:00.458919 to 14:18:09.262285 EDT (UTC-04:00), measured run duration 8.7646532 seconds. Continuation 14:18:39.217256 to 14:18:42.369360 EDT, measured duration 2.9578616 seconds. Report timestamps also include metadata-writing overhead outside the measured interval.
- **Code:** HEAD `b8f8239d` plus the uncommitted training/evaluation implementation. Both run reports/checkpoints record their actual source hashes, Git status, environment and effective configuration. No commit/push occurred.
- **Actual configuration:** `configs/baseline_1epoch.json` with `--functional-check`; seed 42, train/val batch size 2, workers 0, train shuffle true, val shuffle false, SGD learning rate 0.005, momentum 0.9, weight decay 0.0005, no scheduler/warmup/AMP/augmentation, float32, no gradient clipping, deterministic algorithms disabled. Initial target one sample pass; continuation used `--epochs 2`, meaning total sample passes including the saved one. These values were not optimized or chosen from sampled validation performance.
- **Model/environment:** Existing official COCO_V1 cached Faster R-CNN ResNet50-FPN with a replacement 11-class head and unchanged normalization/800-1333 resize/padding/backbone/inference defaults. RTX 5070; Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, CUDA 13.0, NumPy 2.5.3, pycocotools 2.0.11. Existing `.venv` reused; no dependency installation or weight download.
- **Data scope:** Only original annotation-order training indices 0-2 (IDs 2001-2003; 12/17/25 objects) and validation indices 0-1 (IDs 9001/9002) were decoded. Training shuffle changed iteration order within those three samples. This is an explicit functional access limit, not resampling/repartitioning or a new split. Original source annotations, categories, image IDs and transforms were unchanged. No test data or condition subset was accessed.

| Global step | Sample pass | Training IDs | Total loss | Restored boundary |
|---|---:|---|---:|---|
| 1 | 1 | 2003, 2002 | 3.8761606216430664 | No |
| 2 | 1 | 2001 | 2.7137603759765625 | No |
| 3 | 2 | 2002, 2001 | 2.115657091140747 | Yes |
| 4 | 2 | 2003 | 2.2384190559387207 | Yes |

- **Training/checkpoint observations:** Both passes retained the final one-image batch. All losses/gradient norms/updated parameters passed finite checks. Saved/loaded latest and best checkpoints use `weights_only=True`, with completed sample-pass counters 1 then 2, global step 2 then 4, and `completed_full_training_epochs=0`. Latest sizes were 330,444,034 and 330,444,162 bytes. Model/optimizer/RNG/data/config/model-setting/environment checks allowed continuation. There is no claim of GPU bitwise agreement with uninterrupted execution.
- **Sampled evaluation:** Each pass evaluated exactly IDs 9001 and 9002 using eval/inference mode and COCO bbox evaluation; exported 200 then 15 detections. All 12 metrics and per-class values exist in local `epoch_0001/metrics.json` and `epoch_0002/metrics.json`, labeled `purpose=functional_check`, `performance_result=false`. The railway `train` class had zero GT and AP/AP50/AP75 `Not evaluable`. These values only exercised metric export and latest/best selection plumbing and are deliberately not presented as full-validation results. They were not used to choose baseline hyperparameters.
- **Artifacts:** Each run contains effective config/status/provenance, per-step JSONL, latest/best checkpoints, and its completed sample pass's training/metrics/predictions/COCO text summary. No visualization was generated. `outputs/epoch_implementation_check/report.json` records counter/finite-model/missing-class assertions and nine unchanged source hashes/sizes/mtimes. It does not represent an entire-dataset audit.
- **Related synthetic checks:** All 14 unittest checks passed. In-memory COCO perfect/empty/missing-GT cases, prediction/ID/coverage/mode validation, shuffled CPU epoch-trajectory equality after restoration, failed atomic saves, and historical/current best recovery were covered. An initially missing code-provenance field in the synthetic report fixture was corrected; the final full suite passed. No real training attempt failed or was interrupted.
- **Formal results:** Baseline validation mAP/AP50/AP75/size AP/recall/per-class/condition results: `Not measured` because no full run occurred. Test evaluation: `Not evaluated`. Neither functional checkpoint is a selected baseline; formal entry points reject mixing check checkpoints with full runs.
- **Limitations/next action:** Review the provisional one-epoch config and obtain user confirmation before full training plus full validation. One epoch is not assumed sufficient. Full-run stability, condition analysis and shared-evaluator reconciliation are pending. Supported continuation is epoch-boundary, workers 0, matching configuration/data/runtime; arbitrary batch/multi-worker/distributed resume and cross-hardware/version bitwise equality are unsupported.

## Readability-Refactor Functional Check: 20261020_144730_143567

- **Status/purpose:** Completed a bounded code-refactor check, not a formal baseline experiment. No full training epoch or complete validation evaluation.
- **Time:** 2026-10-20 14:47:36.750882 to 14:47:40.161279 EDT (UTC-04:00), including sample loading, model construction, two sample passes, one sampled evaluation and disk restoration. Wall duration approximately 3.410397 seconds; interpreter/import overhead is outside this interval.
- **Code/settings:** HEAD remains `b8f8239d` plus uncommitted refactor changes. Training now resides directly in `src/train.py`, evaluation in `src/evaluate.py`; external baseline JSON and detailed runtime reports were removed. The development script imports the production training/evaluation/save/RNG helpers and the original check's equality helper. Seed 42, batch size 2, SGD LR 0.005/momentum 0.9/weight decay 0.0005, shuffle true, workers 0, float32, constant LR, no AMP/scheduler/warmup/augmentation. These are unchanged provisional settings, not validation-tuned choices.
- **Environment/model:** Existing `.venv`, RTX 5070, Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, CUDA 13.0, existing cached official COCO_V1 Faster R-CNN and replacement 11-class head. No dependency installation/download, transform or trainable-backbone change.
- **Scope:** Original first three training IDs 2001-2003, two batches per sample pass including the final one-image batch. Original first two validation IDs 9001-9002 were inferred/evaluated once between sample passes. No source data edits, new partitions, condition analysis, or test image/annotation access.
- **Updates/results:** Two initial real SGD updates plus two after disk restoration, global step 4. Image-weighted mean loss was 3.4875135086476803 for sample pass 1 and 2.1554824989289045 for the restored pass. Per-loss means are in the local report. Different batches, repeated tiny scope, nondeterministic CUDA algorithms and only four updates do not establish convergence or detection quality.
- **Checkpoint/evaluation:** Model/SGD states matched disk exactly; rebuilt-model restoration preserved state and Python/NumPy/Torch CPU/CUDA RNG exactly, with SGD momentum on the proper device. Continued updates produced finite model values. Sample evaluation used eval/inference mode and original IDs, was explicitly marked functional/non-performance, and retained `Not evaluable` for category 6 AP/AP50/AP75. The sample checkpoint is labeled `functional_check_only` and is deliberately rejected by the normal trainer's format check.
- **Artifacts/integrity:** `outputs/simplification_check/20261020_144730_143567/{check_refactor.py,report.json,sample_checkpoint.pth,sample_validation.json}`. Nine source files (README, mapping, train/val annotations and five images) retained hashes/sizes/mtimes. This is not a complete-dataset audit. Historical outputs/checkpoints are untouched; no visualizations were created.
- **Other checks:** Nine synthetic unittest checks passed with the existing environment. They verify training-mode switching, actual updates, partial batches, uninterrupted-versus-restored shuffled CPU trajectories/RNG, non-finite loss rejection, failed atomic save preservation, and COCO/ID/mode/coverage/undefined-metric behavior. CLI help and compilation passed. No real run failed or was interrupted.
- **Formal performance/next action:** Full validation mAP/AP50/AP75/per-class/size/recall/condition metrics remain `Not measured`; test remains `Not evaluated`. Review top-of-file settings and obtain user confirmation before `python -m src.train` starts complete train/val traversal. Resume is same-run, workers 0, unchanged data/code/environment/settings, completed boundaries only; no data/environment hash validation, batch-level resume or cross-environment bitwise guarantee. No commit or push.

## Template for a Future Run

Copy this template for a real run; it is not an experiment entry.

```text
Run ID:
Status: Planned / Running / Completed / Failed / Interrupted
Start and end time (with timezone):
Objective or hypothesis:
Code commit and uncommitted changes:
Configuration path and actual settings:
Dataset root, fixed split, and annotation/subset files:
Category mapping and image ID handling:
Hardware and device:
Python, PyTorch, Torchvision, CUDA, and evaluator versions:
Random seed and determinism settings:
Model variant, pretrained weights, and detection head:
Transforms, resize settings, and augmentation:
Optimizer, learning rate, weight decay, momentum, and scheduler:
Batch size, epoch count, and any early stopping rule:
Training duration and completed epochs:
Checkpoint path and validation-based selection criterion:
Evaluation settings (IoU thresholds, max detections, and any score filtering):
Validation results (mAP, AP50, AP75, size AP, and recall):
Per-class AP (mark missing ground-truth classes Not evaluable):
Condition results (val_all, clear_daytime, night, dawn_dusk, rainy, overcast):
Artifacts (checkpoint, predictions, metrics, and plots):
Observations, failures, limitations, and next decision:
Final test evaluation, if authorized after settings are frozen: Not evaluated
```
