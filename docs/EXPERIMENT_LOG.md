# Experiment Log

Last updated: 2026-10-20.

## Experiment Status

No full baseline training or model performance evaluation has been run. One bounded training/checkpoint check has completed three real SGD updates and produced check checkpoints, as recorded below. These are not a trained baseline or validation-selected model. Project initialization and dataset preparation reports are not model experiments.

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
