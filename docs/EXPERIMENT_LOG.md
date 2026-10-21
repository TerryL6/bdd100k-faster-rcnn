# Experiment Log

Last updated: 2026-10-21.

## Experiment Status

The user manually completed two independent experiments using the original model-internal aspect-preserving `min_size=800`, `max_size=1333` preprocessing. Run `20261020_155352_056862` completed one full training epoch and complete-validation evaluation: 3,500 updates over 7,000 training images, then all 1,000 validation images; mAP is 0.20033579847015223. Fresh run `20261020_182545_580059` subsequently completed all five epochs and 17,500 updates with validation after every epoch. Its best checkpoint is epoch 4 / global step 14000, with mAP 0.25544452083836444, AP50 0.5170826995660548 and AP75 0.21488694482832407. Its latest checkpoint and current validation/prediction JSON files belong to epoch 5, whose mAP is 0.23822375438048712. These measured results do not establish convergence or a finalized baseline. The old artifacts are preserved.

The user manually completed the third independent run, `20261021_002448_944703`, using the prepared fixed-640-by-640 preprocessing and fresh cached official COCO_V1 initialization with a new 11-class predictor. It completed ten full epochs and 35,000 updates. Best is epoch 4 / step 14000, with mAP 0.17773109411680846, AP50 0.3619035810493331 and AP75 0.15257382817657714. Latest is epoch 10 / step 35000; its mAP is 0.17702375663177553, and its JSON files are the currently retained detailed results. This run's best mAP is 7.7713426721555985 percentage points below the old five-epoch best, without establishing a cause or sufficient convergence. Its artifact audit did not rerun training, inference or scoring. Further assistant-executed training still requires the user's explicit instruction.

On 2026-10-21, after reviewing the lower fixed-640 validation results, the user authorized restoring Torchvision's default 800/1333 input rule for normal training and recording that decision. The ten-epoch target and other settings stayed unchanged. The user completed independent restored-input run `20261021_022856_226381`: ten epochs / 35,000 updates, best epoch 7 mAP 0.2519508348597597; latest epoch-10 mAP is 0.24015201202561187. The user subsequently completed the separate old-five-epoch continuation through epoch 10, adding five epochs / 17,500 updates. Its inherited epoch-4 best remains 0.25544452083836444; latest epoch-10 mAP is 0.24465600637621113. These are four independent experiments plus one continuation branch, with 31 actual full-validation cycles rather than duplicated inherited history. No final baseline or new recipe is selected. Exact elapsed durations remain `Not measured`. Historical preparation and scores retain their original identities.

Earlier bounded training/checkpoint, epoch-driver and readability-refactor checks performed three, four and four updates respectively using tiny samples. Those historical process checks are not full-validation performance or selected baseline models. Project initialization and dataset preparation reports are not model experiments.

The 2026-10-19 model-integration smoke check loaded official COCO_V1 weights, replaced the head, checked one training-mode forward/backward batch without an optimizer step, and ran validation inference without AP. This is not a training experiment. Its configuration, losses, checks, and limitations are recorded in [Current Status](CURRENT_STATUS.md) and ignored `outputs/model_check/report.json`; no experiment entry is warranted for that smoke check. Its performance is `Not measured`. Test evaluation of this repository's Faster R-CNN remains `Not evaluated`.

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

## First Complete Single-Epoch Run: 20261020_155352_056862

- **Status/objective:** Completed by the user through a manual terminal command, not executed by the assistant. Establish the first measured full-epoch/full-validation result using the existing provisional settings. This is not a convergence claim or a finalized baseline.
- **Evidence/code:** Saved `history.csv` and `validation.json` were inspected after completion; terminal output returned normally. Source milestone was `a65cd736`; the working tree was clean before the later five-epoch preparation. The simplified trainer does not save source hashes or a complete runtime snapshot, so no new source/runtime provenance measurement is claimed.
- **Start/end/duration:** Exact start and end times and wall-clock training/validation duration: `Not measured`, because the trainer does not log them. The run directory name and artifact modification times are not substitutes for measured elapsed time.
- **Configuration/data:** `src/train.py`, `EPOCHS=1`, seed 42, train/val batch size 2, train shuffle true, val shuffle false, workers 0, device `cuda:0`, print interval 50. SGD LR 0.005, momentum 0.9, weight decay 0.0005, default dampening 0, Nesterov disabled; float32, no AMP/scheduler/warmup/augmentation. Deterministic algorithms were not forced. One complete pass over fixed `annotations/train.json` and `images/train/` (7,000 images, 3,500 updates), followed by complete `annotations/val.json` and `images/val/` (1,000 images). Original category IDs 1-10 and image IDs were retained; background 0. No test evaluation or condition analysis.
- **Model/environment:** Cached official `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, `fasterrcnn_resnet50_fpn`, a newly initialized 11-class predictor, unchanged internal normalization/800-1333 resize/padding/backbone defaults. Existing `.venv` and documented RTX 5070, Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, CUDA 13.0 and pycocotools 2.0.11. This environment description is inherited from the verified setup, not a new runtime measurement. No dependency installation or additional weight download was needed for preparation.
- **Actual image-weighted training means:** `loss_classifier=0.3028968423924276`, `loss_box_reg=0.444215459299939`, `loss_objectness=0.07495462468772063`, `loss_rpn_box_reg=0.17396516367952738`, `total_loss=0.9960320900596146`. Saved history contains epoch 1 and global step 3500.
- **Evaluation protocol:** `src/evaluate.py` delegates bbox scoring to shared `evaluate_team.py`/COCOeval; IoU thresholds 0.50:0.05:0.95, 101 recall thresholds, maxDets 1/10/100, standard small/medium/large area ranges, category-aware metrics on a 0-1 scale. Inference uses eval/inference mode, original image pixels/IDs and COCO xywh export. Detector score threshold 0.05, NMS IoU 0.5, maximum 100 detections per image; evaluator applies no additional score filter. Report contains `performance_result=true`, `split=val`, `epoch=1`, `image_count=1000` and 79,745 predictions.

| Full-validation metric | Actual value (0-1) |
|---|---:|
| mAP (IoU 0.50:0.95) | 0.20033579847015223 |
| AP50 | 0.422389717269082 |
| AP75 | 0.1666071423344876 |
| AP small | 0.09148089005032771 |
| AP medium | 0.24903162551375516 |
| AP large | 0.37799573736071296 |
| AR maxDets 1 | 0.15513353280535255 |
| AR maxDets 10 | 0.3107474782692513 |
| AR maxDets 100 | 0.3343225680473746 |
| AR small | 0.17267628872641438 |
| AR medium | 0.38156113858606416 |
| AR large | 0.5324282186768101 |

| Category | GT objects | AP | AP50 | AP75 |
|---|---:|---:|---:|---:|
| person | 1280 | 0.26663664898545664 | 0.5731135164193837 | 0.20694134354106955 |
| rider | 62 | 0.11392742159766159 | 0.2638216708051609 | 0.08163758209849026 |
| car | 10236 | 0.3811753918084215 | 0.7176989915620244 | 0.36632522669009254 |
| truck | 401 | 0.23143242115775386 | 0.3743771601446463 | 0.25203631135556726 |
| bus | 165 | 0.23446348293731523 | 0.39611744666380083 | 0.24844835288290287 |
| train | 0 | Not evaluable | Not evaluable | Not evaluable |
| motor | 49 | 0.07262031204085541 | 0.21384151355187128 | 0.012820879352372925 |
| bike | 88 | 0.09006252522835757 | 0.22612786682130778 | 0.07806709309663346 |
| traffic light | 2714 | 0.1596092000774494 | 0.48446419995949364 | 0.05657871266731679 |
| traffic sign | 3470 | 0.25309478239809885 | 0.551945089494049 | 0.19660877932594242 |

- **Checkpoint/artifacts:** `outputs/training/20261020_155352_056862/` contains `latest.pth`, `best.pth`, `history.csv`, `validation.json` and `predictions.json`. Latest/best correspond to the only completed epoch, with terminal `best_epoch=1`; selection maximizes complete-validation mAP. No checkpoint tensor reload was performed while recording these results. Missing railway-class GT is excluded from COCO means, not zero-filled. Condition-specific results: `Not measured`; final test: `Not evaluated`.
- **Observations/next decision:** One epoch successfully exercised the complete training/validation/save path. AP75 is below AP50 and small-object AP is below medium/large AP; this does not isolate causes or prove sufficient training. The user chose to disregard this run as a continuation source and begin a new five-epoch experiment. The existing run remains unchanged on disk; its epoch/results do not count toward or select checkpoints for the new experiment.

## Completed Fresh Five-Epoch Run: 20261020_182545_580059

- **Status/objective:** Completed manually by the user as an independent experiment. All five full training epochs and 17,500 optimizer updates completed, with complete validation after every epoch. The run initialized from cached official COCO_V1 weights and a new 11-class predictor, without resuming `20261020_155352_056862`. Neither experiment's history is combined with the other. Five epochs are an observation window, not evidence that the model has converged.
- **Evidence/code:** Actual `history.csv`, `validation.json`, `predictions.json`, `best.pth` and `latest.pth` were inspected. Source milestone is `a65cd736` plus the existing preparation changes, including `EPOCHS=5`; the simplified trainer did not record source hashes or a complete runtime snapshot. The current audit loaded both this run's checkpoints and both single-epoch checkpoints using the existing `.venv`, `map_location="cpu"` and `weights_only=True`. It did not construct a detector or rerun training, image inference, scoring or timing.
- **Start/end/duration:** Exact start/end times and wall-clock training/validation duration: `Not measured`, because the trainer does not record them. Directory names and file modification times do not establish elapsed training time.
- **Configuration/data:** `EPOCHS=5`, seed 42, train/val batch size 2, train shuffle true, val shuffle false, workers 0, device `cuda:0`, print interval 50. SGD LR 0.005, momentum 0.9, weight decay 0.0005, default dampening 0 and Nesterov disabled; float32, no AMP, scheduler, warmup or added augmentation. Deterministic algorithms were not forced. Every training epoch traversed the fixed 7,000-image train split (3,500 updates), followed by the complete 1,000-image validation split. No condition analysis or test evaluation was performed.
- **Model/environment:** `fasterrcnn_resnet50_fpn`, cached `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, newly initialized 11-class predictor, background 0 and original foreground category IDs 1-10. FrozenBatchNorm2d and trainable backbone body stages `layer2`/`layer3`/`layer4` were retained. Existing local `.venv` and documented RTX 5070 environment were reused; the current audit does not claim a new GPU/runtime measurement, dependency installation or weight download.
- **Actual preprocessing:** Dataset returned original-size RGB float32 `[0, 1]` images and original pixel `xyxy` boxes. The detector applied mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` once, aspect-preserving bilinear resizing with `align_corners=False`, short-side target 800 and long-side maximum 1333, no fixed canvas, then batch padding to multiples of 32. This experiment did not use fixed 640-by-640 input. Saved validation metadata confirms `min_size=[800]`, `max_size=1333`, `size_divisible=32` and `fixed_size=null`.

Actual image-weighted training means:

| Epoch | Global step | Classifier loss | Box loss | RPN objectness loss | RPN box loss | Total loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3500 | 0.30418116176767007 | 0.4446390875152179 | 0.07661206300636487 | 0.17457908455508628 | 1.0000113968443392 |
| 2 | 7000 | 0.26581182139686177 | 0.41452829025472915 | 0.05925446503529591 | 0.15908452140193965 | 0.8986790980888265 |
| 3 | 10500 | 0.2510080945725952 | 0.401666601815394 | 0.05081439554518355 | 0.1521499016525756 | 0.8556389935857484 |
| 4 | 14000 | 0.24190184214924063 | 0.3925883290384497 | 0.04686770626955799 | 0.14839652752596885 | 0.8297544049832172 |
| 5 | 17500 | 0.23654089846994195 | 0.3886951625176838 | 0.044389868071968 | 0.14563250353892468 | 0.8152584325985185 |

Actual complete-validation metrics, on a 0-1 scale:

| Epoch | Global step | mAP (IoU 0.50:0.95) | AP50 | AP75 | Selection |
|---:|---:|---:|---:|---:|---|
| 1 | 3500 | 0.1834592123086974 | 0.3979016050818286 | 0.1465550345797109 | Earlier epoch |
| 2 | 7000 | 0.22445889409023706 | 0.4664552216111791 | 0.1828126828849158 | Earlier epoch |
| 3 | 10500 | 0.23896501467662432 | 0.49930742495009356 | 0.1896600548267454 | Earlier epoch |
| 4 | 14000 | 0.25544452083836444 | 0.5170826995660548 | 0.21488694482832407 | Best checkpoint |
| 5 | 17500 | 0.23822375438048712 | 0.5023644463985995 | 0.19519708766194233 | Latest checkpoint and JSON reports |

- **Evaluation protocol:** Shared `evaluate_team.py` / category-aware bbox COCOeval, IoU 0.50:0.05:0.95, 101 recall thresholds, maxDets 1/10/100 and standard size ranges. Model inference score threshold 0.05, NMS IoU 0.5 and maximum 100 detections per image remained unchanged; the evaluator added no score filter. Original image/category IDs and original-image-pixel COCO `xywh` predictions were retained.
- **Checkpoint identity:** CPU read-only loading confirms `best.pth` has `epoch=4`, `global_step=14000`, `best_epoch=4`, `best_mAP=0.25544452083836444` and four history rows. `latest.pth` has `epoch=5`, `global_step=17500`, `best_epoch=4`, the same best mAP and five history rows. Both heads have classification weight shape `[11, 1024]` and box-regression weight shape `[44, 1024]`. The saved optimizer uses LR 0.005, momentum 0.9 and weight decay 0.0005. These historical checkpoints do not save the transform in their compact training settings.
- **Latest report/predictions:** `validation.json` explicitly records `epoch=5`, `split=val`, `performance_result=true` and all 1,000 original image IDs 9001-10000. `predictions.json` contains 75,689 rows over exactly those 1,000 IDs. These files were overwritten after each epoch and are not epoch-4 predictions or detailed epoch-4 metrics.

The following additional metrics belong exclusively to the latest **epoch 5** report:

| Full-validation metric | Epoch 5 actual value (0-1) |
|---|---:|
| AP small | 0.09862201757907721 |
| AP medium | 0.28406951040373796 |
| AP large | 0.42116857076006453 |
| AR maxDets 1 | 0.1786465744578652 |
| AR maxDets 10 | 0.3414586028042812 |
| AR maxDets 100 | 0.3633652647477816 |
| AR small | 0.1987752951880332 |
| AR medium | 0.40825719486511186 |
| AR large | 0.5474731114931933 |

| Category | GT objects | Epoch 5 AP | Epoch 5 AP50 | Epoch 5 AP75 |
|---|---:|---:|---:|---:|
| person | 1280 | 0.24985695502667887 | 0.5802896305643392 | 0.17790428670672098 |
| rider | 62 | 0.20892616420479454 | 0.43702790616944936 | 0.16852227632557978 |
| car | 10236 | 0.40691916392184907 | 0.7395005022948862 | 0.39526004063791 |
| truck | 401 | 0.23370844839239713 | 0.3931502729018184 | 0.24815073772589744 |
| bus | 165 | 0.2749283524874503 | 0.4501454154913736 | 0.3138924634050609 |
| train | 0 | Not evaluable | Not evaluable | Not evaluable |
| motor | 49 | 0.15189792641900268 | 0.42986468279942647 | 0.03943212503068489 |
| bike | 88 | 0.1483799609689954 | 0.3329491887173545 | 0.1172264780962685 |
| traffic light | 2714 | 0.185862364245379 | 0.5718299767464834 | 0.060904595890513745 |
| traffic sign | 3470 | 0.2835344537578371 | 0.5865224419022649 | 0.23548078513884446 |

- **Unavailable best-epoch detail:** Epoch-4 size AP, recall, per-class AP/AP50/AP75 and predictions were not separately retained. They are unavailable from the current artifacts, even though its aggregate mAP/AP50/AP75 are preserved in history and the checkpoint. The audit did not regenerate them or substitute epoch-5 details. Missing railway-train GT is excluded from COCO means rather than zero-filled.
- **Artifacts/observations:** `outputs/training/20261020_182545_580059/{latest.pth,best.pth,history.csv,validation.json,predictions.json}` remains intact alongside the independent single-epoch run. Training mean loss decreased each epoch; validation mAP peaked at epoch 4 and decreased at epoch 5. This observation alone does not establish convergence or identify the cause. Condition results remain `Not measured`; test remains `Not evaluated`. No training, real image inference, rescoring, benchmarking, dataset edits, installation, download, commit or push occurred during the current artifact audit.

## Historical Preparation: Fixed 640-by-640 Ten-Epoch Observation Experiment

This section preserves the preparation-stage record before the user started the run. Its then-planned execution and checks are historical; the completed run is recorded separately below.

- **Status/objective:** Code preparation for a separate experiment is authorized. `EPOCHS=10` and `IMAGE_SIZE=640`; ten epochs are an observation stage, not a claim of sufficient training or convergence. No new run has started, no new training directory/checkpoint/results are recorded, and all new-run performance and elapsed duration are `Not measured`.
- **Fresh initialization:** The manual command is `& .\.venv\Scripts\python.exe -m src.train` from the repository root, without `--resume`. It uses the existing local cached official `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, replaces the predictor with a new 11-class head, and creates a separate timestamped run under `outputs/training/`. Neither the single-epoch nor the five-epoch checkpoint is a restoration source. Both historical experiments and all their artifacts remain preserved.
- **Unchanged settings:** Train/val batch size 2, SGD LR 0.005, momentum 0.9, weight decay 0.0005, seed 42, workers 0, device `cuda:0`, print interval 50, float32, train shuffle true and val shuffle false. No AMP, scheduler, warmup or added data augmentation. Mean/std, FrozenBatchNorm2d, backbone freezing and inference score/NMS/detection-count thresholds remain unchanged. Use the existing `.venv` and local RTX 5070; no cloud or paid resource is created.
- **Dataset boundary:** Dataset continues to return original-size RGB float32 `[0, 1]` tensors and original pixel `xyxy` boxes. Fixed train/val splits, IDs, categories, images and annotations remain unchanged. Image resize/padding, target-box scaling and prediction restoration occur only in memory inside the model; Dataset converts source `xywh` to `xyxy`, and the export adapter converts restored `xyxy` back to COCO `xywh`. There is no Dataset resize or duplicate normalization.
- **Actual resize/padding rule:** The detector normalizes once with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Its transform inherits Torchvision's aspect-preserving bilinear resize with `min_size=max_size=640`, `align_corners=False` and `recompute_scale_factor=True`; the long edge fits 640, smaller images may be enlarged, and the actual integer resized content dimensions are authoritative. Content is aligned to the canvas's top-left corner. Padding is added only on the right and bottom to produce an exact 640-by-640 canvas in both training and validation. Fill value is zero in normalized tensor space, equivalent to the RGB channel means before normalization, not RGB 114. There is no direct stretching to a square.
- **Coordinate handling:** Target boxes scale with actual resized width/height ratios. Top-left placement has zero x/y padding offsets. The transform retains each actual content size for RPN/ROI clipping and the native postprocess, which restores predictions to original-image pixel coordinates before the evaluator adapter converts `xyxy` to COCO `xywh`. `SquareCanvasRoIAlign` uses the fixed canvas only to infer FPN feature scales, preventing the content size from implying the wrong pyramid scale; it does not replace the content sizes used for clipping or output restoration.
- **Saving/selection/continuation:** Every epoch still trains all 7,000 images, performs complete 1,000-image validation and retains the latest/best/history/validation/predictions saving flow. Planned total updates are 35,000 if all ten epochs complete; actual completed updates for this experiment are none. Best selection maximizes complete-validation mAP. Latest JSON files continue to belong to the latest evaluated epoch. New checkpoint settings include the full transform identity and reject restoration when settings are absent or differ, preventing old 800/1333 checkpoints from being silently restored under the new preprocessing.
- **Comparison limitation:** The target input size is aligned with the teammates' nominal 640-by-640 size. As supplied by the user, YOLO commonly uses aspect-preserving letterbox preprocessing, while the RT-DETRv2 configuration uses direct Resize. Equal target dimensions do not establish identical preprocessing across models; this experiment explicitly records its own placement, fill and interpolation rules.
- **Verification completed:** The existing `.venv` passed all 32 synthetic tests with `python -m unittest discover -s tests -v`, plus `python -m compileall -q src tests` and training CLI `--help`. Tests cover original Dataset RGB/box/ID outputs using temporary synthetic files, train/val geometry, mixed/partial batches, normalized padding, original-coordinate box/COCO export roundtrip, unchanged files/tensors/COCO records, correct FPN scales and ROI pooling coordinates, report metadata and rejection of missing/different preprocessing on resume. Existing synthetic evaluator and tiny-model update/restore checks also passed; no real Faster R-CNN was constructed or trained. The historical 18-test result is distinct from this rerun.
- **Preservation checks:** All ten files in the two completed training runs and the dataset README, class mapping and train/val annotations retained identical SHA-256, sizes and modification times before/after this preparation (14 files). No new training run directory was created. This is not a full-dataset integrity audit. The official cached weight hash matches the previously recorded value. English project text, local Markdown links, JSON syntax and Git whitespace checks passed; data/outputs remain ignored/untracked and nothing is staged.
- **Execution boundary:** No actual Faster R-CNN training, real-data inference/scoring or benchmarking was performed. No test access, dataset modification, dependency installation, weight download, cloud migration, condition analysis, old-best inference, PDF edits, commit or push is included. The assistant must not execute the displayed training command or click Run unless the user later explicitly says that the assistant should execute training.

## Completed Fixed 640-by-640 Ten-Epoch Run: 20261021_002448_944703

- **Status/objective:** Completed manually by the user as the third independent experiment. Ten full 7,000-image training epochs, 35,000 optimizer updates and complete 1,000-image validation after each epoch were recorded. This run uses the prepared fresh initialization rather than either earlier experiment's checkpoint; histories and selection remain separate. Ten epochs are the completed observation window, not proof of sufficient convergence.
- **Evidence/code:** Inspected actual `history.csv`, `validation.json`, `predictions.json`, `best.pth` and `latest.pth` under `outputs/training/20261021_002448_944703/`. The audit started from a clean working tree at source HEAD `a52bbcbb`. This describes the checkout inspected during the audit: the trainer does not save source/runtime hashes or a complete runtime snapshot, so exact execution provenance is not independently verified by those files. Both checkpoints were loaded read-only with the existing `.venv`, `map_location="cpu"` and `weights_only=True`; no detector was constructed.
- **Start/end/duration:** Exact start/end times and elapsed training/validation duration are `Not measured`. The trainer does not log elapsed time; directory names and artifact modification times are not training-duration measurements.
- **Configuration/data:** `EPOCHS=10`, `IMAGE_SIZE=640`, seed 42, train/val batch size 2, train shuffle true, val shuffle false, workers 0, device `cuda:0`, print interval 50. SGD LR 0.005, momentum 0.9 and weight decay 0.0005; float32, no AMP, scheduler, warmup or added augmentation. Every epoch uses the unchanged fixed training and validation splits, preserving category/image IDs; each epoch contributes 3,500 updates. No condition or test evaluation is included.
- **Model/environment:** `fasterrcnn_resnet50_fpn`, cached official `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` and a newly initialized 11-class predictor, background 0 and foreground IDs 1-10. Original normalization, FrozenBatchNorm2d, backbone body `layer2`/`layer3`/`layer4` trainability and inference thresholds remain unchanged. The existing local `.venv` and documented RTX 5070 environment were reused; the audit does not remeasure the runtime or hardware.
- **Actual preprocessing identity:** Dataset retains original-size RGB float32 `[0, 1]` images and original-pixel `xyxy` targets. Model normalization occurs once using mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`. Both checkpoint settings and validation metadata record `min_size=[640]`, `max_size=640`, `resize_policy=aspect_ratio_square_pad`, `canvas_size=[640,640]`, bilinear interpolation, `align_corners=false`, `recompute_scale_factor=true` and upscaling allowed. Content is top-left aligned; only right/bottom padding is added with normalized value 0, equivalent to raw channel means. Box offsets are `[0,0]`, actual integer content dimensions determine box scaling, and canvas size determines ROI pooling scales. Content sizes remain the clipping/postprocess reference, with predictions restored to original pixels before COCO `xywh` conversion. This is distinct from historical 800/1333 resizing and does not establish identical preprocessing to other architectures.

Actual image-weighted total training loss and complete-validation results:

| Epoch | Global step | Total loss | mAP (IoU 0.50:0.95) | AP50 | AP75 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3500 | 1.0828485061348017 | 0.11758046687581661 | 0.264503422035083 | 0.09627534727274363 |
| 2 | 7000 | 0.9590682450990592 | 0.148037401056455 | 0.31199640469346507 | 0.12314215361056938 |
| 3 | 10500 | 0.912056720873607 | 0.15872083492055766 | 0.33375560403834387 | 0.12963764592723348 |
| 4 | 14000 | 0.8848807721981513 | 0.17773109411680846 | 0.3619035810493331 | 0.15257382817657714 |
| 5 | 17500 | 0.8632093523663602 | 0.15999417439059063 | 0.33556640872907517 | 0.13625735523200688 |
| 6 | 21000 | 0.8456400707735281 | 0.16107028785650454 | 0.346054787012798 | 0.1340221030900302 |
| 7 | 24500 | 0.8310740645942943 | 0.16890162265567157 | 0.3537080332202199 | 0.14718643561720532 |
| 8 | 28000 | 0.8259068590121876 | 0.1644112879079097 | 0.35038669675249795 | 0.14523736435663695 |
| 9 | 31500 | 0.8080166099937632 | 0.17554160644711256 | 0.3739437412203296 | 0.14472928176358896 |
| 10 | 35000 | 0.7986755262462955 | 0.17702375663177553 | 0.37214856773506766 | 0.14525713983837651 |

- **Evaluation protocol:** Unchanged shared `evaluate_team.py` / category-aware bbox COCOeval, IoU 0.50:0.05:0.95, 101 recall thresholds, maxDets 1/10/100 and standard size ranges, on a 0-1 scale. Score threshold 0.05, NMS IoU 0.5 and maximum 100 detections per image are retained; the evaluator applies no extra score filter. Original image/category IDs and original-image-pixel COCO `xywh` output remain the export contract. The audit reads saved scores only and does not recompute them.
- **Best/latest identity:** `best.pth` has `epoch=4`, `global_step=14000`, `best_epoch=4`, `best_mAP=0.17773109411680846` and four history rows. Its final history row contains AP50 0.3619035810493331 and AP75 0.15257382817657714. `latest.pth` has `epoch=10`, `global_step=35000`, the same best epoch/mAP and ten history rows. Both retain the expected 11-output classification and 44-output box heads, full transform settings and SGD LR/momentum/weight-decay identity. Best selection is by complete-validation mAP, not the last epoch or AP50 alone.
- **Current JSON identity/coverage:** `validation.json` belongs to epoch 10, has `split=val`, `performance_result=true` and exactly the 1,000 original image IDs 9001-10000. `predictions.json` has 64,422 rows over exactly those IDs, with no missing or extra prediction image IDs. Prediction category IDs are `[1,2,3,4,5,7,8,9,10]`. Category 6 `train` has zero validation GT and AP/AP50/AP75 `Not evaluable`; it is excluded from means rather than zero-filled.

The following detailed saved metrics belong exclusively to **epoch 10**:

| Full-validation metric | Epoch 10 actual value (0-1) |
|---|---:|
| AP small | 0.0524937505145067 |
| AP medium | 0.19410183056589073 |
| AP large | 0.4045866031328996 |
| AR maxDets 1 | 0.15524041432313265 |
| AR maxDets 10 | 0.27884967516789716 |
| AR maxDets 100 | 0.29317983022421173 |
| AR small | 0.12122189507793019 |
| AR medium | 0.31933999352536196 |
| AR large | 0.5581966653726094 |

| Category | Epoch 10 AP |
|---|---:|
| person | 0.1415420476455141 |
| rider | 0.15975130825584433 |
| car | 0.3242483577871314 |
| truck | 0.2295652149617582 |
| bus | 0.22386071055812987 |
| train | Not evaluable |
| motor | 0.12133547681274046 |
| bike | 0.08145846914857677 |
| traffic light | 0.1155826971389361 |
| traffic sign | 0.19586952737734856 |

- **Unavailable best-epoch detail:** Epoch-4 per-class AP/AP50/AP75, size AP, recall and predictions were not saved separately and were overwritten by later JSON files. They remain unavailable from the retained artifacts. Epoch-10 details must not be relabeled as best-checkpoint results; the audit did not regenerate missing details. All retained per-class AP50/AP75 and counts remain in the epoch-10 validation JSON.
- **Comparison/interpretation:** Old 800/1333 run `20261020_182545_580059` achieved best mAP 0.25544452083836444; this fixed-640 run's best is 0.17773109411680846, an absolute reduction of 0.07771342672155598 or 7.7713426721555985 percentage points (approximately 30.42% relative). This is an observed configuration/run difference, not an isolated causal estimate. Total training loss decreased throughout, but validation mAP fluctuated after epoch 4; it rose from 0.1644112879079097 to 0.17554160644711256 and then 0.17702375663177553 in the last two transitions. Epoch 10 is only 0.0007073374850329328 below the best. Neither decreasing loss nor an earlier best epoch proves sufficient convergence, and no claim that further training will close the gap is made.
- **Related bounded diagnosis:** A later read-only code review and three targeted synthetic geometry/ROI/anchor groups found no clear coordinate, normalization, pooler-scale or head-wiring error; real detector/GPU execution was not checked. Comparing the old epoch-5 JSON with the new epoch-10 JSON shows small/medium/large AP changes of -4.612826706457051/-8.996767983784723/-1.6581967627164906 percentage points, respectively. Settings excluding transform, image coverage and per-class GT counts match. This latest-to-latest comparison is not best-to-best or a controlled causal estimate; all nine evaluable category APs declined, while large-object AR rose. Reduced effective resolution and increased padded-canvas sampling are hypotheses, not established causes. A controlled learning-rate schedule is a possible future experiment only; no code, configuration or new training was changed or executed. See `CURRENT_STATUS.md` for the bounded checks and limitations.
- **Audit scope/remaining results:** This stage only reads saved artifacts and synchronizes documentation. No actual training, detector construction, real-data inference, rescoring, timing benchmark, condition/test access, dataset/output/code/setting edits, dependency installation/download, commit or push was performed. The preparation-stage 32 synthetic tests were not rerun and are not current-audit test results. English text, local Markdown links and Git whitespace checks passed. Before/after SHA-256, sizes and modification times matched for all fifteen artifacts in the three runs plus the dataset README, mapping and train/val annotations (19 files); this is not a full-dataset audit. Only five documentation files changed, with no staged changes or tracked data/output files. Condition metrics remain `Not measured`; this repository's Faster R-CNN test evaluation remains `Not evaluated`. All three runs and their original artifacts are preserved.

## Input-Preprocessing Decision and Preparation: 2026-10-21

- **Status/scope:** The user authorized restoring the official Torchvision input defaults for normal Faster R-CNN training, with necessary checks and documentation synchronization only. This entry records a decision and code preparation, not a completed, failed or interrupted training experiment. At that preparation boundary no new 800/1333 ten-epoch run had started; its results were then `Not measured`. The subsequent user-completed run is recorded in the next entry.
- **Decision timing:** This choice was made after observing the completed fixed-640 ten-epoch run's lower validation performance. It is a validation-informed change and must be reported as such, rather than presented as a decision made before seeing those results. The old 800/1333 five-epoch run and the fixed-640 ten-epoch run differ in training duration and preprocessing; their comparison is not a strict, isolated resolution ablation and does not establish the cause of their performance difference.
- **Normal training input rule:** Restore native `GeneralizedRCNNTransform` with short-side target `min_size=(800,)`, long-side limit `max_size=1333`, aspect-preserving bilinear resize and no fixed square canvas. Apply the existing mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` normalization once inside the model. Use native zero batch padding to multiples of 32 and native `MultiScaleRoIAlign`. Dataset continues to output original-size images and original-pixel target boxes; native model target scaling and prediction postprocess retain original-coordinate COCO export.
- **640 reproducibility:** Keep the existing fixed-640 transform and canvas-aware ROI pooling helpers available for reproducing that historical configuration. Normal training no longer selects those helpers. Retain the completed 640 history, checkpoints, latest JSON files and their exact preprocessing metadata, without rewriting or relabeling them.
- **Unchanged configuration:** Keep `EPOCHS=10`, train/val batch size 2, SGD LR 0.005, momentum 0.9, weight decay 0.0005, seed 42, workers 0, `cuda:0`, print interval 50, train shuffle true, val shuffle false and float32. No scheduler, AMP, warmup or new augmentation is added. The ten-epoch target remains an observation window, not a convergence guarantee. Preserve the existing normalization, FrozenBatchNorm2d, backbone freezing and inference thresholds.
- **Fresh initialization and identity:** If the user later manually starts the normal command without `--resume`, create a new timestamped run using the existing cached official COCO_V1 detector weights and a new 11-class predictor. Do not automatically restore either old 800/1333 run or the fixed-640 run. Saved settings still include the full transform identity, and continuation still rejects missing or mismatched preprocessing/settings. Old checkpoints lacking the required transform metadata remain unsupported by that continuation check, even when their historical geometry used 800/1333.
- **Saving/evaluation:** Keep full 7,000-image training and complete 1,000-image validation each epoch, latest/best/history/validation/predictions saving and best selection by complete-validation mAP. Continue the same shared dataset splits, original image/category IDs, class mapping, COCO evaluator protocol and validation-based model-selection rule. The old best mAP 0.25544452083836444 belongs only to `20261020_182545_580059`; it is not the result of the newly prepared ten-epoch configuration.
- **Primary comparison rationale/limits:** Prefer each model's documented, justified official input rule for the primary configured-detector comparison, while keeping dataset, classes, evaluator and validation-selection policy consistent. Document actual input processing for every model. This does not claim equal compute budgets, identical preprocessing or an architecture-only causal effect. Preserve the fixed-640 result as part of the history rather than omitting an unfavorable run.
- **Checks/execution boundary:** All 34 synthetic tests passed in the existing `.venv`, with compilation and training CLI help. New tests exercise the current normal-entry input setting with a mocked detector factory, verify native transform/pooler identity and default geometry/coordinate restoration, and reject missing legacy or mismatched 640 preprocessing on resume. No real Faster R-CNN was constructed/trained/inferred; no real-data scoring, condition/test access, data/output modification, installation/download, commit or push occurred. The earlier 32-test result and bounded diagnosis remain historical evidence. All three original run directories are preserved. The assistant must not execute training or click Run unless explicitly authorized to execute training.

## Completed Restored 800/1333 Ten-Epoch Run: 20261021_022856_226381

- **Status/objective:** Fourth independent experiment, completed manually by the user after restoring the default input rule. Ten full 7,000-image training epochs, 35,000 updates and complete 1,000-image validation each epoch. The no-resume launch initializes from existing cached official COCO_V1 weights with a fresh 11-class head; no old checkpoint is continued. All earlier runs and their artifacts remain separate.
- **Evidence/provenance:** Inspected actual `history.csv`, `validation.json`, `predictions.json`, `best.pth` and `latest.pth`. Both checkpoints were loaded read-only on CPU with `weights_only=True` in the existing `.venv`; no detector was constructed. The inspected checkout was clean at `a39d332ddc1bc498be0c95a9f29ffb71040eaac1` when the audit began. Exact executed source/runtime identity is not independently recorded by the trainer; hardware/environment observations were reused rather than remeasured.
- **Configuration:** `EPOCHS=10`, `IMAGE_SIZE=None`, batches 2/2, seed 42, workers 0, `cuda:0`, print interval 50, train shuffle true / val shuffle false. SGD LR 0.005 throughout, momentum 0.9, weight decay 0.0005; float32 with no AMP, scheduler, warmup or augmentation. Official Faster R-CNN ResNet50-FPN COCO_V1 initialization, new 11-class head, unchanged FrozenBatchNorm2d/backbone freezing/normalization and inference thresholds (score 0.05, NMS 0.5, maximum 100 detections/image).
- **Preprocessing:** Original-size RGB float32 `[0,1]` Dataset images and original-pixel target boxes. Normalize once with mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`; native aspect-preserving bilinear resize, short-edge target 800 / long-edge cap 1333, `align_corners=false`, `recompute_scale_factor=true`; native right/bottom normalized-zero padding to multiples of 32, no fixed square canvas, native ROI pooling. Content dimensions control box scaling/clipping and postprocess to original pixels before COCO `xywh` export. Actual transform settings agree in both checkpoints and the latest report.
- **Evaluation:** Unchanged fixed splits/classes/original IDs and shared category-aware COCO bbox evaluator: IoU 0.50:0.05:0.95, 101 recall points, maxDets 1/10/100 and standard area ranges. No new score filtering or rescoring during this audit. Both old and new latest reports cover the same 1,000 image IDs / 18,465 GT objects and share evaluation settings outside the expanded transform metadata.
- **Time:** Exact start/end timestamps, elapsed duration and peak memory are `Not measured`; the trainer does not measure them. Directory names and file modification times are not duration measurements.

Actual image-weighted training loss and saved full-validation results (metric scale 0-1):

| Epoch | Global step | Total loss | mAP (IoU 0.50:0.95) | AP50 | AP75 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3500 | 0.9971091929755307 | 0.195144981849982 | 0.40668205491776194 | 0.1646870359604207 |
| 2 | 7000 | 0.8961587963152144 | 0.23389168976056873 | 0.47181043534958367 | 0.19798646216993845 |
| 3 | 10500 | 0.8561116008582924 | 0.23765010969605666 | 0.49473986268077863 | 0.19038821601933834 |
| 4 | 14000 | 0.8299073457889524 | 0.25137010647794716 | 0.5140678430977413 | 0.2095660880006186 |
| 5 | 17500 | 0.8157637486882242 | 0.23506753835948324 | 0.49805800364121766 | 0.18322952429708936 |
| 6 | 21000 | 0.7978298019894532 | 0.233550760030251 | 0.4816063646324043 | 0.19753675665999915 |
| 7 | 24500 | 0.7881503465442387 | 0.2519508348597597 | 0.5155850158288147 | 0.2030129302252958 |
| 8 | 28000 | 0.7736049120045666 | 0.2416734699434145 | 0.4823699226553351 | 0.21798809906720185 |
| 9 | 31500 | 0.7603901787392928 | 0.24587131019345576 | 0.4962756341000956 | 0.21592809220434162 |
| 10 | 35000 | 0.7528044606677057 | 0.24015201202561187 | 0.4924967102524228 | 0.20027806460803751 |

- **Best/latest identity:** `best.pth` is epoch 7 / global step 24,500, best mAP 0.2519508348597597 and seven history rows; AP50 0.5155850158288147 and AP75 0.2030129302252958. `latest.pth` is epoch 10 / step 35,000 with ten history rows and the same best identity. Latest mAP/AP50/AP75 are 0.24015201202561187 / 0.4924967102524228 / 0.20027806460803751. Both saved heads have 11 classifier outputs and 44 box outputs; SGD groups retain the expected parameters.
- **JSON identity/coverage:** The report explicitly identifies epoch 10, `split=val`, `performance_result=true` and all 1,000 original image IDs 9001-10000. Predictions contain 61,137 rows over exactly those IDs; predicted categories are `[1,2,3,4,5,7,8,9,10]`. Category 6 railway `train` has zero GT; its AP/AP50/AP75 are `Not evaluable`, excluded from means. Epoch-7 detailed metrics/predictions were overwritten by later JSON writes and are not retained separately. The following detailed tables belong only to **epoch 10**, never the best checkpoint.

| Latest full-validation metric | Epoch 10 value |
|---|---:|
| bbox_mAP | 0.24015201202561187 |
| bbox_AP50 | 0.4924967102524228 |
| bbox_AP75 | 0.20027806460803751 |
| bbox_AP_small | 0.09581938387065826 |
| bbox_AP_medium | 0.28606043272792203 |
| bbox_AP_large | 0.4255061936776634 |
| bbox_AR_max1 | 0.1825208029621535 |
| bbox_AR_max10 | 0.35237557369223627 |
| bbox_AR_max100 | 0.37332992290560874 |
| bbox_AR_small | 0.18877485464013857 |
| bbox_AR_medium | 0.4305536493201595 |
| bbox_AR_large | 0.5666718598635181 |

| Category | Epoch 10 AP |
|---|---:|
| person | 0.23648955689577647 |
| rider | 0.15216148281496825 |
| car | 0.40450197430887525 |
| truck | 0.2885852694575507 |
| bus | 0.2923567531505868 |
| train | Not evaluable |
| motor | 0.1865043535187716 |
| bike | 0.10734677109256865 |
| traffic light | 0.19977980581088847 |
| traffic sign | 0.2936421411805211 |

- **Interpretation/next decision:** Best mAP 0.2519508348597597 is below the old five-epoch best 0.25544452083836444 by 0.0034936859786047214, or 0.34936859786047214 percentage points (approximately 1.37% relative). It is substantially above the fixed-640 best 0.17773109411680846. These are observed run/configuration differences, not controlled causal or statistically significant effects. Epoch 7 is slightly better than epoch 4; validation then fluctuates without sustained gains while training loss keeps decreasing. More identical constant-LR epochs are not justified by a promise of improvement. The old five-epoch epoch-4 checkpoint remains the highest observed validation candidate, but no final baseline, learning-rate schedule or further training has been selected in this review.
- **Audit limits:** Documentation synchronization and saved-artifact inspection only. No actual training, detector construction, inference, rescoring, benchmark, condition/test access, dataset/output changes, installation/download, commit or push. The historical 34 synthetic tests were not rerun for this documentation-only stage. Condition metrics remain `Not measured` and this repository's test evaluation remains `Not evaluated`. English/whitespace/local Markdown links, logged values versus saved history/report, full original validation-ID coverage and Git scope/whitespace checks passed. All twenty run artifacts plus four dataset contract files retained identical SHA-256, sizes and modification times (24 files); this is not a full-dataset audit. Exactly five documentation files changed, with no staged changes or tracked data/output files.

## Historical Preparation of Old Five-Epoch Continuation: 20261021_legacy5_to_epoch10

This section records the original preparation boundary. The user subsequently completed the continuation, as recorded in the following entry; the preparation manifests were not rewritten to claim completion.

- **Status/scope:** Prepared / Not started on 2026-10-21. This is a preparation record, not a completed, failed or interrupted experiment. The assistant has not executed training or clicked Run; execution remains user-only unless explicitly authorized. Zero new epochs/optimizer updates, no new validation result. New mAP/AP50/AP75/per-class/size/recall/duration are `Not measured`; condition results remain `Not measured` and final test is `Not evaluated`.
- **Parent and branch point:** Original `outputs/training/20261020_182545_580059/latest.pth`, completed epoch 5 / global step 17,500. New independent output directory: `outputs/training/20261021_legacy5_to_epoch10/`. The planned next epoch is 6, total target 10, adding five full epochs / 17,500 updates if completed. Its history rows 1-5 are inherited measurements, not newly executed epochs. All four original experiment directories/files remain preserved.
- **Explicit compatibility evidence:** Direct original-checkpoint resume still rejects missing `settings.transform` / `settings.num_workers`. The separate CPU-only `src.prepare_continuation` restricts conversion to the audited parent and fingerprints all five artifacts. Saved epoch-5 transform mean/std/min/max/padding/fixed-size fields, historical `a65cd736` train/model source and installed native Torchvision establish only the two missing settings. `continuation.json` records each field's basis, parent/source paths and SHA-256 fingerprints, branch point, preparation runtime and plan. Existing model/SGD/momentum/counters/history/Python/NumPy/Torch CPU/CUDA RNG values are preserved; `continuation` and `best_source` are added. Global settings validation stays strict; no automatic migration of other legacy checkpoints or weights-only fallback is introduced.
- **Inherited best and report identity:** Parent `best.pth`, epoch 4 / step 14,000, mAP 0.25544452083836444, AP50 0.5170826995660548 and AP75 0.21488694482832407, is copied compatibly with its state preserved. Original parent best path/hash remain in `best_source` until a new complete-validation mAP strictly exceeds it; ties preserve the earlier best. Future checkpoints retain lineage and best identity, while future validation reports distinguish actual evaluated `epoch` from `best_model`. Parent `validation.json` / `predictions.json` remain epoch 5, mAP 0.23822375438048712, and are not copied into the branch. Missing epoch-4 detailed reports/predictions are not regenerated.
- **Planned unchanged configuration:** `EPOCHS=10` is the total target including inherited epochs. Native aspect-preserving 800/1333 resize, normalized-zero padding to multiples of 32 and native ROI pooling; original-size RGB `[0,1]` Dataset output and original-pixel boxes, one internal normalization. Batches 2/2, seed 42, workers 0, `cuda:0`, SGD LR 0.005 / momentum 0.9 / weight decay 0.0005, float32, unchanged backbone freezing and inference thresholds 0.05/0.5/100. No scheduler, AMP, warmup or augmentation. Each new epoch must traverse all 7,000 train images and evaluate all 1,000 validation images using the unchanged shared category-aware COCO bbox protocol and original IDs. Absent-GT railway `train` remains `Not evaluable`.
- **Verification:** All 41 synthetic CPU tests passed with `CUDA_VISIBLE_DEVICES=-1`, compilation of `src` / `tests` and training/preparation CLI help. Independent CPU `weights_only=True` disk roundtrips confirmed every original checkpoint key/state value except the evidenced settings additions and lineage fields. Each checkpoint retains 295 finite model tensors, 72 finite nonzero momentum buffers and all four RNG families including one saved CUDA state; counters/history remain latest 5/17,500/five rows and best 4/14,000/four rows. `checkpoint_verification.json` records the comparison. Train/val annotation JSON and mapping confirm 7,000 / 128,949 and 1,000 / 18,465 image/box counts, unchanged categories and absent validation category 6, without decoding images. Cached official weights exist (167,502,836 bytes) without loading/downloading them. No real detector construction/execution, training, real-data inference/scoring, benchmark, test access or GPU restoration/trajectory test occurred. No dependency installation, weight download, dataset changes, commit or push occurred.
- **Preservation:** All twenty original files across the four completed training directories retained identical SHA-256, sizes and modification times before/after preparation/testing, as recorded in `checkpoint_verification.json`. This is an original-run artifact audit, not a new dataset hash audit. No original dataset file was written. The five pre-existing uncommitted result-documentation changes were preserved, with no staged or tracked data/output files.
- **Provenance and interpretation limits:** The historical trainer did not record executed source/runtime/data hashes. The added metadata is supported by saved fields and source evidence, without independent proof of exact original execution identity. Saved RNG is preserved and restored before the next shuffled iterator; numerical equality to uninterrupted CUDA training is unverified. Five additional constant-LR epochs do not guarantee improvement or convergence. No final baseline, weather analysis, cloud migration or report PDF change is included.

Prepared user-only command from the repository root; not executed by the assistant:

```powershell
& .\.venv\Scripts\python.exe -m src.train --resume .\outputs\training\20261021_legacy5_to_epoch10\latest.pth
```

## Completed Old Five-Epoch Continuation: 20261021_legacy5_to_epoch10

- **Status and lineage:** Completed manually by the user. Parent `outputs/training/20261020_182545_580059/latest.pth` supplied completed epoch 5 / step 17,500, with full model/SGD/momentum/history/best/RNG state prepared as documented above. This branch added epochs 6-10, five complete training/validation passes and 17,500 optimizer updates. Total epoch 10 / step 35,000 includes the inherited first five rows, which exactly match the parent; it is not a fifth independent experiment or ten new epochs.
- **Evidence and configuration:** Actual `history.csv`, `validation.json`, `predictions.json`, `latest.pth` and `best.pth` were inspected. CPU loading used `weights_only=True`; latest/best consistency, settings, lineage and 72 finite momentum buffers per checkpoint passed inspection. Review began with a clean working tree at `579cd7a3`; exact executed source/runtime identity is not independently recorded by the trainer. Unchanged native 800/1333 transform/padding/ROI pooling and one internal normalization; original-size Dataset images/boxes; batches 2/2, SGD LR 0.005 / momentum 0.9 / weight decay 0.0005, seed 42, workers 0, `cuda:0`, float32, no scheduler/AMP/warmup/augmentation, unchanged freezing and thresholds. Every added epoch used all 7,000 training images and all 1,000 validation images with the shared category-aware COCO bbox protocol and original IDs.

Actual new-epoch results, on a 0-1 metric scale; epochs 1-5 remain recorded in the parent experiment above:

| Epoch | Global step | Total loss | mAP (IoU 0.50:0.95) | AP50 | AP75 |
|---:|---:|---:|---:|---:|---:|
| 6 | 21000 | 0.798129555279123 | 0.24640704578621095 | 0.4934787846772891 | 0.20326056738294865 |
| 7 | 24500 | 0.7855951697832373 | 0.24794138977720387 | 0.5145997757231647 | 0.20912144855064338 |
| 8 | 28000 | 0.7721615708709828 | 0.24324549105981796 | 0.49338598968937986 | 0.20293218049444656 |
| 9 | 31500 | 0.7651760260863895 | 0.22546045840518608 | 0.4686647231198381 | 0.18345379593799194 |
| 10 | 35000 | 0.7575437616668641 | 0.24465600637621113 | 0.4986412390330815 | 0.20920142467848557 |

- **Best/latest identity:** No added epoch exceeded inherited best mAP 0.25544452083836444. `best.pth` remains epoch 4 / step 14,000 / four history rows, AP50 0.5170826995660548 and AP75 0.21488694482832407, with parent source/hash retained in `best_source`. Its hash still matches the prepared best copy. `latest.pth` is epoch 10 / step 35,000 / ten history rows. Current JSON files belong to epoch 10; `best_model` separately identifies the parent epoch-4 best. The latest report has `performance_result=true`, all 1,000 original validation IDs 9001-10000 and 60,067 predictions over exactly those IDs. Railway `train` has zero GT and AP/AP50/AP75 `Not evaluable`. Latest per-class/size/recall values remain in `validation.json`; they are not epoch-4 best-model metrics. Missing epoch-4 detailed JSON/predictions have not been regenerated.
- **Interpretation:** Best new epoch 7 mAP 0.24794138977720387 is 0.7503131061160573 percentage points below the inherited best. Training loss decreased from parent epoch-5 0.8152584325985185 to epoch-10 0.7575437616668641, approximately 7.08%, while validation fluctuated without sustained gains. This supports a validation plateau under the unchanged constant-LR 0.005 recipe, not proof of strict optimization convergence, the model's theoretical performance ceiling, a causal explanation or statistically established overfitting. No new training recipe or final baseline is selected.
- **Preservation and limits:** All five parent artifact hashes match the preparation manifest; all twenty original artifacts retain their pre-preparation SHA-256, sizes and modification times. `continuation.json` / `checkpoint_verification.json` remain historical preparation records; current completion comes from actual latest/history/report. Exact start/end time, elapsed duration and peak memory are `Not measured`. This review only reads saved artifacts and synchronizes five documents: no new training, detector construction, inference, rescoring, benchmark, GPU/device check, condition/test access, dataset/output edits, installation/download, commit or push. The historical 41 synthetic tests were not rerun. Condition metrics remain `Not measured` and final test is `Not evaluated`.

## Full Validation of the Retained Epoch-4 Checkpoint: 20261021_143617_089705_epoch4_best

- **Status and scope:** Completed once on 2026-10-21 under explicit user authorization. This is checkpoint-only evaluation, not a new training experiment: zero updates, no condition analysis, no test access, no model/recipe selection. Earlier entries' unavailable epoch-4 detail describes what was retained at those audit boundaries; this new pass now supplies the missing per-category/size/recall metrics and predictions without replacing original training outputs.
- **Source and setup:** Original `outputs/training/20261020_182545_580059/best.pth`, epoch 4 / step 14,000, SHA-256 `53945d6e5bfddaff18db0c4dce07a300b56369172f5ebf8d9e028add1b992577`. All 295 model-state tensors matched the prepared inherited reference. Reused existing model/evaluator code, native 800/1333 transform/pooler and original ROI/RPN/normalization/score 0.05/NMS 0.5/max 100 settings. `cuda:0`, batch size 2, workers 0 and seed 42; original val split only, 1,000 IDs 9001-10000 / 18,465 GT objects. Shared category-aware COCO bbox protocol and original IDs/COCO xywh export are unchanged.
- **Artifacts and coverage:** `outputs/evaluation/20261021_143617_089705_epoch4_best/` contains the [readable validation summary](../outputs/evaluation/20261021_143617_089705_epoch4_best/validation_summary.md), `validation.json`, `predictions.json`, [metrics.csv](../outputs/evaluation/20261021_143617_089705_epoch4_best/metrics.csv), [per_category.csv](../outputs/evaluation/20261021_143617_089705_epoch4_best/per_category.csv), `run_manifest.json`, self-contained `run_evaluation.py` and source-integrity records. Predictions total 69,740 over all 1,000 validation IDs. Per-category CSV records AP/AP50/AP75 and GT counts; absent-GT category 6 `train` is `Not evaluable`, excluded from means. Parent latest JSON remains epoch 5; continuation latest JSON remains epoch 10. Those files do not supply these new epoch-4 results.

New full-validation measurements (0-1 scale):

| Metric | Epoch-4 checkpoint value |
|---|---:|
| mAP (IoU 0.50:0.95) | 0.25544452083836444 |
| AP50 | 0.5170826995660548 |
| AP75 | 0.21488694482832407 |
| AP small | 0.10212307453157807 |
| AP medium | 0.29971580854025076 |
| AP large | 0.4520452156300116 |
| AR maxDets 1 | 0.1976618281798444 |
| AR maxDets 10 | 0.3727919952948325 |
| AR maxDets 100 | 0.3937052480806767 |
| AR small | 0.21973632502911689 |
| AR medium | 0.4553823602648104 |
| AR large | 0.5825670098195601 |

- **Reproduction and preservation:** mAP/AP50/AP75 each differ from historical epoch 4 by exactly 0.0. All 295 model-state tensors stayed unchanged after inference and no parameter gradients were created. Integrity checks passed for 1,037 protected files: 27 existing artifacts, six source modules, cached weights, dataset README/mapping/val annotations and 1,000 val images. No train/test data enumeration, original artifact/data edits or tracked-source/test changes occurred. Historical synthetic suites were not rerun; script syntax/review checks passed. No installation/download, commit or push.
- **Runtime and limits:** Existing Python 3.13.15, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130, CUDA 13.0 and RTX 5070. Manifest start/end are 2026-10-21 14:38:14.501436 to 14:39:07.408354 EDT (UTC-04:00); the timed evaluation portion is 48.30145260000427 seconds, not a historical training duration or dedicated throughput benchmark. Historical training durations remain `Not measured`. This is one additional checkpoint evaluation beyond the 31 training-associated complete validation cycles, with no new training result or convergence claim. Condition metrics remain `Not measured`; final test is `Not evaluated`.

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
