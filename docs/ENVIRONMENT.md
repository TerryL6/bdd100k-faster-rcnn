# Environment Setup and Verification

Last verified: 2026-10-19.

## Stage Status

The isolated project environment has been created at `.venv` using the existing CPython 3.13.15 installation. All selected dependencies are installed, `pip check` passed, and the CPU/CUDA operator and synthetic COCOeval checks passed inside this environment. Global packages were not changed.

During the environment-setup stage, no pretrained weights were downloaded, no Faster R-CNN detector was constructed, and no training or evaluation on dataset images was performed. COCOeval was exercised only with tiny synthetic ground truth and predictions held in memory. Later stages reused this environment for model integration, a three-step SGD/checkpoint check, and the epoch-driver's bounded three-train/two-val-image integration/resume check plus 14 synthetic unittest checks; see [Current Status](CURRENT_STATUS.md). The later readability refactor reused the same environment for nine simplified synthetic checks and a three-train/two-val-image GPU continuation check. No dependencies were installed or changed for those stages. The shared-team evaluator integration later passed 18 synthetic tests and an archived two-image prediction comparison in the same environment, without new inference or training. The evaluator no longer injects a project-external `.runtime_deps` path. Full training and full-validation performance remained unmeasured at that integration stage.

The user later completed independent one-epoch (`20261020_155352_056862`) and five-epoch (`20261020_182545_580059`) runs using 800/1333 resizing, then ten epochs with a fixed 640 canvas in `20261021_002448_944703`. All three used this existing environment; results are in [Experiment Log](EXPERIMENT_LOG.md). Historical 640 preparation passed 32 synthetic tests. After the completed-run review, the user authorized restoring the default 800/1333 input rule. That restoration passed 34 synthetic tests, compilation and CLI help with the same `.venv`, without real detector construction/training/inference, real-data scoring, a benchmark, test access, installation/download or cloud work. The user subsequently completed the fourth independent run, `20261021_022856_226381`, with restored default inputs and ten epochs; its best mAP is 0.2519508348597597 at epoch 7. The later artifact review loaded checkpoint metadata on CPU without detector execution or environment changes. Hardware/runtime observations below retain their original verification date; this is not a new environment audit.

The old-five-epoch continuation under `outputs/training/20261021_legacy5_to_epoch10/` is **Prepared / Not started**, using the same `.venv` without dependency installation or weight download. Its explicit preparation reads original checkpoints with `map_location="cpu"`, `weights_only=True`, preserves model/SGD/momentum/history/counters and Python/NumPy/Torch CPU/CUDA RNG values, and records preparation-runtime versions in `continuation.json`. All 41 synthetic CPU tests passed with CUDA hidden, plus compilation and training/preparation CLI help. Independent disk roundtrips confirmed exact preservation of all original state values, including 295 finite model tensors, 72 finite nonzero SGD momentum buffers and four RNG families per checkpoint; `checkpoint_verification.json` records the results. Cached official weights exist (167,502,836 bytes) without loading/downloading them. No actual detector construction/execution, training, real-data inference/scoring, benchmark or test access occurred. The intended runtime remains `cuda:0` with workers 0, batch sizes 2/2 and the unchanged pinned setup. Original executed source/runtime/data hashes were not recorded, so saved-state recovery does not establish exact historical execution identity or bitwise CUDA trajectory. Saved CUDA RNG was preserved on CPU; CUDA restoration/trajectory was not executed. New epoch-6-to-10 metrics/duration are `Not measured`.

## Observed Host

| Item | Observed value |
|---|---|
| OS | Windows 11 Pro, x64, version 10.0.26300 |
| CPU | AMD Ryzen 7 9850X3D, 8 cores / 16 logical processors |
| System memory | 31.19 GiB reported by Windows |
| GPU | NVIDIA GeForce RTX 5070 |
| GPU memory | 12 GB class; `nvidia-smi` reports 12,227 MiB |
| GPU compute capability | 12.0 |
| NVIDIA driver | 617.42 |
| Driver CUDA UMD version | 13.4, reported by `nvidia-smi` |
| Python | CPython 3.13.15, standard 64-bit build |
| Python launcher | `py -3.13` resolves to the existing Python 3.13 installation |
| Project environment | `.venv`, isolated from global site-packages |
| pip | 26.2.1 in both the project and existing global environment |
| Standalone CUDA compiler | `nvcc` not found on PATH |

Driver CUDA capability, the runtime bundled with PyTorch, and a separately installed CUDA Toolkit are different things. The working PyTorch package uses CUDA 13.0 despite the driver's CUDA UMD 13.4 report. The current prebuilt-wheel workflow does not require installing a standalone CUDA Toolkit or finding `nvcc` on PATH.

## Verified Project Packages and Checks

| Package | Installed project version | Verification |
|---|---|---|
| torch | 2.14.0+cu130 | Import, CPU/CUDA operations passed |
| torchvision | 0.29.0+cu130 | Import, CPU/CUDA NMS and ROIAlign passed |
| numpy | 2.5.3 | In-memory image roundtrip with Pillow and Torch passed |
| Pillow | 12.3.0 | In-memory image roundtrip with NumPy and Torch passed |
| pycocotools | 2.0.11 | COCO/COCOeval imports and synthetic bbox evaluation passed |

Checks executed with `.venv/Scripts/python.exe`:

- The interpreter and installed package locations resolve inside `.venv`; `pyvenv.cfg` specifies `include-system-site-packages = false`.
- `torch.cuda.is_available()` returned `True` and identified the RTX 5070.
- `torch.version.cuda` reported `13.0`; cuDNN reported `92400`; compiled CUDA architectures include `sm_120`.
- CPU and CUDA matrix multiplication matched expected numeric results.
- CPU and CUDA `torchvision.ops.nms` returned the expected retained indices.
- CPU and CUDA `torchvision.ops.roi_align` matched expected numeric results.
- A small CUDA convolution produced finite output with the expected shape.
- Pillow encoded and decoded a tiny RGB PNG in memory; NumPy and Torch conversions preserved the expected pixel values.
- `python -m pip check` reported no broken requirements in `.venv`.
- COCO and COCOeval imported successfully. A one-image synthetic bbox case produced AP approximately 1 for an exact detection and 0 for a disjoint detection, as expected.
- The COCO checks preserved a non-positional synthetic image ID, used no BDD100K data, and produced no model performance measurements.
- Final direct dependency versions match `requirements.txt`; all 15 installed distribution versions, including pip and setuptools, are captured in `requirements-lock.txt`.

These are environment checks with synthetic inputs, not model experiments. They do not by themselves validate the complete Faster R-CNN pipeline, dataset loader, training memory requirements, checkpoint handling, or full-dataset evaluation. No training experiment entry was added for the environment stage. The later bounded real-update check is recorded separately in `docs/EXPERIMENT_LOG.md`.

## Installed Dependencies and Isolation

The existing standard CPython 3.13 installation was used to create a clean project-local `.venv`. The working Torch pair was preserved, with a separate installation inside `.venv` and no inheritance from global site-packages.

| Direct dependency | Installed version | Reason |
|---|---|---|
| torch | 2.14.0+cu130 | Already works on this GPU; official Windows CPython 3.13 wheel exists |
| torchvision | 0.29.0+cu130 | Official pair for Torch 2.14.0; detection operators already pass |
| numpy | 2.5.3 | Match the existing environment |
| Pillow | 12.3.0 | Match the existing environment |
| pycocotools | 2.0.11 | Standard COCO API/COCOeval with a compatible Windows ABI3 wheel |

`requirements.txt` pins the five direct dependencies. `requirements-lock.txt` records the complete 15-package resolved version snapshot from `pip freeze --all`, including transitive dependencies and packaging tools. The snapshot is specific to this Windows x64, CPython 3.13, CUDA 13.0 setup; it is not an artifact-hash or cross-platform lock. Torchaudio, a custom CUDA build, and additional training frameworks were not installed.

The existing global Torch, Torchvision, NumPy, Pillow, and pip versions were rechecked after installation and remain unchanged; global pycocotools is still absent. The project environment contains pycocotools independently.

The environment's file lengths total approximately 3.05 GiB at setup time. This excludes the shared pip download cache and filesystem allocation overhead. No pip cache was deleted.

## Recorded Installation Commands

These commands were executed from the repository root in PowerShell. They also describe setup in a fresh clone where `.venv` does not yet exist. Routine project use does not require recreating the existing environment.

```powershell
py -3.13 -m venv .venv

& .\.venv\Scripts\python.exe -m pip --isolated --disable-pip-version-check install --only-binary=:all: --progress-bar off torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130

& .\.venv\Scripts\python.exe -m pip --isolated --disable-pip-version-check install --only-binary=:all: --progress-bar off -r requirements.txt --index-url https://pypi.org/simple

& .\.venv\Scripts\python.exe -m pip --isolated --disable-pip-version-check check
```

Install the Torch pair from its official CUDA index first. The second install resolves the remaining packages from the standard package index while keeping the exact Torch builds already installed. Running only `pip install -r requirements.txt` in a fresh environment is insufficient because the CUDA-specific Torch builds require the official Torch index. Do not use the CUDA index for the entire requirements file: its NumPy mirror did not contain the selected 2.5.3 release at inspection time, while PyPI did.

The first installation temporarily resolved NumPy 2.5.2 from the CUDA index; the second installation replaced it with the pinned 2.5.3 inside `.venv`. For future recreation of all tested versions, replace `-r requirements.txt` in the second installation command with `-r requirements-lock.txt`, then repeat the functional checks. That full snapshot was recorded after the successful installation; it was not the input to the original installation.

Use `.venv` without `--system-site-packages`. Calling its Python directly avoids PowerShell activation-policy changes and accidental use of global pip. The `--isolated` pip option avoids user-level pip settings and environment-variable overrides; package indexes are explicitly selected. Do not add `--extra-index-url` or replace the selected CUDA builds with an unverified CPU/older-GPU build.

## Using the Environment

From the repository root, invoke the project interpreter directly:

```powershell
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

When configuring an editor, select `.venv/Scripts/python.exe` as this project's interpreter. Shell activation is optional; no PowerShell execution policy was changed during setup. Keep `.venv/` ignored and recreate it from the recorded dependencies on another machine rather than copying the directory.

Repeat version, dependency, CPU/CUDA operator, image-conversion, and synthetic COCOeval checks after recreating or changing the environment. Data-loading verification is recorded separately in [Current Status](CURRENT_STATUS.md); environment checks alone do not validate a dataset adapter or model pipeline. Current provisional settings were used for the completed default-input run with `IMAGE_SIZE=None` and ten epochs. Exact elapsed duration and peak memory use were not recorded by the trainer. The completed 640 run's exact elapsed duration and peak memory use were not recorded by the trainer.

## Official Compatibility References

- [PyTorch 2.14.0 general availability](https://dev-discuss.pytorch.org/t/pytorch-2-14-0-general-availability/3431): publishes Torch 2.14.0 and Torchvision 0.29.0 together.
- [Official CUDA 13.0 Torch wheel index](https://download.pytorch.org/whl/cu130/torch/) and [Torchvision wheel index](https://download.pytorch.org/whl/cu130/torchvision/): contain the selected `cp313-cp313-win_amd64` builds.
- [NVIDIA CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html): CUDA 13.x requires driver branch 580 or later. The observed 617.42 satisfies that condition, and actual CUDA operator checks also passed.
- [NVIDIA CUDA GPU table](https://developer.nvidia.com/cuda/gpus): identifies RTX 5070 as compute capability 12.0.
- [PyTorch maintainer explanation of bundled CUDA dependencies](https://discuss.pytorch.org/t/should-i-install-the-extra-cudatoolkit-and-cudnn/194528): prebuilt packages include the runtime dependencies for this workflow.
- [pycocotools 2.0.11 release files](https://pypi.org/project/pycocotools/2.0.11/): includes `pycocotools-2.0.11-cp312-abi3-win_amd64.whl`.
- [NumPy 2.5.3 release files](https://pypi.org/project/numpy/2.5.3/#files): includes `numpy-2.5.3-cp313-cp313-win_amd64.whl`.
- [Pillow 12.3.0 release files](https://pypi.org/project/pillow/12.3.0/#files): includes `pillow-12.3.0-cp313-cp313-win_amd64.whl`.
- [Python packaging compatibility tags](https://packaging.pypa.io/en/stable/tags.html) and [pycocotools build configuration](https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/setup.py): explain stable ABI support. The `cp312-abi3` wheel is compatible with this standard CPython 3.13 interpreter; a `cp313t` free-threaded wheel is a different build.
