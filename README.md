# SMF-ECSeg-Morphology-Aware-Wafer-Surface-Defect-Segmentation-with-Detail-and-Boundary-Supervision
Wafer surface defect instance segmentation with SDC, MABG and FBAS, including WSDIS-4 data access and RK3588/Jetson TX2 deployment tools.
The password for the Quark Cloud Drive can be requested from the corresponding author.https://pan.quark.cn/s/c7561634211a
# SMF-ECSeg: Morphology-Aware Segmentation of Wafer Surface Defects with Detail Compensation and Boundary-Aware Supervision

 [Modules](docs/MODULES.md) | [Dataset](datasets/README.md) | [Deployment](docs/DEPLOYMENT.md)

Source code for **SMF-ECSeg**, an instance segmentation method for multi-morphology defects on wafer surfaces. The method builds on the Small instance segmentation configuration of EdgeCrafter (ECSeg-S).

The repository contains the three proposed components, their integration into the training criterion and feature graph, ablation configurations, and deployment tools for **LubanCat-5 RK3588** and **NVIDIA Jetson TX2**.

## Proposed components

| Component | Implementation | Purpose |
|---|---|---|
| **SDC** — Semantic-guided Detail Compensation | [sdc.py](ecdetseg/engine/edgecrafter/modules/sdc.py) | Adds genuine stride-8 stem details to the semantic P3 representation using a semantic gate and learnable residual scaling. |
| **MABG** — Morphology-Adaptive Background Gating | [mabg.py](ecdetseg/engine/edgecrafter/modules/mabg.py) | Combines local, horizontal-strip, vertical-strip and dilated depthwise features with spatial branch weighting and contextual gating. |
| **FBAS** — Foreground and Boundary-Aware Supervision | [fbas.py](ecdetseg/engine/edgecrafter/modules/fbas.py) | Allocates 25%/25%/50% of a fixed point budget to foreground/boundary/original mixed sampling and adds a tolerant soft-boundary F1 loss. Training only. |

`WaferDetailAdapter` and `WaferMorphologyBlock` retain their original class names and parameter keys for checkpoint compatibility. The full criterion integration is in [criterion.py](ecdetseg/engine/edgecrafter/criterion.py); it includes the existing auxiliary and denoising mask outputs.
## Dataset

**WSDIS-4 — Wafer Surface Defect Instance Segmentation Dataset** contains four categories: Multi-ball, Scratch, Particle and Stain. 35,000 images and 70,172 instances, with 28,000/3,500/3,500 images in the training/validation/test splits. Image size is 640 × 640.
## Installation

Use Python 3.11 for the training/export environment. Install matching PyTorch and torchvision builds for your CUDA version, then:

```bash
python -m pip install -r requirements.txt
```

The source environment used PyTorch 2.8.0+cu128, torchvision 0.23.0, NumPy 2.0.1 and PyYAML 6.0.3. These describe the source environment, not mandatory Jetson/RK3588 packages. Board environments are separate.
## Configure your paths

Copy `configs/paths.example.yml` to `configs/paths.local.yml`, then fill the required image and annotation paths. Relative paths resolve against the paths file. Local path files are ignored by Git.

```yaml
train_images: ""
train_annotations: ""
val_images: ""
val_annotations: ""
test_images: ""
test_annotations: ""
pretrained_checkpoint: ""
resume_checkpoint: ""
inference_checkpoint: ""
```
## Training and evaluation

Run from the repository root:

```bash
# Full SMF-ECSeg
python train.py --config configs/smf_ecseg.yml --paths configs/paths.local.yml --device cuda:0 --seed 0

# ECSeg-S baseline
python train.py --config configs/ecseg_s_baseline.yml --paths configs/paths.local.yml --device cuda:0 --seed 0

# Evaluate a trained checkpoint on the test split
python train.py --config configs/smf_ecseg.yml --paths configs/paths.local.yml --test-only --split test --device cuda:0
```
## Image inference

The following arguments are placeholders to fill with your own files:

```bash
python infer.py --config configs/smf_ecseg.yml --checkpoint "$CHECKPOINT" --images "$IMAGES" --output outputs/predictions --device cuda:0
```
## Validation

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

## Attribution and citation

The initial [release notes](docs/RELEASE_NOTES.md) summarize the source package.

Based on [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter). See [LICENSE.md](LICENSE.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [source provenance](SOURCE_PROVENANCE.json). The inherited EdgeCrafter materials permit non-commercial research use under their supplied license; they are not relicensed as MIT or Apache-2.0 by this repository.



