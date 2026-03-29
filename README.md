# VAMOS-OCTA: Vessel-Aware Multi-Axis Orthogonal Supervision for Inpainting Motion-Corrupted OCT Angiography Volumes

🏆 SPIE Medical Imaging 2026 — Image Processing Best Student Paper Award  
🎤 Selected for Oral Deep-Dive Presentation  
📄 Paper: [arXiv:2602.00995](https://arxiv.org/pdf/2602.00995)

Handheld OCT Angiography (OCTA) enables retinal imaging in uncooperative patients but is highly susceptible to motion artifacts, including fully corrupted B-scans. **VAMOS-OCTA** is a deep learning framework for inpainting motion-corrupted B-scans in 3D OCTA volumes while preserving vessel structure in both cross-sectional and projection views.

<p align="center">
  <img src="/figures/vamos_framework.png" alt="VAMOS Framework" width="800"/>
</p>

<p align="center">
  <img src="/figures/mip_results_vamos.png" alt="VAMOS projection results" width="800"/>
</p>

## Highlights

- **2.5D U-Net** inpainting model conditioned on neighboring B-scans
- **VAMOS loss** combining vessel-weighted MSE with orthogonal projection supervision
- **Dynamic synthetic corruption** during training with support for static corrupted test and validation volumes
- End-to-end pipeline for training, k-fold evaluation, inference, and metric reporting

## Installation

```bash
git clone https://github.com/MedICL-VU/VAMOS-OCTA.git
cd VAMOS-OCTA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

Input volumes are expected as 3D `.tif` stacks of shape `(D, H, W)` and dtype `uint16`.

Each case should provide three files sharing a common stem:

```text
data/
├── volume1_corrupted.tif
├── volume1_gt.tif
├── volume1_mask.tif
├── volume2_corrupted.tif
├── volume2_gt.tif
├── volume2_mask.tif
└── ...
```

Mask files are interpreted slice-wise. The current code accepts either:

- a 1D binary mask of length `D`
- a binary 3D mask where each slice is constant and the slice label is read from `mask[:, 0, 0]`

## Usage

Train and evaluate with full k-fold cross-validation:

```bash
python main.py --data_dir data/train --epochs 100 --stride 1 --kfold
```

Run a single fold:

```bash
python main.py --data_dir data/train --fold_idx 1
```

Evaluate previously saved checkpoints without retraining:

```bash
python main.py --data_dir data/train --skip_train
```

For the full CLI, run:

```bash
python main.py --help
```

## Execution Flow

The public pipeline entry point is `main.py`, which performs the following steps:

1. Discover `{corrupted, gt, mask}` volume triplets from `--data_dir`
2. Build deterministic k-fold train/validation/test splits at the volume level
3. Train `models/unet2p5d.py` using online or static corruption settings
4. Load the best checkpoint for each run/fold
5. Inpaint held-out test volumes slice-by-slice
6. Save inpainted outputs to `output/inpainted/`
7. Report B-scan and projection metrics

Additional model files in `models/` are retained as related research variants, but the public training and evaluation entry point currently instantiates `UNet2p5D`.

## Evaluation Metrics

### B-scan metrics

- L1
- Mean Intensity Error
- PSNR
- SSIM (window size 11)
- Global NCC
- Gradient L1
- LPIPS
- Edge Preservation Ratio
- Laplacian Blur Score Difference

### Projection (MIP) metrics

- L1
- Mean Intensity Error
- PSNR
- SSIM (window size 11)
- Global NCC
- Gradient L1

## Reproducibility Notes

- Global seeding is applied to Python, NumPy, and PyTorch in `main.py`
- cuDNN is configured for deterministic execution (`deterministic=True`, `benchmark=False`)
- Training, validation, and metric computations operate on float32 volumes normalized to `[0, 1]`
- Saved inpainted volumes are written back to disk as `uint16`
- K-fold splitting is deterministic for a fixed seed and dataset contents

These notes describe the released code path here; the repository behavior is the source of truth for all experiments.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{disanto2026vamos,
  title={VAMOS-OCTA: Vessel-Aware Multi-Axis Orthogonal Supervision for Inpainting Motion-Corrupted OCT Angiography Volumes},
  author={DiSanto, Nick and Khodapanah Aghdam, Ehsan and Liu, Han and Watson, Jacob and Tao, Yuankai K. and Li, Hao and Oguz, Ipek},
  booktitle={SPIE Medical Imaging},
  year={2026}
}
```

## Contact

For questions or feedback, contact [Nick DiSanto](mailto:nicolas.c.disanto@vanderbilt.edu).

## License

This repository is released under the [MIT License](LICENSE).
