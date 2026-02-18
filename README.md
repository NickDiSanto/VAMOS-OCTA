## VAMOS-OCTA: Vessel-Aware Multi-Axis Orthogonal Supervision for Inpainting Motion-Corrupted OCT Angiography Volumes

🏆 SPIE Medical Imaging 2026 — Image Processing Best Student Paper Award  
🎤 Selected for Oral Deep-Dive Presentation

Handheld OCT Angiography (OCTA) enables retinal imaging in uncooperative patients but suffers from severe motion artifacts, including fully corrupted B-scans. **VAMOS-OCTA** is a deep learning framework for inpainting motion-corrupted B-scans in 3D OCT Angiography (OCTA) volumes, enabling volumetric reconstruction in challenging handheld imaging scenarios.

<p align="center"> <img src="/figures/vamos_framework.png" alt="VAMOS Framework" width="800"/> </p>

Our method restores missing slices while preserving vessel structures across both cross-sectional and projection views.

<p align="center"> <img src="/figures/mip_results_vamos.png" alt="MIP Results" width="800"/> </p>

### Key Features:
- **2.5D U-Net** inpainting model conditioned on surrounding slices
- **VAMOS loss** combining vessel-weighted MSE with orthogonal projection supervision (axial + lateral)
- **Dynamic synthetic corruption** pipeline simulates realistic B-scan dropouts during training
- Evaluation across multiple metrics targeting both pixel-level accuracy and perceptual quality

### Installation
```bash
git clone https://github.com/MedICL-VU/VAMOS-OCTA.git
cd VAMOS-OCTA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage
#### Prepare data:
OCTA volumes must be 3D .tif stacks of shape (D, H, W) and type uint16. Place all volumes in a folder such as:
```kotlin
data
├── volume1_corrupted.tif
├── volume1_gt.tif
├── volume1_mask.tif
├── volume2_corrupted.tif
├── .
```

#### Train and test a model:
```bash
python main.py --data_dir data/train --epochs 100 --stride 1 --kfold
```
A complete list of flags is available in main.py.

#### Evaluate a previously-trained model:
```bash
python main.py --skip_train
```

### Evaluation Metrics
VAMOS-OCTA uses both pixel-wise accuracy and perceptual quality metrics:
#### B-scan Metrics
- Gradient L1
- LPIPS
- Laplacian Blur Diff
- Edge Preservation Ratio
- PSNR

#### Projection Metrics (MIPs)
- L1
- MIE (Mean Intensity Error)
- SSIM
- NCC
- PSNR

#### Citation
If you use this work, please cite:
```
@inproceedings{disanto2026vamos,
  title={VAMOS-OCTA: Vessel-Aware Multi-Axis Orthogonal Supervision for Inpainting Motion-Corrupted OCT Angiography Volumes},
  author={DiSanto, Nick and Khodapanah Aghdam, Ehsan and Liu, Han and Watson, Jacob and Tao, Yuankai K. and Li, Hao and Oguz, Ipek},
  booktitle={SPIE Medical Imaging},
  year={2026}
}
```

### Contact
For questions or feedback, please contact [**Nick DiSanto**](mailto:nicolas.c.disanto@vanderbilt.edu)

### License
This repository is licensed under the [MIT License](LICENSE).
