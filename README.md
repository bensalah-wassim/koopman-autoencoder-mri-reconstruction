# Scan-specific k-space interpolation with deep Koopman autoencoders

This repository implements a scan-specific, interpretable k-space interpolation technique for the reconstruction of accelerated MRI data. The code loads fully sampled k-space data, retrospectively undersamples it, compares baseline reconstructions, trains a Koopman model, and saves quantitative results.

The main entry point is `main.py`. It orchestrates data loading, baseline reconstruction, Koopman training/inference, and result export.

## What the project does

- loads fastMRI-style multicoil `.h5` data from the `data/` folder
- applies retrospective undersampling and ACS-based training logic
- computes GRAPPA and RAKI baseline reconstructions
- trains or loads a scan-specific Koopman autoencoder from undersampled k-space data
- reconstructs the missing k-space lines and converts them back to image space
- saves metrics and visual comparisons under `results/`

## Evaluation protocol

- Quantitative evaluation includes NRMSE, SSIM, and PSNR.
- These metrics are computed on a foreground mask estimated from the absolute ground-truth image with the triangle thresholding method (`skimage.filters.threshold_triangle`) to avoid bias from noisy background.
- This masking is used because the Koopman reconstruction acts as a denoiser, and a global metric on the full image would otherwise be strongly influenced by noise in the reference.
- RAKI is trained for each multicoil k-space acquisition.
- The Koopman autoencoder is trained as a single scan-specific model to reconstruct the undersampled multi-coil k-space.

## Repository structure

```text
koopman-autoencoder-mri-reconstruction/
├── main.py                     # main execution script
├── README.md                   # project documentation
├── requirements.txt            # Python dependencies
├── data/                       # fastMRI-like HDF5 files
├── results/                    # visual outputs and comparison CSVs
├── weights/                    # trained model checkpoints
│   ├── koopman/
│   └── raki/
├── helpers/
│   ├── downloaders/
│   ├── evaluate/
│   ├── plots/
│   ├── preprocessing/
│   ├── savers/
│   └── utils/
├── baseline_models/
│   ├── GRAPPA/
│   ├── GRAPPA_vectorized/
│   └── RAKI/
├── src/
│   └── KAE/
│       ├── networks.py
│       ├── training.py
│       └── run_koopman.py
├── notebooks/
├── investigations/
├── hyperparameter_search/
├── optuna_results/
└── docs/
```

## Data requirements

The raw dataset is not bundled in this repository. To use the project, you must request access to the official fastMRI dataset and download a licensed `.h5` file from the [fastMRI portal](https://fastmri.med.nyu.edu/).

### fastMRI access and setup

1. Register and request authorization on the official [fastMRI portal](https://fastmri.med.nyu.edu/).
2. Download a representative fully sampled file, for example `file_brain_AXFLAIR_200_6002469.h5`.
3. Place it in the local `data/` directory at the root of the project:

```text
koopman-autoencoder-mri-reconstruction/
└── data/
    └── file_brain_AXFLAIR_200_6002469.h5
```

4. In `main.py`, select the appropriate contrast parameter. For example, the project sets:

```python
lambda_by_contrast = {
    "AXT1POST": 0.16,
    "AXT1PRE": 0.15,
    "AXT1": 0.15,
    "AXT2": 0.09,
    "AXFLAIR": 0.12,
}
```

and uses:

```python
lambda_by_contrast['AXFLAIR']
```

for the example AXLAIR dataset.

## Installation

Requirements:

- Python 3.12.11
- Git
- pip
- PyTorch 2.8.0 with CUDA 12.6 support, as specified in `requirements.txt` for complex-valued neural-network operations

Setup:

```bash
git clone <repository-url>
cd koopman-autoencoder-mri-reconstruction
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the project

From the project root:

```bash
python main.py
```

This script:

1. loads the MRI data,
2. runs the GRAPPA and RAKI baselines,
3. trains or loads the Koopman autoencoder,
4. saves reconstructions and metrics in `results/`,
5. stores trained weights in `weights/`.

## Outputs

- `results/`: plots, comparison figures, and CSV summaries
- `weights/koopman/`: Koopman checkpoints
- `weights/raki/`: RAKI weights



## Acknowledgments

[1] Griswold MA, Jakob PM, Heidemann RM, et al. Generalized autocalibrating partially parallel acquisitions (GRAPPA). Mag Res Med. 2002;47(6):1202–1210. doi: 10.1002/mrm.10171

[2] Akçakaya, Mehmet et al. “Scan-specific robust artificial-neural-networks for k-space interpolation (RAKI) reconstruction: Database-free deep learning for fast imaging.” Magnetic Resonance in Medicine, vol. 81, no. 1, 2019, pp. 439–453. doi:10.1002/mrm.27420.

[3] Knoll, Florian et al. “fastMRI: A Publicly Available Raw k-Space Dataset for Accelerated MRI Reconstruction Using Machine Learning.” Radiology: Artificial Intelligence, vol. 2, no. 1, 2020, e190007. doi:10.1148/ryai.2020190007.

[4] Zbontar, Jure et al. “fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.” arXiv preprint arXiv:1811.08839, 2018. https://arxiv.org/abs/1811.08839.

