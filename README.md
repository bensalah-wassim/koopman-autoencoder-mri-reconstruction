# Scan-specific k-space interpolation using deep Koopman autoencoders

This repository implements a scan-specific, interpretable k-space interpolation technique for the reconstruction of accelerated MRI data using deep Koopman autoencoder. The project contains data pre- and post-processing, baseline reconstruction methods, model training, and quantitative evaluation in a single reproducible pipeline.

The primary execution entrypoint is `main.py`, which orchestrates data loading, baseline comparisons, model training, reconstruction, and result saving.

## Overview

The repository was designed for accelerated MRI reconstruction, the code and data example 
undersampled k-space data. The pipeline includes:

- preprocessing and postprocessing from k-space to image domain
- GRAPPA and RAKI baseline reconstructions
- Koopman autoencoder training and inference
- quantitative evaluation using NRMSE and SSIM-based metrics
- saving of trained weights and reconstructed outputs

The project is tailored for NYU fastMRI datasets and can be extended to additional acquisition settings and contrasts.

## Repository Structure

```text
koopman-autoencoder-mri-reconstruction/
├── main.py                             # main execution script and project orchestration
├── comparison.py                       # comparison and evaluation workflow
├── README.md                           # project documentation
├── data/                               # input MRI data files (.h5)
├── results/                            # experiment outputs and generated images
├── weights/                            # trained model weights
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
├── investigations/
│   ├── comparison.py
│   ├── results/
│   └── runs/
├── hyperparameter_search/
├── optuna_results/
└── notebooks/
    ├── eda/
    ├── model_experiments/
    └── results/
```

## Main Script

The script `main.py` is the central execution module. It performs the following operations:

1. Configures the project root and environment variables for portability.
2. Loads MRI data from the dataset directory.
3. Builds baseline reconstructions with GRAPPA and RAKI.
4. Trains or loads a Koopman autoencoder model on each slice.
5. Saves trained model weights to the appropriate folders.
6. Produces reconstructed image outputs and evaluation summaries.
7. Stores comparison statistics and visual outputs under `results/`.

The execution flow is organized around the following functions:

- `train_run_baseline(...)`: runs the baseline methods and saves reconstruction outputs.
- `orchestra(...)`: iterates over files and slices for the baseline pipeline.
- `run_koopman_experiments(...)`: launches the Koopman-based reconstruction workflow.
- `_orchestra_experiments(...)`: trains or loads the Koopman model per slice.
- `evaluate_and_save_experiments(...)`: summarizes quantitative results and saves figures and CSV outputs.

## Internal Modules

### `src/KAE/`

This directory contains the Koopman-based reconstruction model and its training logic.

- `networks.py`: defines the complex-valued Koopman autoencoder architecture and activation functions.
- `training.py`: training loop, optimizer configuration, and learning logic.
- `run_koopman.py`: data preparation, undersampling, reconstruction, and checkpoint handling.

### `baseline_models/`

This folder contains reference reconstruction methods used for comparison.

- `GRAPPA/` and `GRAPPA_vectorized/`: implementation of the GRAPPA reconstruction baseline.
- `RAKI/`: RAKI model training, inference, and weight export utilities.

### `helpers/`

This folder contains reusable utilities used across the project.

- `downloaders/data_loaders.py`: loads `.h5` fastMRI-style files and crops coil data.
- `preprocessing/preprocessing.py`: k-space and image-domain transformations.
- `plots/plots.py`: plotting utilities, comparison layouts, and reconstruction visualization.
- `utils/metrics.py`: NRMSE, SSIM, and summary metric computations.
- `evaluate/metrics.py`: additional evaluation utilities.

## Data Pipeline

The input dataset is expected to be stored in the `data/` directory and formatted as multicoil k-space arrays in HDF5 format. Each file contains multiple slices, and the code processes each slice independently.

The typical reconstruction pipeline is:

1. Load a multicoil k-space volume.
2. Apply calibration and undersampling logic.
3. Generate training samples from the ACS region.
4. Train the Koopman autoencoder or baseline model.
5. Reconstruct the missing k-space lines.
6. Convert reconstructed k-space to image-space for comparison.

## Weights and Outputs

### `weights/`

This directory stores model checkpoints and learned parameters.

- `weights/koopman/`: Koopman model checkpoints saved as `.pth` files.
- `weights/raki/`: per-coil RAKI model weights.

### `results/`

This directory stores generated outputs, including:

- reconstructed image plots
- loss curves
- comparison figures
- CSV summaries of quantitative metrics

## Installation and Environment Setup

This repository is designed to be cloned and installed outside the original local workspace. The following procedure is suitable for a fresh machine or a clean environment.

### 1. Prerequisites

- Python 3.12.11
- pip
- Git
- A working CUDA-enabled environment is strongly recommended for training, but CPU execution is also possible for testing and debugging.

This project was validated with Python 3.12.11 and its dependencies are managed through `requirements.txt`.

### 2. Clone the repository

```bash
git clone <repository-url>
cd koopman-autoencoder-mri-reconstruction
```

### 3. Create a virtual environment

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install project dependencies

Install all required packages directly from the repository requirements file:

```bash
pip install -r requirements.txt
```

This project is configured to use the dependency set defined in `requirements.txt`. If a GPU is available and you want to use CUDA-optimized PyTorch, install the CUDA-compatible build from the official PyTorch website instead of the default CPU-only package.

### 6. Optional: install the repository in editable mode

```bash
pip install -e .
```

If the repository does not provide a `setup.py` or `pyproject.toml`, this step can be skipped and the project can be run directly from the repository root.

### 7. Prepare the data directory

Create a `data/` folder and place the `.h5` MRI files inside it, or update the dataset path in the configuration variables in the execution scripts.

```text
koopman-autoencoder-mri-reconstruction/
├── data/
│   └── your_file.h5
└── ...
```

### 8. Run the project

From the project root:

```bash
python main.py
```

For a comparison-only workflow:

```bash
python comparison.py
```

## Dependencies

The project depends on the following Python libraries:

- Python 3.10+
- PyTorch
- NumPy
- pandas
- matplotlib
- h5py
- scikit-image

## Execution

To run the main pipeline from the project root:

```bash
python main.py
```

## Notes

- The repository is designed for reproducible experiments and supports saving results in a structured project-local directory.
- The training pipeline explicitly saves weights to avoid repeated retraining when a matching model checkpoint already exists.
- For scientific reproducibility, it is recommended to preserve the generated configuration, model checkpoints, and metric summaries together with the corresponding dataset version.

## Publication-Ready Summary

This work presents a multicoil MRI reconstruction approach based on a Koopman-inspired autoencoder trained on undersampled k-space data. The framework benchmarks the learned model against classical reconstruction baselines, including GRAPPA and RAKI, and evaluates performance using quantitative image quality metrics. The implementation is organized to enable controlled experimentation, model reuse, and reproducible result generation across slices and acquisition settings.
