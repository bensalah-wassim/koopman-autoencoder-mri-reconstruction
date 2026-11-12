"""
KAE MRI Reconstruction Pipeline
================================

Sections
--------

1.  Vectorised data preparation  (generate_data_vec / reconstruct_vec)
7.  Metrics & visualisation utilities
8.  GRAPPA baseline
9.  KAE pipeline  (train_run_model_kae)
10. Baseline orchestra  (GRAPPA + baseline KAE)
11. Experiments pipeline
12. Evaluation & visualisation
13. Main execution

Training vs reconstruction batch note
--------------------------------------
Training uses FULL-BATCH gradient descent: every ACS patch is passed
through the network in a single forward/backward call each epoch.
This is intentional — the ACS region is small (typically ~24 × nx patches)
and the full-batch solution is both faster per epoch and more stable for
this problem size.

'batch_size_recon' (default 4096) applies ONLY to the reconstruction
(inference) phase, where we need to predict missing lines for the full
k-space matrix (nx × ny/R positions).  This can be millions of patches
and must be chunked to fit GPU VRAM.
"""

# ==================================================
# 1. IMPORTS & PATH CONFIGURATION
# ==================================================
import os, sys, math
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from skimage import filters
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] skimage not found — SSIM disabled.")


from .training import Trainer
from .networks import cKAE


# ==================================================
# 1. VECTORISED DATA PREPARATION  (GPU-READY)
# ==================================================

def _ry(R):
    """
    Extract scalar Ry from R.

    R can be:
      - int / float          → used directly  (KAE pipeline convention)
      - list / array [Rx,Ry] → Ry = R[1]      (GRAPPA pipeline convention)

    This is the single fix point for the TypeError:
        unsupported operand type(s) for -: 'int' and 'list'
    """
    if isinstance(R, (list, np.ndarray)):
        return int(R[1])
    return int(R)

def _src_y_offsets_kae(kernel, R):
    """
    Relative y-index offsets for the GRAPPA/KAE kernel.

    For kernel=[kx, ky], acceleration Ry:
        neighbours left  : [y-1 - Ry·s  for s in 1..num+ky%2]
        immediate pair   : [y-1,  y-1+Ry]
        neighbours right : [y-1+Ry + Ry·s  for s in 1..num]

    Offsets are expressed relative to y (target position), so they
    can be broadcast over any array of target positions.

    Returns: np.ndarray
    """
    Ry  = _ry(R)                    # ← always scalar, regardless of R type
    num = kernel[1] // 2 - 1
    return np.array(
        [-1 - Ry * s for s in range(1, num + kernel[1] % 2 + 1)]
        + [-1, -1 + Ry]
        + [-1 + Ry + Ry * s for s in range(1, num + 1)],
        dtype=int,
    )

def generate_data_vec(kspace: np.ndarray, kernel: list, R):
    """
    Vectorised full-batch training data generation from ACS.

    Replaces the nested Python loops in the original generate_data() with
    numpy advanced indexing — no Python loop overhead.

    Training regime
    ---------------
    The returned (X, Y) arrays are passed ENTIRELY to the Trainer in one
    shot: full-batch gradient descent, not mini-batch SGD.
    This is appropriate because:
      - The ACS is small (typically 24 × ~300 positions ≈ 7200 patches).
      - Full-batch has stable, deterministic gradient → faster convergence
        for this problem size.
      - No data loader / shuffling overhead.

    Layout
    ------
    X[n, c, :n_src]  = acquired neighbour patch   (source, known)
    X[n, c, n_src:]  = zeros                       (missing slots, unknown)
    Y[n, c, :n_src]  = same neighbour patch        (target, src consistency)
    Y[n, c, n_src:]  = actual missing line values  (target, reconstruction)

    Returns
    -------
    X, Y : (N, c, n_feat)  complex64,   N = len(kx) × len(ky)
    """
    Ry    = _ry(R)
    c, nx, ny = kspace.shape
    padx  = kernel[0] // 2
    num   = kernel[1] // 2 - 1
    l_pady = 1 + (num + kernel[1] % 2) * Ry
    r_pady = Ry - 1 + num * Ry
    n_src  = kernel[0] * kernel[1]
    n_miss = Ry - 1

    kx = np.arange(padx,   nx - padx,   dtype=int)
    ky = np.arange(l_pady, ny - r_pady, dtype=int)
    N  = len(kx) * len(ky)

    src_y_off  = _src_y_offsets_kae(kernel, Ry)           # (kernel[1],)
    x_off      = np.arange(-padx, padx + 1, dtype=int)  # (ker_x,)
    miss_y_off = np.arange(0, n_miss, dtype=int)        # (Ry-1,)

    # Absolute index arrays
    src_x_abs  = kx[:, None] + x_off[None, :]         # (n_kx, ker_x)
    src_y_abs  = ky[:, None] + src_y_off[None, :]     # (n_ky, kernel[1])
    miss_y_abs = ky[:, None] + miss_y_off[None, :]    # (n_ky, Ry-1)

    # Source patches — shape (c, n_kx, n_ky, ker_x, kernel[1])
    xi = src_x_abs[:, np.newaxis, :, np.newaxis]       # (n_kx, 1, ker_x, 1)
    yi = src_y_abs[np.newaxis,  :, np.newaxis, :]      # (1, n_ky, 1, kernel[1])
    src_patches = kspace[:, xi, yi]
    src_patches = src_patches.transpose(1, 2, 0, 3, 4).reshape(N, c, n_src)

    # Missing values — shape (c, n_kx, n_ky, Ry-1)
    xi_m    = kx[:, np.newaxis, np.newaxis]
    yi_m    = miss_y_abs[np.newaxis, :, :]
    miss_vals = kspace[:, xi_m, yi_m]
    miss_vals = miss_vals.transpose(1, 2, 0, 3).reshape(N, c, n_miss)

    X = np.zeros((N, c, n_src + n_miss), dtype=kspace.dtype)
    Y = np.zeros((N, c, n_src + n_miss), dtype=kspace.dtype)
    X[:, :, :n_src] = src_patches
    Y[:, :, :n_src] = src_patches
    Y[:, :, n_src:] = miss_vals
    return X, Y


# ---------------
#     Trakcing --
# ---------------

def compute_pos(kx, ky, x_off, src_y_off, miss_y_off, n_coils):
    """
    Calcule les coordonnées (x, y, coil) pour chaque point de Y.
    Retourne un tableau de forme (N, c, n_feat, 3)
    """
    N = len(kx) * len(ky)
    n_src = len(x_off) * len(src_y_off)
    n_miss = len(miss_y_off)
    
    # 1. Créer la grille des centres de patches (x, y)
    # kx_grid: (n_kx, n_ky), ky_grid: (n_kx, n_ky)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    kx_flat = kx_grid.reshape(N, 1)
    ky_flat = ky_grid.reshape(N, 1)

    # 2. Positions des points Sources
    # On combine les offsets x et y pour chaque patch
    src_x_coords = kx_flat[:, :, None] + x_off[None, :, None]      # (N, n_ker_x, 1)
    src_y_coords = ky_flat[:, :, None] + src_y_off[None, None, :]  # (N, 1, n_ker_y)
    
    # On aplatit pour obtenir (N, n_src, 2)
    src_pos = np.stack(np.broadcast_arrays(src_x_coords, src_y_coords), axis=-1)
    src_pos = src_pos.reshape(N, n_src, 2)

    # 3. Positions des points Missing
    miss_x_coords = kx_flat[:, :, None]                            # (N, 1, 1)
    miss_y_coords = ky_flat[:, :, None] + miss_y_off[None, None, :] # (N, 1, n_miss)
    
    miss_pos = np.stack(np.broadcast_arrays(miss_x_coords, miss_y_coords), axis=-1)
    miss_pos = miss_pos.reshape(N, n_miss, 2)

    # 4. Concaténation spatiale (N, n_feat, 2)
    spatial_pos = np.concatenate([src_pos, miss_pos], axis=1)

    # 5. Ajout de la dimension Coil (N, c, n_feat, 3)
    # On répète les positions spatiales pour chaque coil et on ajoute l'index du coil
    pos_Y = np.zeros((N, n_coils, n_src + n_miss, 3), dtype=int)
    
    # Remplissage X et Y
    pos_Y[:, :, :, :2] = spatial_pos[:, np.newaxis, :, :]
    
    # Remplissage de l'index du Coil (0, 1, 2...)
    coil_indices = np.arange(n_coils)[None, :, None] # (1, c, 1)
    pos_Y[:, :, :, 2] = coil_indices

    return pos_Y



# ---------------
#     Trakcing --
# ---------------

def gdv_debug(kspace: np.ndarray, kernel: list, R):
    """
    Vectorised full-batch training data generation from ACS.

    Replaces the nested Python loops in the original generate_data() with
    numpy advanced indexing — no Python loop overhead.

    Training regime
    ---------------
    The returned (X, Y) arrays are passed ENTIRELY to the Trainer in one
    shot: full-batch gradient descent, not mini-batch SGD.
    This is appropriate because:
      - The ACS is small (typically 24 × ~300 positions ≈ 7200 patches).
      - Full-batch has stable, deterministic gradient → faster convergence
        for this problem size.
      - No data loader / shuffling overhead.

    Layout
    ------
    X[n, c, :n_src]  = acquired neighbour patch   (source, known)
    X[n, c, n_src:]  = zeros                       (missing slots, unknown)
    Y[n, c, :n_src]  = same neighbour patch        (target, src consistency)
    Y[n, c, n_src:]  = actual missing line values  (target, reconstruction)

    Returns
    -------
    X, Y : (N, c, n_feat)  complex64,   N = len(kx) × len(ky)
    """
    Ry    = _ry(R)
    c, nx, ny = kspace.shape
    padx  = kernel[0] // 2
    num   = kernel[1] // 2 - 1
    l_pady = 1 + (num + kernel[1] % 2) * Ry
    r_pady = Ry - 1 + num * Ry
    n_src  = kernel[0] * kernel[1]
    n_miss = Ry - 1

    kx = np.arange(padx,   nx - padx,   dtype=int)
    ky = np.arange(l_pady, ny - r_pady, dtype=int)
    N  = len(kx) * len(ky)

    src_y_off  = _src_y_offsets_kae(kernel, Ry)           # (kernel[1],)
    x_off      = np.arange(-padx, padx + 1, dtype=int)  # (ker_x,)
    miss_y_off = np.arange(0, n_miss, dtype=int)        # (Ry-1,)

    # Absolute index arrays
    src_x_abs  = kx[:, None] + x_off[None, :]         # (n_kx, ker_x)
    src_y_abs  = ky[:, None] + src_y_off[None, :]     # (n_ky, kernel[1])
    miss_y_abs = ky[:, None] + miss_y_off[None, :]    # (n_ky, Ry-1)

    # Source patches — shape (c, n_kx, n_ky, ker_x, kernel[1])
    xi = src_x_abs[:, np.newaxis, :, np.newaxis]       # (n_kx, 1, ker_x, 1)
    yi = src_y_abs[np.newaxis,  :, np.newaxis, :]      # (1, n_ky, 1, kernel[1])
    src_patches = kspace[:, xi, yi]
    src_patches = src_patches.transpose(1, 2, 0, 3, 4).reshape(N, c, n_src)

    # Missing values — shape (c, n_kx, n_ky, Ry-1)
    xi_m    = kx[:, np.newaxis, np.newaxis]
    yi_m    = miss_y_abs[np.newaxis, :, :]
    miss_vals = kspace[:, xi_m, yi_m]
    miss_vals = miss_vals.transpose(1, 2, 0, 3).reshape(N, c, n_miss)

    X = np.zeros((N, c, n_src + n_miss), dtype=kspace.dtype)
    Y = np.zeros((N, c, n_src + n_miss), dtype=kspace.dtype)
    X[:, :, :n_src] = src_patches
    Y[:, :, :n_src] = src_patches
    Y[:, :, n_src:] = miss_vals
    
    pos_Y = compute_pos(kx, ky, x_off, src_y_off, miss_y_off, c)

    return X, Y, pos_Y



# ---------------
# ---------------
def reconstruct_vec(model: nn.Module, kspace: np.ndarray,
                    kernel: list, R,
                    device='cpu',
                    batch_size_recon: int = 4096) -> np.ndarray:
    """
    Vectorised + GPU-batched reconstruction.

    Why batch_size_recon is needed HERE (not in training)
    -------------------------------------------------------
    During reconstruction we must predict missing lines for the ENTIRE
    undersampled k-space: nx × (ny/Ry) positions, which can be
    ~300 × 150 ≈ 45 000 patches at once.  Passing all of them in one
    GPU call would require ~multi-GB VRAM for a 20-coil dataset.

    batch_size_recon controls how many (x, y) positions are processed
    per forward() call.  Training data (ACS only) is ~7 200 patches
    and fits comfortably in a single call — no batching needed there.

    Typical values:
      4096  — safe for 8 GB VRAM,  most fastMRI brain slices
      8192  — 16 GB VRAM
      16384 — 24 GB VRAM  (A100 / RTX 4090)

    Returns: reconst (c, nx, ny) complex64
    """
    Ry    = _ry(R)
    c, nx, ny = kspace.shape
    padx  = kernel[0] // 2
    num   = kernel[1] // 2 - 1
    l_pady = 1 + (num + kernel[1] % 2) * Ry
    r_pady = Ry - 1 + num * Ry
    n_src  = kernel[0] * kernel[1]
    n_miss = Ry - 1
    n_feat = n_src + n_miss

    pdata   = np.pad(kspace,
                     ((0, 0), (padx, padx), (l_pady, r_pady)),
                     mode='constant', constant_values=0)
    reconst = pdata.copy()
    _, nx_p, ny_p = pdata.shape

    kx     = np.arange(padx,   nx_p - padx,   dtype=int)
    ky     = np.arange(l_pady, ny_p - r_pady, dtype=int)
    ky_tgt = ky[1::Ry]                     # start of each missing-line block
    N      = len(kx) * len(ky_tgt)

    src_y_off = _src_y_offsets_kae(kernel, Ry)
    x_off     = np.arange(-padx, padx + 1, dtype=int)

    src_x_abs = kx[:, None]     + x_off[None, :]       # (n_kx, ker_x)
    src_y_abs = ky_tgt[:, None] + src_y_off[None, :]   # (n_ky_tgt, kernel[1])

    xi = src_x_abs[:, np.newaxis, :, np.newaxis]
    yi = src_y_abs[np.newaxis,   :, np.newaxis, :]
    src_patches = pdata[:, xi, yi]                      # (c, n_kx, n_ky_tgt, ker_x, kernel[1])
    src_patches = (src_patches.transpose(1, 2, 0, 3, 4)
                               .reshape(N, c, n_src)
                               .astype(np.complex64))

    inputs_flat = np.zeros((N, c, n_feat), dtype=np.complex64)
    inputs_flat[:, :, :n_src] = src_patches
    inputs_flat = inputs_flat.reshape(N, -1)

    # Batched GPU inference
    model.eval()
    model = model.to(device)
    preds = []
    with torch.no_grad():
        for b in range(0, N, batch_size_recon):
            batch = torch.tensor(inputs_flat[b: b + batch_size_recon],
                                 dtype=torch.complex64, device=device)

            preds.append(model(batch).cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(N, c, n_feat)

    # Scatter missing-line predictions back
    miss_p  = preds[:, :, n_src:].reshape(len(kx), len(ky_tgt), c, n_miss)
    mp      = miss_p.transpose(2, 0, 1, 3)            # (c, n_kx, n_ky_tgt, n_miss)

    miss_y_fill = (ky_tgt[:, None]
                   + np.arange(n_miss, dtype=int)[None, :])  # (n_ky_tgt, n_miss)
    xi_f = kx[:, np.newaxis, np.newaxis]
    yi_f = miss_y_fill[np.newaxis, :, :]
    reconst[:, xi_f, yi_f] = mp

    return reconst[:, padx:-padx, l_pady:-r_pady]


def undersample(kspace, r, num_acs):
    """Uniform Ry=r undersampling with central ACS region."""
    Ry  = _ry(r)
    res = np.zeros_like(kspace)
    res[:, :, ::Ry] = kspace[:, :, ::Ry]
    mid = kspace.shape[2] // 2
    hw  = num_acs // 2
    acs = kspace[:, :, mid - hw: mid + hw]
    res[:, :, mid - hw: mid + hw] = acs
    return res, acs


# ==================================================
# 7. METRICS & VISUALISATION UTILITIES
# ==================================================

def nrmse(gt, pred):
    gt_m, pr_m = np.abs(gt), np.abs(pred)
    return (np.sqrt(np.mean((gt_m - pr_m) ** 2))
            / (np.sqrt(np.mean(gt_m ** 2)) + 1e-12))


def round_sig(x, sig=4):
    if x == 0: return 0.0
    d = math.ceil(math.log10(abs(x) + 1e-300))
    return round(float(x), sig - d)


def print_metrics(gt, pred, print_bool=True):
    n = nrmse(gt, pred)
    result = {"nrmse": n, "ssim": float('nan')}
    if HAS_SKIMAGE:
        gt_m = np.abs(gt); pr_m = np.abs(pred)
        s    = ssim_func(gt_m, pr_m, data_range=gt_m.max() - gt_m.min() + 1e-12)
        result["ssim"] = s
    if print_bool:
        print(f"    NRMSE={result['nrmse']:.4f}  SSIM={result['ssim']:.4f}")
    return result


def get_nrmse_metrics_new(gt, pred, method='mean', p_val=50):
    """
    Standard NRMSE + masked NRMSE on anatomically significant region.

    method : 'mean'       — threshold = mean(|gt|)          (default)
             'percentile' — threshold = percentile(|gt|, p_val)
             'otsu'       — Otsu's method  (requires skimage)
             'li'         — Li's method    (requires skimage)
    """
    gt_m  = np.abs(gt)
    n_std = nrmse(gt, pred)

    if method == 'percentile':
        thresh = np.percentile(gt_m, p_val)
    elif method == 'otsu' and HAS_SKIMAGE:
        thresh = filters.threshold_otsu(gt_m)
    elif method == 'li' and HAS_SKIMAGE:
        thresh = filters.threshold_li(gt_m)
    else:
        thresh = np.mean(gt_m)

    mask     = gt_m > thresh
    n_masked = nrmse(gt[mask], pred[mask]) if np.any(mask) else n_std
    return n_std, n_masked


def dict_metrics(images, titles, run_dict, save_path=None, save_name=None,
                  print_bool=True):
    all_m  = []
    gt_img = images[0]
    for name, img in zip(titles[1:], images[1:]):
        m        = print_metrics(gt_img, img, print_bool=print_bool)
        ssim_val = round_sig(m["ssim"])
        nrm_val  = round_sig(m["nrmse"])
        all_m.append({"ssim": ssim_val, "nrmse": nrm_val})
        if save_path:
            row = {"filename": run_dict.get("filename", ""),
                   "slice": run_dict.get("slice", ""),
                   "model": name,
                   "acc_rate": run_dict.get("R", ""),
                   "ACS_lines": run_dict.get("number_columns", ""),
                   "ssim": ssim_val, "nrmse": nrm_val}
            csv_file = os.path.join(save_path, save_name or "results.csv")
            pd.DataFrame([row]).to_csv(
                csv_file, mode='a', index=False,
                header=not os.path.exists(csv_file))
    return all_m




# ==================================================
# 9. KAE PIPELINE
# ==================================================

def _unwrap(v):
    """[False] → False,  [1e-3] → 1e-3,  etc. — backward compat."""
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def train_run_model_kae(kspace, params, model_class=cKAE, trainer_class=Trainer,
                         model_extra_args=None, global_norm=None):
    """
    Full KAE training + reconstruction pipeline for one slice.
    """
    p = {k: _unwrap(v) for k, v in params.items()}

    scaled = kspace.copy().astype(np.complex64)
    nor    = 1.0
    c, nx_k, ny_k = scaled.shape


    if p.get('outerbound') is not None :
        if p.get('norm_type', 'single') == "single":
            nor = np.max(np.abs(scaled[:]))
            nor = params["outerbound"] / nor

        elif global_norm:
            nor = params["outerbound"] / global_norm
        scaled = scaled * nor

    Udata, acs = undersample(scaled, r=p['R'], num_acs=p['number_columns'])
    X, Y       = generate_data_vec(acs, p['kernel'], p['R'])

    n1 = (p['kernel'][0] * p['kernel'][1] + _ry(p['R']) - 1) * c
    n2 = int(n1 * p.get('reduc_d', 1.0))

    full_args = {
        'n1':              n1,
        'n2':              n2,
        'bias':            p.get('bias', False),
        'beta_init':       p.get('beta_init', 0.0),
        'activation_name': p.get('activation_name', 'modrelu'),
        'init_seed':       p.get('init_seed', 42),
        'device':          p.get('device', 'cpu'),
    }

    if model_extra_args:
        full_args.update(model_extra_args)

    KAE = model_class(**full_args)

    trainer = trainer_class(
        model          = KAE,
        momentum_rate  = p.get('momentum_rate'),
        optimizer_type = p['optimizer'],
        separate_lrs   = p.get('separate_lrs', False),
        epoch          = p['epochs'],
        lr_weight      = p.get('lr_weight', 1e-3),
        lr_bias        = p.get('lr_bias', 1e-3),
        device         = p.get('device', 'cpu'),
        l2_param_name  = p.get('l2_param_name',  'latent')
    )

    # 3. Training (full-batch GD — no batch_size parameter)
    t0  = time()
    log = trainer.train(
        inputs    = X,
        outputs   = Y,
        loss_type = p.get('loss_type', 'mse'),
        l2        = p.get('l2', False),
        l2_lambda = p.get('l2_lambda', 1e-4),
        lambda_t  = p.get('lambda_t', 1.0),
        lambda_s  = p.get('lambda_s', 1.0),
    )
    log['model'] = KAE
    print(f"  Training: {time()-t0:.1f}s")

    # 4. Reconstruction (batched GPU inference — batch_size_recon controls VRAM)
    t0      = time()
    reconst = reconstruct_vec(
        KAE, Udata,
        kernel           = p['kernel'],
        R                = p['R'],
        device           = p.get('device', 'cpu'),
        batch_size_recon = p.get('batch_size_recon', 4096),
    )
    print(f"  Reconstruction: {time()-t0:.1f}s")

    # Save checkpoint after training so it can be re-used later.
    if 'save_path' in p and p['save_path']:
        os.makedirs(p['save_path'], exist_ok=True)
        torch.save({'model_state_dict': KAE.state_dict(), 'params': p}, os.path.join(p['save_path'], 'kae_model.pth'))

    if p.get('ACS_flag', True):
        mid = ny_k // 2
        hw  = p['number_columns'] // 2
        reconst[:, :, mid - hw: mid + hw] = acs

    if p.get('outerbound') is not None:
        reconst = reconst/nor

    return reconst, log


def load_run_kae(kspace, params, checkpoint, model_class=cKAE, model_extra_args=None, global_norm=None):
    """Load pre-trained KAE weights and reconstruct (no training)."""
    p = {k: _unwrap(v) for k, v in params.items()}
    c = kspace.shape[0]

    scaled = kspace.copy().astype(np.complex64)
    nor    = 1.0

    if p.get('outerbound') is not None :
        if p.get('norm_type', 'single') == "single":
            nor = np.max(np.abs(scaled[:]))
            nor = params["outerbound"] / nor

        elif p.get('norm_type') == "div_std":
            std  = np.std(scaled)
            nor  = 1.0 / std 

        elif p.get('norm_type') == "standardization":
            mean = np.mean(scaled)
            std  = np.std(scaled)
            scaled = (scaled-mean)
            nor  = 1.0 / std 
        elif global_norm:
            print("We are doing a nornalisation on the volume.")
            nor = params["outerbound"] / global_norm
        scaled = scaled * nor

    Udata, acs = undersample(scaled, r=p['R'], num_acs=p['number_columns'])
    n1 = (p['kernel'][0] * p['kernel'][1] + _ry(p['R']) - 1) * c
    n2 = int(n1 * p.get('reduc_d', 1.0))

    full_args = {
        'n1':              n1,
        'n2':              n2,
        'bias':            p.get('bias', False),
        'beta_init':       p.get('beta_init', 0.0),
        'activation_name': p.get('activation_name', 'modrelu'),
        'init_seed':       p.get('init_seed', 42),
        'device':          p.get('device', 'cpu'),
    }

    if model_extra_args:
        full_args.update(model_extra_args)

    KAE = model_class(**full_args)

    KAE.load_state_dict(checkpoint['model_state_dict'])
    KAE.eval()

    reconst = reconstruct_vec(
        KAE, Udata,
        kernel           = p['kernel'],
        R                = p['R'],
        device           = p.get('device', 'cpu'),
        batch_size_recon = p.get('batch_size_recon', 4096),
    )

    if p.get('ACS_flag', True):
        mid = kspace.shape[2] // 2
        hw  = p['number_columns'] // 2
        reconst[:, :, mid - hw: mid + hw] = acs

    if p.get('outerbound') is not None:
        if p.get('norm_type') == "standardization":
            reconst = (reconst / nor) + mean
        else:
            reconst = reconst/nor



    return reconst, {'model': KAE}

