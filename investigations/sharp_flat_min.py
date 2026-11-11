''''

We are looking at studying the minimum that we get to at the end of the training. Is the solution sharp or flat minima.


Keskar et al. (2016).
 1- look at the hessian matrice of the Loss L, if the eigenvalues are big > sharp minima, if the eigenvalues are small we have a flat minima.


SAM (Sharpness-Aware Minimization)  min L (w + epsilon)
'''
# Libraries & Modules
import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from skimage import filters
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] skimage not found — SSIM disabled.")



# Internal Modules

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.downloaders.data_loaders import process_fastmri_h5
from baseline_models.GRAPPA_vectorized.vec_grappa import retrospective_grappa
from baseline_models.RAKI.main import raki_recon_algo
from src.KAE_vec_investigation.networks import (cKAE, KWN)
from src.KAE_vec_investigation.training import (Trainer, Trainer_KWN)
from src.KAE_vec_investigation.run_koopman import (train_run_model_kae, load_run_kae)
from helpers.preprocessing.preprocessing import kspace_to_image




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


def plot_loss_curve(losses, title="Training Loss", save_path=None, filename="loss"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(losses, lw=1.2, color='steelblue')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title(title); ax.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"), dpi=150)
    else:
        plt.show()
    plt.close(fig)


def dynamic_windowing(img, window_perc=99.9):
    return 0.0, float(np.percentile(np.abs(img), window_perc))


def get_zoom_region(img, x, y, xbox, ybox):
    return img[x - xbox: x + xbox, y - ybox: y + ybox]


def grid_new(images, names, window_perc=99.9, subfactor=1,
             x=50, y=50, box_size=(20, 20),
             filename="comparison", save_path=None):
    """
    3-row comparison grid: [full image | zoom | difference map].

    Shared windowing
    ----------------
    Image rows  → vmax from GT percentile (same for all columns).
    Diff row    → vmax from max error across ALL methods (shared colour scale).
                  Colour intensity = absolute error magnitude, directly comparable.
    Colormap 'hot': black=0 error, white=max error.
    """
    n = len(images)
    if n <= 1: return

    gt_img        = np.abs(images[0])
    methods_imgs  = [np.abs(im) for im in images[1:]]
    plot_images   = methods_imgs + [gt_img]
    plot_names    = names[1:] + [names[0]]
    ncols         = len(plot_images)

    vmin_img, vmax_img = dynamic_windowing(gt_img, window_perc)
    diff_images = [np.abs(gt_img - im) for im in methods_imgs]
    diff_max    = np.percentile(np.concatenate([d.ravel() for d in diff_images]),
                                window_perc)
    xbox, ybox  = (box_size, box_size) if isinstance(box_size, int) else box_size

    fig = plt.figure(figsize=(ncols * 3.5 + 0.6, 10))
    gs  = fig.add_gridspec(3, ncols + 1,
                           width_ratios=[3.5] * ncols + [0.25],
                           wspace=0.02, hspace=0.05)
    axs      = np.array([[fig.add_subplot(gs[r, c])
                          for c in range(ncols)] for r in range(3)])
    cax      = fig.add_subplot(gs[2, ncols])
    diff_ref = None

    for col in range(ncols):
        img   = plot_images[col]
        is_gt = (col == ncols - 1)

        axs[0, col].imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[0, col].set_title(plot_names[col], fontsize=9, fontweight='bold', pad=3)
        if is_gt:
            axs[0, col].add_patch(
                Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                           linewidth=1.5, edgecolor='r', facecolor='none'))
        axs[0, col].axis('off')

        axs[1, col].imshow(get_zoom_region(img, x, y, xbox, ybox),
                           cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[1, col].axis('off')

        if not is_gt:
            n_std, _ = get_nrmse_metrics_new(gt_img, img)
            diff_ref = axs[2, col].imshow(
                diff_images[col] * subfactor,
                cmap='gray', vmin=0, vmax=diff_max)
            # cmap='hot', vmin=0, vmax=diff_max, interpolation='nearest'
            axs[2, col].text(0.02, 0.97, f'NRMSE: {n_std:.4f}',
                             transform=axs[2, col].transAxes,
                             color='cyan', fontsize=8, fontweight='bold', va='top')
            axs[2, col].text(0.02, 0.03,
                             f'max|err|: {np.max(diff_images[col]):.2e}',
                             transform=axs[2, col].transAxes,
                             color='white', fontsize=7, va='bottom')
            if subfactor != 1:
                axs[2, col].text(0.98, 0.03, f'×{subfactor}',
                                 transform=axs[2, col].transAxes,
                                 color='yellow', fontsize=7, va='bottom', ha='right')
        else:
            axs[2, col].axis('off')

    if diff_ref is not None:
        cb = plt.colorbar(diff_ref, cax=cax)
        cb.set_label('|GT − recon|', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cax.text(0.5, -0.04, 'shared scale',
                 transform=cax.transAxes, fontsize=6, ha='center', color='gray')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"),
                    bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close(fig)



# ==================================================
# 10. BASELINE ORCHESTRA  (GRAPPA + baseline KAE)
# ==================================================

def train_run_baseline(kspace, params, run_dict,
                        save_path_raki_weights="D:/RAKI_weights/", save_path_koopman="D:/Baseline_KAE_draft/",
                        save_path=None, save_name=None):
    filename  = run_dict["filename"].replace(".h5", "")
    slice_idx = run_dict["slice"]

    print("  [GRAPPA]")
    grappa_recon = retrospective_grappa(
        kspace.copy(), params["R"], params["number_columns"],
        params["kernel"], ACS_bool=True)
    
    print("  [RAKI]")
    raki_recon = raki_recon_algo(kspace.copy(), params["R"], params["number_columns"], ACS_flag=True, device='cuda', save_path=save_path_raki_weights, slice=run_dict["slice"], filename=run_dict["filename"])


    kae_file = os.path.join(save_path_koopman,
                            f"{filename}_slice{slice_idx}.pth")
    if os.path.exists(kae_file):
        print(f"  [KAE baseline] Loading Koopman weights {kae_file} for slice {slice_idx}...")
        ckpt      = torch.load(kae_file, map_location=params['device'],
                                weights_only=True)
        kae_recon, log = load_run_kae(kspace, params, ckpt)
    else:
        print("  [KAE baseline] Training...")
        kae_recon, log = train_run_model_kae(kspace.copy(), params)
        os.makedirs(save_path_koopman, exist_ok=True)
        torch.save({'model_state_dict': log['model'].state_dict(),
                    'params': params}, kae_file)

    kspace_recons = [kspace, grappa_recon, raki_recon, kae_recon]
    titles        = ["Ground Truth", "GRAPPA", "RAKI", "Koopman Baseline"]
    img_recons    = [kspace_to_image(k.conj()) for k in kspace_recons]
    metrics       = dict_metrics(img_recons, titles, run_dict, save_path, save_name)
    return kspace_recons, img_recons, titles, metrics, log


def orchestra(data_dict, filename_list, base_params,
              save_path=None, save_path_raki_weights="D:/RAKI_weights/", save_path_koopman="D:/Baseline_KAE_draft/"):
    results_master = {}
    log_master     = {}
    if save_path: os.makedirs(save_path, exist_ok=True)

    for filename in filename_list:
        results_master[filename] = {}
        log_master[filename]     = {}
        print(f"\n=== {filename} ===")

        for slice_idx, tensor in data_dict[filename].items():
            print(f"  Slice {slice_idx}")
            run_dict = {"filename": filename, "slice": slice_idx,
                        "R": base_params["R"],
                        "number_columns": base_params["number_columns"]}

            ksp_r, img_r, titles, metrics, log = train_run_baseline(
                tensor, base_params, run_dict,
                save_path_raki_weights=save_path_raki_weights,
                save_path_koopman=save_path_koopman,
                save_path=save_path, save_name="baseline_metrics.csv")

            slice_data = {}
            for i, title in enumerate(titles):
                slice_data[title] = {
                    "kspace":  ksp_r[i],
                    "image":   img_r[i],
                    "metrics": metrics[i - 1] if i > 0 else None,
                }
            results_master[filename][slice_idx] = slice_data
            log_master[filename][slice_idx]     = log

    return results_master, log_master




# ==================================================
# 13. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    base_save_path = "D:/exp_RESULTS/CDT_exp/041226"
    exp_folder = 'lrw_1e-3_lrb_1e-3_kae_div_std'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR      = "D:/fastmri/multicoil_train/"
    filename_list = [
        "file_brain_AXFLAIR_200_6002512.h5",
        # "file_brain_AXT1POST_201_6002668.h5",
        # "file_brain_AXT2_200_2000485.h5",
    ]


    SAVE_PATH = os.path.join(base_save_path, exp_folder)

    # all_data needed for global normalisation across slices
    all_data  = process_fastmri_h5(DATA_DIR, filelist=filename_list, all_slices=True)
    data_dict = process_fastmri_h5(DATA_DIR, filelist=filename_list, all_slices=True)

    # ── Baseline: GRAPPA + standard KAE ──────────────────────────────
    base_params = {
        'optimizer':         'adam',
        'momentum_rate':     None,
        'epochs':            4000,
        'lr_weight':         1e-3,
        'lr_bias':           1e-3,
        'l2':                False,
        'l2_lambda':         None,
        'ACS_flag':          True,
        'kernel':            [5, 4],
        'R':                 4,          # phase-encoding acceleration (Ry)
        'number_columns':    40,
        'device':            DEVICE,
        'norm_type':         'single',
        'outerbound':        1.0,
        'reduc_d':           1.0,
        'bias':              False,
        'beta_init':         0.0,
        'separate_lrs':      False,
        'activation_name':   'modrelu',  # standard baseline
        'loss_type':         'mse',
        # batch_size_recon: only affects reconstruction inference (VRAM budget)
        # Training is always full-batch GD regardless of this value
        'batch_size_recon':  4096,
    }

    print("\n" + "="*60 + "\nBASELINE\n" + "="*60)
    master_baseline, baseline_logs = orchestra(
        data_dict, filename_list, base_params,
        save_path         = "./results/baseline/",
        save_path_raki_weights="D:/RAKI_weights/",
        save_path_koopman = "D:/Baseline_KAE_draft/",
    )




