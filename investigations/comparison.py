
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
from src.KAE_vec_investigation.networks import (cKAE, KR, KWN, cKAENorm, kaekr, cKAE_1nl, cKAE_1nl_1l, debug_activation)
from src.KAE_vec_investigation.training import (Trainer, Trainer_KWN, Trainer_KR, Trainer_batch, Trainer_val, Trainer_koopman)
from src.KAE_vec_investigation.run_koopman import (train_run_model_kae, load_run_kae, undersample, generate_data_vec, generate_test_data_vec)
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
    return n_std, n_masked, mask

def ssim_mssim(gt_img, img, mask):

    s = ssim_func(gt_img, img, data_range=gt_img.max() - gt_img.min())
    mgt_img = gt_img[mask]
    mimg    = img[mask]
    ms = ssim_func(mgt_img, mimg, data_range=mgt_img.max() - mgt_img.min())
    return s, ms

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
            n_std, mn_std, mask = nrmse_mask(gt_img, img)
            diff_ref = axs[2, col].imshow(
                diff_images[col] * subfactor,
                cmap='gray', vmin=0, vmax=diff_max)
            # cmap='hot', vmin=0, vmax=diff_max, interpolation='nearest'
            axs[2, col].text(0.02, 0.97, f'NRMSE: {n_std:.4f} \nMNRMSE: {mn_std:.4f}',
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

def nrmse_mask(gt, pred, method='triangle', p_val=50):
    """
    Retourne maintenant le masque utilisé pour le calcul.
    """
    gt_m  = np.abs(gt)
    n_std = nrmse(gt, pred)

    if method == 'percentile':
        thresh = np.percentile(gt_m, p_val)
    elif method == 'otsu':
        thresh = filters.threshold_otsu(gt_m)
    elif method == 'li':
        thresh = filters.threshold_li(gt_m)
    elif method == 'triangle':
        thresh = filters.threshold_triangle(gt_m)
    elif method == 'median':
        thresh = np.median(gt_m)
    else:
        thresh = np.mean(gt_m)

    mask = gt_m > thresh
    n_masked = nrmse(gt[mask], pred[mask]) if np.any(mask) else n_std
    
    return n_std, n_masked, mask

def mplot_maskonly(images, names, window_perc=99.9, subfactor=1,
             x=50, y=50, box_size=(20, 20),
             filename="comparison", save_path=None):
    
    n = len(images)
    if n <= 1:
        return

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
    
    axs = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(ncols)] for r in range(3)])
    cax = fig.add_subplot(gs[2, ncols])
    
    current_mask = None # Pour stocker le masque généré par la première méthode

    for col in range(ncols):
        img   = plot_images[col]
        is_gt = (col == ncols - 1)

        # ROW 0: Full Image
        axs[0, col].imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[0, col].set_title(plot_names[col], fontsize=9, fontweight='bold', pad=3)
        if is_gt:
            axs[0, col].add_patch(
                Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                           linewidth=1.5, edgecolor='r', facecolor='none'))
        axs[0, col].axis('off')

        # ROW 1: Zoom
        axs[1, col].imshow(get_zoom_region(img, x, y, xbox, ybox),
                           cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[1, col].axis('off')

        # ROW 2: Diff Maps OR Mask (for GT)
        if not is_gt:
            # On récupère n_std, mn_std ET le mask
            n_std, mn_std, mask = nrmse_mask(gt_img, img)
            current_mask = mask # On sauvegarde le masque pour l'afficher plus tard
            
            diff_ref = axs[2, col].imshow(
                diff_images[col] * subfactor,
                cmap='gray', vmin=0, vmax=diff_max)
            
            axs[2, col].text(0.02, 0.97, f'NRMSE: {n_std:.4f} \nMNRMSE: {mn_std:.4f}',
                             transform=axs[2, col].transAxes,
                             color='white', fontsize=8, fontweight='bold', va='top')
            axs[2, col].text(0.02, 0.03,
                             f'max|err|: {np.max(diff_images[col]):.2e}',
                             transform=axs[2, col].transAxes,
                             color='white', fontsize=7, va='bottom')
            
            if subfactor != 1:
                axs[2, col].text(0.98, 0.03, f'×{subfactor}',
                                 transform=axs[2, col].transAxes,
                                 color='yellow', fontsize=7, va='bottom', ha='right')
        else:
            if current_mask is not None:
                axs[2, col].imshow(current_mask, cmap='bone') # 'bone' ou 'gray' pour le masque
                axs[2, col].text(0.5, 0.95, 'MNRMSE MASK', 
                                 transform=axs[2, col].transAxes,
                                 color='yellow', fontsize=8, fontweight='bold', 
                                 ha='center', va='top')
        axs[2, col].axis('off')

    # Colorbar pour les cartes de différence
    if 'diff_ref' in locals():
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


def mplot(images, names, window_perc=99.9, subfactor=1,
          x=50, y=50, box_size=(20, 20),
          show_mask=False, 
          filename="comparison", save_path=None):
    
    n = len(images)
    if n <= 1:
        return

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
    
    axs = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(ncols)] for r in range(3)])
    cax = fig.add_subplot(gs[2, ncols])
    
    current_mask = None 

    for col in range(ncols):
        img   = plot_images[col]
        is_gt = (col == ncols - 1)

        # ROW 0: Full Image
        axs[0, col].imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[0, col].set_title(plot_names[col], fontsize=9, fontweight='bold', pad=3)
        if is_gt:
            axs[0, col].add_patch(
                Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                           linewidth=1.5, edgecolor='r', facecolor='none'))
        axs[0, col].axis('off')

        # ROW 1: Zoom
        axs[1, col].imshow(get_zoom_region(img, x, y, xbox, ybox),
                           cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[1, col].axis('off')

        # ROW 2: Diff Maps OR (Mask / Black Image for GT)
        if not is_gt:
            n_std, mn_std, mask = nrmse_mask(gt_img, img)
            current_mask = mask 
            
            diff_ref = axs[2, col].imshow(
                diff_images[col] * subfactor,
                cmap='gray', vmin=0, vmax=diff_max)
            
            axs[2, col].text(0.02, 0.97, f'NRMSE: {n_std:.4f} \nMNRMSE: {mn_std:.4f}',
                             transform=axs[2, col].transAxes,
                             color='white', fontsize=8, fontweight='bold', va='top')
            axs[2, col].text(0.02, 0.03,
                             f'max|err|: {np.max(diff_images[col]):.2e}',
                             transform=axs[2, col].transAxes,
                             color='white', fontsize=7, va='bottom')
            
            if subfactor != 1:
                axs[2, col].text(0.98, 0.03, f'×{subfactor}',
                                 transform=axs[2, col].transAxes,
                                 color='yellow', fontsize=7, va='bottom', ha='right')
        else:
            # ◄ Condition pour choisir entre Masque ou Noir Complet (GT - GT = 0)
            if show_mask:
                axs[2, col].imshow(current_mask, cmap='bone')
                axs[2, col].text(0.5, 0.95, 'MNRMSE MASK', 
                                 transform=axs[2, col].transAxes,
                                 color='yellow', fontsize=8, fontweight='bold', 
                                 ha='center', va='top')
            else:
                # Image entièrement noire alignée sur l'échelle de différence
                axs[2, col].imshow(np.zeros_like(gt_img), cmap='gray', vmin=0, vmax=diff_max)

        axs[2, col].axis('off')

    # Colorbar
    if 'diff_ref' in locals():
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

def mplot_close0(images, names, window_perc=99.9, subfactor=1,
          x=50, y=50, box_size=(20, 20),
          show_mask=False, 
          wspace=0.0, hspace=0.0,
          filename="comparison", save_path=None):

    n = len(images)
    if n <= 1:
        return

    gt_img        = np.abs(images[0])
    methods_imgs  = [np.abs(im) for im in images[1:]]
    plot_images   = methods_imgs + [gt_img]
    plot_names    = names[1:] + [names[0]]
    ncols         = len(plot_images)
    nrows         = 3

    vmin_img, vmax_img = dynamic_windowing(gt_img, window_perc)
    diff_images = [np.abs(gt_img - im) for im in methods_imgs]
    diff_max    = np.percentile(np.concatenate([d.ravel() for d in diff_images]),
                                window_perc)
    xbox, ybox  = (box_size, box_size) if isinstance(box_size, int) else box_size

    cell_size = 3.0
    fig = plt.figure(figsize=(ncols * cell_size, nrows * cell_size))

    # --- Désactive tout moteur de layout automatique (constrained/compressed) ---
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        pass  # anciennes versions de matplotlib : pas de layout engine par défaut

    # --- Positionnement manuel des axes (aucune interférence possible) ---
    col_w = (1 - wspace * (ncols - 1)) / ncols
    row_h = (1 - hspace * (nrows - 1)) / nrows

    axs = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            left   = c * (col_w + wspace)
            bottom = 1 - (r + 1) * row_h - r * hspace
            axs[r, c] = fig.add_axes([left, bottom, col_w, row_h])
    # ---------------------------------------------------------------------

    current_mask = None

    for col in range(ncols):
        img   = plot_images[col]
        is_gt = (col == ncols - 1)

        # ROW 0: Full Image
        axs[0, col].imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest', aspect='auto')
        axs[0, col].set_title(plot_names[col], fontsize=9, fontweight='bold', pad=3)
        if is_gt:
            axs[0, col].add_patch(
                Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                           linewidth=1.5, edgecolor='r', facecolor='none'))
        axs[0, col].axis('off')

        # ROW 1: Zoom
        axs[1, col].imshow(get_zoom_region(img, x, y, xbox, ybox),
                           cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest', aspect='auto')
        axs[1, col].axis('off')

        # ROW 2: Diff Maps OR Mask (for GT)
        if not is_gt:
            n_std, mn_std, mask = nrmse_mask(gt_img, img)
            current_mask = mask

            axs[2, col].imshow(diff_images[col] * subfactor,
                               cmap='gray', vmin=0, vmax=diff_max, aspect='auto')


            if subfactor != 1:
                axs[2, col].text(0.98, 0.03, f'×{subfactor}',
                                 transform=axs[2, col].transAxes,
                                 color='yellow', fontsize=7, va='bottom', ha='right')
        else:

            if show_mask and current_mask is not None:
                axs[2, col].imshow(current_mask, cmap='bone')
            else:
                # Image entièrement noire alignée sur l'échelle de différence
                axs[2, col].imshow(np.zeros_like(gt_img), cmap='gray', vmin=0, vmax=diff_max)
        axs[2, col].axis('off')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"),
                    bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.close(fig)


def mplot_close(images, names, window_perc=99.9, subfactor=1,
          x=50, y=50, box_size=(20, 20),
          show_mask=False, 
          wspace=0.0, hspace=0.0,
          filename="comparison", save_path=None,
          layout="vertical"): # <-- Nouveau paramètre ici

    n = len(images)
    if n <= 1:
        return

    gt_img        = np.abs(images[0])
    methods_imgs  = [np.abs(im) for im in images[1:]]

    if layout == "horizontal":
        plot_images   = methods_imgs + [gt_img]
        plot_names    = names[1:] + [names[0]]
    else: # vertical
        plot_images   = [gt_img] + methods_imgs
        plot_names    = [names[0]] + names[1:] 


    n_images      = len(plot_images)
    
    # --- Bascule intelligente de la grille ---
    if layout == "horizontal":
        nrows, ncols = 3, n_images
    elif layout == "vertical":
        nrows, ncols = n_images, 3
    else:
        raise ValueError("Le paramètre 'layout' doit être 'horizontal' ou 'vertical'.")

    vmin_img, vmax_img = dynamic_windowing(gt_img, window_perc)
    diff_images = [np.abs(gt_img - im) for im in plot_images]
    diff_max    = np.percentile(np.concatenate([d.ravel() for d in diff_images]),
                                window_perc)
    xbox, ybox  = (box_size, box_size) if isinstance(box_size, int) else box_size

    cell_size = 3.0
    fig = plt.figure(figsize=(ncols * cell_size, nrows * cell_size))

    # --- Désactive tout moteur de layout automatique ---
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        pass  

    # --- Positionnement manuel des axes ---
    col_w = (1 - wspace * (ncols - 1)) / ncols
    row_h = (1 - hspace * (nrows - 1)) / nrows

    axs = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            left   = c * (col_w + wspace)
            bottom = 1 - (r + 1) * row_h - r * hspace
            axs[r, c] = fig.add_axes([left, bottom, col_w, row_h])

    current_mask = None

    for i in range(n_images):
        img   = plot_images[i]

        # dynamic axe definition
        if layout == "horizontal":
            is_gt = (i == n_images - 1)
            ax_img, ax_zoom, ax_diff = axs[0, i], axs[1, i], axs[2, i]
        else: # vertical
            is_gt = (i == 0)
            ax_img, ax_zoom, ax_diff = axs[i, 0], axs[i, 1], axs[i, 2]

        # Image
        ax_img.imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img, aspect='auto')
        if layout == "horizontal":
            ax_img.set_title(plot_names[i], fontsize=9, fontweight='bold', pad=3)
            ax_img.axis('off')
        else: # verticale
            ax_img.set_ylabel(plot_names[i], fontsize=15, fontweight='bold', rotation=0, ha='right', va='center')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_visible(False)
        if is_gt:
            ax_img.add_patch(
                Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                           linewidth=1.5, edgecolor='r', facecolor='none'))
        

        # 2. Zoom
        ax_zoom.imshow(get_zoom_region(img, x, y, xbox, ybox),
                       cmap='gray', vmin=vmin_img, vmax=vmax_img,
                       aspect='auto')
        ax_zoom.axis('off')

        # 3. Différence absolue / Masque
        if not is_gt:
            n_std, mn_std, mask = nrmse_mask(gt_img, img)
            current_mask = mask

            ax_diff.imshow(diff_images[i] * subfactor,
                           cmap='gray', vmin=0, vmax=diff_max, aspect='auto')

            if subfactor != 1:
                ax_diff.text(0.98, 0.03, f'×{subfactor}',
                             transform=ax_diff.transAxes,
                             color='yellow', fontsize=7, va='bottom', ha='right')
        else:
            if show_mask and current_mask is not None:
                ax_diff.imshow(current_mask, cmap='bone')
            else:
                ax_diff.imshow(diff_images[i] * subfactor, cmap='gray', vmin=0, vmax=diff_max, aspect='auto')
        ax_diff.axis('off')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"),
                    bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_save_individual_image(data, save_bool=None, filepath=None, save_name=None):
    """
    Function to display the reconstructed image for a single dataset.

    Parameters:
    - data: A single data array (X, Y, Z, etc.)
    - label: Text label corresponding to the data set
    - coil: Coil axis to move (default is 0)

    Notes:
    K-space data needs to have the coils dimension as the first dimension,
    followed by read-out and phase-encoding dimensions.
    """

    # Image reconstruction
    reconstructed_image = kspace_to_image(data)

    # Create a figure for the reconstructed image
    fig = plt.figure(figsize=(10, 10), frameon=False, dpi=96)
    vmin, vmax = np.quantile(reconstructed_image, (0.01, 0.999))  # cut bottom 1% and top 1% of pixels
    plt.imshow(reconstructed_image, vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis("off")  # Turn off axis
    plt.margins(0, 0)
    if save_bool:
        if filepath: 
            os.makedirs(filepath, exist_ok=True)

            plt.savefig(os.path.join(filepath,save_name))
    else:
        plt.show()
    plt.close(fig)


def plot_windowed_recon(data, gt_data=None, save_bool=False, filepath=None, save_name=None, window_perc=(0.01, 0.99)):
    """
    Function to display or save the reconstructed magnitude image for multicoil k-space data.

    Parameters:
    - data: ndarray (channel, RE, PE). Single data array.
    - gt_data: ndarray (channel, RE, PE), optional. Ground truth data used for setting the same windowing.
    - save_bool: bool. If True, saves the image. Defaults to False.
    - filepath: str, optional. Directory to save the image.
    - save_name: str, optional. Filename for the saved image.
    - window_perc: tuple of floats. Quantiles (0 to 1) to cut bottom and top pixels. Defaults to (0.01, 0.999).

    Notes:
    K-space data needs to have the coils dimension as the first dimension,
    followed by read-out and phase-encoding dimensions.
    """

    # Image reconstruction and magnitude calculation
    reconstructed_image = kspace_to_image(data)

    # Determine Windowing
    if gt_data is not None:
        # Use ground truth for windowing if provided
        gt_img = kspace_to_image(gt_data)
        vmin, vmax = np.quantile(gt_img, window_perc)
    else:
        # No ground truth provided, compute windowing on the reconstructed image
        vmin, vmax = np.quantile(reconstructed_image, window_perc) 

    # Create a figure for the reconstructed image
    fig = plt.figure(figsize=(10, 10), frameon=False, dpi=96)
    
    # Plot image
    plt.imshow(reconstructed_image, vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis("off")  # Turn off axis
    plt.margins(0, 0)

    # Handle Saving vs Showing
    if save_bool and filepath and save_name:
        os.makedirs(filepath, exist_ok=True)
        # Using bbox_inches='tight' removes unwanted white borders around the image
        plt.savefig(os.path.join(filepath, save_name), bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        
    plt.close(fig)

def plot_kspace_images(kspace, plot_type="image", title="K-space Visualization", 
                       cols=5, par=None, save_dir=None, window=None):
    """
    Parameters:
    - window: tuple, float, or None
        If tuple (vmin, vmax), uses those absolute values.
        If float (e.g., 99.5), uses that percentile to calculate vmax (vmin=0).
        If None, uses the image's min/max.
    """
    save = save_dir is not None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        
    num_coils = kspace.shape[0]
    rows = math.ceil(num_coils / cols)

    # Ensure we handle cases where num_coils=1 (subplots returns a single ax, not array)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.75), 
                            squeeze=False, constrained_layout=True)
    axs = axs.flatten()

    for coil in range(num_coils):
        data = kspace[coil]

        if plot_type == "image":
            # 2D IFFT on the last two axes
            img = np.abs(np.fft.ifftshift(np.fft.ifftn(data, axes=(-2, -1)), axes=(-2, -1)))
            plot_data = np.squeeze(img)
            label = "Image"
        elif plot_type == "kspace":
            # Log magnitude for visibility
            plot_data = np.squeeze(np.log10(np.abs(data) + 1e-15))
            label = "K-space"
        else:
            raise ValueError("plot_type must be either 'image' or 'kspace'")

        # --- Windowing Logic ---
        vmin, vmax = None, None
        if window is not None:
            if isinstance(window, (list, tuple)):
                vmin, vmax = window
            elif isinstance(window, (int, float)):
                vmin = 0
                vmax = np.percentile(plot_data, window)
        
        im = axs[coil].imshow(plot_data, cmap="gray", vmin=vmin, vmax=vmax)
        axs[coil].set_title(f"Coil {coil} {label}")
        
        # Optional: Add a colorbar to one of them to see the scale
        # if coil == 0: fig.colorbar(im, ax=axs[coil], fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for i in range(num_coils, len(axs)):
        axs[i].axis("off")

    fig.suptitle(title, fontsize=16)
    
    if save:
        filename = f"{par}_{plot_type}.png" if par else f"{plot_type}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)
    else:
        plt.show()

# ==================================================
# 10. BASELINE ORCHESTRA  (GRAPPA + baseline KAE)
# ==================================================


def _unwrap(v):
    """[False] → False,  [1e-3] → 1e-3,  etc. — backward compat."""
    return v[0] if isinstance(v, list) and len(v) == 1 else v

def train_run_baseline(kspace, params, run_dict,
                        save_path_raki_weights="D:/RAKI_weights/", save_path_koopman="D:/Baseline_KAE_draft/",
                        save_path=None, save_name=None):
    filename  = run_dict["filename"].replace(".h5", "")
    slice_idx = run_dict["slice"]

    acs = np.zeros_like(kspace)
    mid = kspace.shape[2] // 2
    hw  = params["number_columns"] // 2
    acs[:, :, mid - hw: mid + hw] =  kspace[:, :, mid - hw: mid + hw]

    print("  [GRAPPA]")
    grappa_recon = retrospective_grappa(
        kspace.copy(), params["R"], params["number_columns"],
        params["kernel"], ACS_bool=True)
    img_path = os.path.join(save_path, "imgs/GRAPPA/" )

    plot_windowed_recon(grappa_recon.conj(), gt_data=kspace.conj(), save_bool=True, filepath=img_path, save_name=f"GRAPPA_{filename}_sl_{slice_idx}")
    
    print("  [RAKI]")
    raki_recon = raki_recon_algo(kspace.copy(), params["R"], params["number_columns"], ACS_flag=True, device='cuda', save_path=save_path_raki_weights, slice=run_dict["slice"], filename=run_dict["filename"])


    img_path = os.path.join(save_path, "imgs/RAKI/" )
    plot_windowed_recon(raki_recon.conj(), gt_data=kspace.conj(), save_bool=True, filepath=img_path, save_name=f"RAKI_{filename}_sl_{slice_idx}")

    # dont need now - writing the paper
    # kae_file = os.path.join(save_path_koopman,
    #                         f"{filename}_slice{slice_idx}.pth")
    # if os.path.exists(kae_file):
    #     print(f"  [KAE baseline] Loading Koopman weights {kae_file} for slice {slice_idx}...")
    #     ckpt      = torch.load(kae_file, map_location=params['device'],
    #                             weights_only=True)
    #     kae_recon, log = load_run_kae(kspace, params, ckpt)
    # else:
    #     print("  [KAE baseline] Training...")
    #     kae_recon, log = train_run_model_kae(kspace.copy(), params)
    #     os.makedirs(save_path_koopman, exist_ok=True)
    #     torch.save({'model_state_dict': log['model'].state_dict(),
    #                 'params': params}, kae_file)

    # img_path = os.path.join(save_path, "imgs/KAE_baseline/" )
    # plot_windowed_recon(kae_recon.conj(), gt_data=kspace.conj(), save_bool=True, filepath=img_path, save_name=f"KAE_baseline_{filename}_sl_{slice_idx}")

    img_path = os.path.join(save_path, "imgs/GT/" )
    plot_windowed_recon(kspace.conj(), gt_data=kspace.conj(), save_bool=True, filepath=img_path, save_name=f"GT_{filename}_sl_{slice_idx}")


    # kspace_recons = [kspace, acs, grappa_recon, raki_recon,  kae_recon]
    # titles        = ["Ground Truth", "ACS", "GRAPPA", "RAKI", "Koopman Baseline"]
    kspace_recons = [kspace, acs, grappa_recon, raki_recon]
    titles        = ["Ground Truth", "ACS", "GRAPPA", "RAKI"]
    img_recons    = [kspace_to_image(k.conj()) for k in kspace_recons]
    metrics       = dict_metrics(
                        img_recons, 
                        titles,
                        run_dict, save_path, save_name)
    return kspace_recons, img_recons, titles, metrics, {}# metrics, log

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
                    "metrics": 0 ,#metrics[i - 1] if i > 0 else None,
                }
            results_master[filename][slice_idx] = slice_data
            log_master[filename][slice_idx]     = log

    return results_master, log_master

# ==================================================
# 11. EXPERIMENTS PIPELINE
# ==================================================

def run_koopman_experiments(all_data_dict, data_dict, filenames,
                             experiments_dict, save_path):
    """
    Run all experiments.

    experiments_dict format
    -----------------------
    { 'exp_name': (ModelClass, {param_dict}), ... }

    Each param_dict may override any key from COMMON_CONFIG.
    """
    all_results = {}
    all_logs    = {}
    for exp_name, (model_class, trainer_class, full_params) in experiments_dict.items():
        print(f"\n>>> Experiment: {exp_name}")
        res, logs = _orchestra_experiments(
            all_data_dict, data_dict, filenames,
            full_params, model_class, trainer_class, save_path, exp_name)
        all_results[exp_name] = res
        all_logs[exp_name]    = logs
    return all_results, all_logs


def _orchestra_experiments(all_data_dict, data_dict, filename_list,
                            params, model_class, trainer_class, save_path, exp_name="", weights_root="E:/Koopman_paper_results/parametric_study_final/KAE_weights/"):
    results = {}
    logs    = {}
    p       = {k: _unwrap(v) for k, v in params.items()}

    for fname in filename_list:
        results[fname] = {}
        logs[fname]    = {}

        global_max = None
        if p.get('norm_type') == 'global':
            global_max = max(np.max(np.abs(t))
                             for t in all_data_dict[fname].values())

        for slice_idx, tensor in data_dict[fname].items():
            print(f"  [{fname}] slice {slice_idx}")

            debug_training = p.get('debug_training', False)
            if debug_training:
                acs_recon, recon, log = train_run_model_kae(
                    tensor, p, model_class=model_class, trainer_class=trainer_class, global_norm=global_max, debug_training=debug_training)
                
                res = np.zeros_like(tensor)
                mid = tensor.shape[2] // 2
                hw  = params["number_columns"] // 2
                res[:, :, mid - hw: mid + hw]  = acs_recon
                
                results[fname][slice_idx] = {
                    "acs": res,
                    "kspace": recon,
                    "acs_image":  kspace_to_image(res.conj()),
                    "image":  kspace_to_image(recon.conj()),
                }
                debug_activation( p.get('activation_name', 'none'), log['model'], save_path=save_path + 'activation' + f"/{exp_name}_{fname[:-3]}_sl_{slice_idx}/")
                plot_loss_curve(log["losses"], title=f"Loss — {fname} sl{slice_idx}", save_path=save_path+"/losses/", filename=f"{exp_name}_{fname[:-3]}_sl_{slice_idx}")

            else:
                # recon, log = train_run_model_kae(
                #     tensor, p, model_class=model_class, trainer_class=trainer_class, global_norm=global_max, debug_training=debug_training)
                print("dim", p.get("reduc_d", "NA"))
                lmbd_str = str(p.get("lambda_s", "NA")).replace(".", "_")
                acs_str = str(p.get("number_columns", "NA"))
                dim_str = str(p.get("reduc_d", "NA")).replace(".", "_")
                kernel_val = p.get("kernel", "NA")
                kernel_str = "x".join(map(str, kernel_val)) if isinstance(kernel_val, list) else str(kernel_val)
                kae_filename = f"{fname[:-3]}_slice{slice_idx}_L{lmbd_str}_ACS{acs_str}_K{kernel_str}_D{dim_str}.pth"
                kae_file = os.path.join(weights_root, kae_filename)

                print(kae_file)

                if os.path.exists(kae_file):
                    print(f"    [KAE] Loading weights {kae_filename}")
                    ckpt = torch.load(kae_file, map_location=p.get("device", "cpu"))
                    try:
                        recon, log = load_run_kae(tensor, p, ckpt,
                                                    model_class=cKAE_1nl_1l, trainer_class=Trainer,
                                                    global_norm=global_max)
                    except Exception as exc:
                        print(f"    [KAE] Load failed ({exc}), retraining")
                        recon, log = train_run_model_kae(tensor, p,
                                                            model_class=cKAE_1nl_1l, trainer_class=Trainer,
                                                            global_norm=global_max, debug_training=False)
                else:
                    recon, log = train_run_model_kae(tensor, p,
                                                        model_class=cKAE_1nl_1l, trainer_class=Trainer,
                                                        global_norm=global_max, debug_training=False)
                    plot_loss_curve(log["losses"], title=f"Loss — {fname} sl{slice_idx}", save_path=save_path+"/losses/", filename=f"{exp_name}_{fname[:-3]}_sl_{slice_idx}")



                results[fname][slice_idx] = {
                    "kspace": recon,
                    "image":  kspace_to_image(recon.conj()),
                }
            logs[fname][slice_idx] = log
            img_path_methodB = os.path.join(save_path, "imgs/KAE/")
            plot_kspace_images(recon, plot_type="kspace", save_dir=os.path.join(img_path_methodB,f"{exp_name}_{fname[:-3]}_sl_{slice_idx}"))
            plot_windowed_recon(recon.conj(), gt_data=tensor.conj(), save_bool=True, filepath=img_path_methodB, save_name=f"{exp_name}_{fname[:-3]}_sl_{slice_idx}")
            # investigation_results = model_evaluate_errors(tensor, params, log["model"], slice_idx)



    return results, logs

# ==================================================
# 12. EVALUATION & VISUALISATION
# ==================================================

def evaluate_and_save_experiments(all_results, master_baseline,
                                   filenames, save_path,
                                   zoom_x=170, zoom_y=160,
                                   box_size=(40, 40)):
    """
    Compare all experiment variants against baselines.
    Generates one grid per (file, slice) and saves summary CSV.
    """
    os.makedirs(save_path, exist_ok=True)
    summary_stats    = []
    experiment_names = list(all_results.keys())
    if not experiment_names:
        print("[WARN] No results to evaluate."); return None

    for fname in filenames:
        first_exp = experiment_names[0]
        slices    = sorted(all_results[first_exp].get(fname, {}).keys())

        for sl in slices:
            bl_data = master_baseline.get(fname, {}).get(sl, {})
            gt_img  = np.abs(bl_data.get("Ground Truth", {}).get(
                "image", np.zeros((1, 1))))
            
            has_acs = "ACS" in bl_data
            gt_acs = np.abs(bl_data.get("ACS", {}).get(
                "image", np.zeros((1, 1))))
            images_to_plot = [gt_img]
            acs_imgs = [gt_acs]
            
            methods_name   = ["Ground Truth"]
            acs_methods = ["GT"]
            # Baselines
            # for ref in ["GRAPPA", "RAKI", "Koopman Baseline"]:
            for ref in ["GRAPPA", "RAKI"]:
                if ref in bl_data:
                    img_ref         = np.abs(bl_data[ref]["image"])
                    n_std, n_masked, mask = nrmse_mask(gt_img, img_ref)
                    # s = ssim_func(gt_img, img_ref, data_range=gt_img.max() - gt_img.min())
                    s, sm = ssim_mssim(gt_img, img_ref, mask)

                    images_to_plot.append(img_ref)
                    # methods_name.append(
                    #     f"{ref}\nNRMSE:{n_std:.4f}\nM:{n_masked:.4f}\n ssim:{s:.4f}\n mssim:{sm:.4f}")
                    methods_name.append(
                        f"{ref}\nNRMSE:{n_masked:.4f}\n SSIM:{sm:.4f}")
                    summary_stats.append({
                        "Experiment": ref, "File": fname, "Slice": sl,
                        "NRMSE": n_std, "MNRMSE": n_masked, "SSIM": s, "MSSIM": sm})

            # Experiment variants
            for exp_name in experiment_names:
                data      = all_results[exp_name].get(fname, {}).get(sl, {})
                img_recon = np.abs(data.get("image", np.zeros_like(gt_img)))
                n_std, n_masked, mask = nrmse_mask(gt_img, img_recon)
                # s = ssim_func(gt_img, img_recon, data_range=gt_img.max() - gt_img.min())
                s, sm = ssim_mssim(gt_img, img_recon, mask)
                images_to_plot.append(img_recon)

                methods_name.append(
                    f"{exp_name.replace('_','.').upper()}"
                    f"\nNRMSE:{n_masked:.4f}\n SSIM:{sm:.4f}")
                # methods_name.append(
                #     f"{exp_name.replace('_',' ').upper()}"
                #     f"\nNRMSE:{n_std:.4f}\nM:{n_masked:.4f}\n ssim:{s:.4f}\n mssim:{sm:.4f}")
                summary_stats.append({
                    "Experiment": exp_name, "File": fname, "Slice": sl,
                    "NRMSE": n_std, "MNRMSE": n_masked, "SSIM": s,  "MSSIM": sm})
                
                if has_acs:
                    img_acs_recon = np.abs(data.get("acs_image", np.zeros_like(gt_acs)))
                    acs_n_std, acs_n_masked, acs_mask = nrmse_mask(gt_acs, img_acs_recon)
                    acs_s, acs_sm = ssim_mssim(gt_acs, img_acs_recon, acs_mask)
                    acs_imgs.append(img_acs_recon)

                    acs_methods.append(
                        f"{exp_name.replace('_',' ').upper()}"
                        f"\nNRMSE:{acs_n_std:.4f}\nM:{acs_n_masked:.4f}\n ssim:{acs_s:.4f}\n mssim:{acs_sm:.4f}")

            if has_acs:
                mplot(
                    acs_imgs, acs_methods,
                    window_perc=99.9, subfactor=1,
                    x=zoom_x, y=zoom_y, box_size=box_size,
                    filename=f"ACS_debug_{fname[:-3]}_sl{sl}",
                    save_path=save_path)

            # # reordering
            # methods_name = (
            #     [methods_name[0]]
            #     + [methods_name[2]]
            #     + methods_name[3:]
            #     + [methods_name[1]]
                
            # )

            # images_to_plot = (
            #     [images_to_plot[0]]
            #     + [images_to_plot[2]]
            #     + images_to_plot[3:]
            #     + [images_to_plot[1]]
                
            # )
            # ###
            mplot_close(
                images_to_plot, methods_name,
                window_perc=99.9, subfactor=1,
                x=zoom_x, y=zoom_y, box_size=box_size,
                filename=f"Comp_{fname[:-3]}_sl{sl}",
                save_path=save_path, layout="vertical")

    df = pd.DataFrame(summary_stats)
    csv_path = os.path.join(save_path, "final_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


def compute_nmse(gt, pred):
    """Calcule la NMSE entre le ground truth et la prédiction (complexe)."""
    mse = np.sum(np.abs(gt - pred)**2)
    norm = np.sum(np.abs(gt)**2)
    return mse / norm


def evaluate_with_debug(model, X_np, Y_np, batch_size=4096, device='cuda'):
    """Évalue le modèle par batch en utilisant debug_forward pour les stats et samples."""
    model.eval()
    model.to(device)
    N = X_np.shape[0]
    all_preds, debug_stats, samples = [], {}, {}

    with torch.no_grad():
        for b in range(0, N, batch_size):
            end = min(b + batch_size, N)
            x_batch = torch.tensor(X_np[b:end], dtype=torch.complex64, device=device).reshape(end-b, -1)
            
            # Passage unique dans debug_forward
            debug_out = model.debug_forward(x_batch)
            all_preds.append(debug_out["decoder"].cpu().numpy())

            for key, tensor in debug_out.items():
                abs_val = torch.abs(tensor).cpu()
                if key not in debug_stats:
                    debug_stats[key] = {"sparsity": [], "mean": []}
                    samples[key] = abs_val.numpy().flatten() # On garde le 1er batch pour l'histogramme
                
                debug_stats[key]["sparsity"].append((abs_val < 1e-8).float().mean().item())
                debug_stats[key]["mean"].append(abs_val.mean().item())

    final_stats = {k: {"sparsity": np.mean(v["sparsity"])*100, "mean": np.mean(v["mean"])} for k, v in debug_stats.items()}
    return np.concatenate(all_preds, axis=0), final_stats, samples


def run_investigation(model, acs, kspace, kernel, R, slice_idx, batch_size=4096, device='cuda'):
    print(f"\n" + "="*70)
    print(f" INVESTIGATION : - Slice {slice_idx}")
    print("="*70)
    
    # 1. Génération des données
    X_tr, Y_tr = generate_data_vec(acs, kernel, R)
    X_te, Y_te = generate_test_data_vec(kspace, kernel, R)
    
    n_src = kernel[0] * kernel[1] # Indice de césure entre Sources et Targets
    
    # 2. Évaluation (Extraction des prédictions et des stats internes)
    pred_tr_flat, stats_tr, samples_tr = evaluate_with_debug(model, X_tr, Y_tr, batch_size, device)
    pred_te_flat, stats_te, samples_te = evaluate_with_debug(model, X_te, Y_te, batch_size, device)
    
    # 3. Reshape des prédictions pour retrouver la forme (N, Coils, Features)
    pred_tr = pred_tr_flat.reshape(Y_tr.shape)
    pred_te = pred_te_flat.reshape(Y_te.shape)
    
    # 4. Calculs NMSE Séparés (Global, Sources, Targets)
    nmse = {
        "tr_glob": compute_nmse(Y_tr, pred_tr),
        "tr_src":  compute_nmse(Y_tr[:, :, :n_src], pred_tr[:, :, :n_src]),
        "tr_tgt":  compute_nmse(Y_tr[:, :, n_src:], pred_tr[:, :, n_src:]),
        
        "te_glob": compute_nmse(Y_te, pred_te),
        "te_src":  compute_nmse(Y_te[:, :, :n_src], pred_te[:, :, :n_src]),
        "te_tgt":  compute_nmse(Y_te[:, :, n_src:], pred_te[:, :, n_src:])
    }
    
    # 5. Calcul de la sparsité (zeros) spécifiquement pour le Décodeur (Sources vs Targets)
    stats_tr["decoder_src"] = {"sparsity": (np.abs(pred_tr[:, :, :n_src]) < 1e-8).mean() * 100, "mean": np.abs(pred_tr[:, :, :n_src]).mean()}
    stats_tr["decoder_tgt"] = {"sparsity": (np.abs(pred_tr[:, :, n_src:]) < 1e-8).mean() * 100, "mean": np.abs(pred_tr[:, :, n_src:]).mean()}
    
    stats_te["decoder_src"] = {"sparsity": (np.abs(pred_te[:, :, :n_src]) < 1e-8).mean() * 100, "mean": np.abs(pred_te[:, :, :n_src]).mean()}
    stats_te["decoder_tgt"] = {"sparsity": (np.abs(pred_te[:, :, n_src:]) < 1e-8).mean() * 100, "mean": np.abs(pred_te[:, :, n_src:]).mean()}

    # Remplacement du sample global 'decoder' par les versions splittées pour le graphique
    if "decoder" in samples_tr:
        del samples_tr["decoder"]
        del samples_te["decoder"]
        
    samples_tr["decoder_src"] = np.abs(pred_tr[:, :, :n_src]).flatten()
    samples_tr["decoder_tgt"] = np.abs(pred_tr[:, :, n_src:]).flatten()
    
    samples_te["decoder_src"] = np.abs(pred_te[:, :, :n_src]).flatten()
    samples_te["decoder_tgt"] = np.abs(pred_te[:, :, n_src:]).flatten()

    # 6. Affichage structuré dans la console
    print(f"{'MÉTRIQUE':<18} | {'TRAIN (ACS)':<20} | {'TEST (HORS-ACS)':<20}")
    print("-" * 65)
    print(f"{'NMSE Globale':<18} | {nmse['tr_glob']:<20.6f} | {nmse['te_glob']:<20.6f}")
    print(f"{'NMSE Sources':<18} | {nmse['tr_src']:<20.6f} | {nmse['te_src']:<20.6f}")
    print(f"{'NMSE Targets':<18} | {nmse['tr_tgt']:<20.6f} | {nmse['te_tgt']:<20.6f}")
    print("-" * 65)
    print(f"{'% Zéros (Sources)':<18} | {stats_tr['decoder_src']['sparsity']:<19.2f}% | {stats_te['decoder_src']['sparsity']:<19.2f}%")
    print(f"{'% Zéros (Targets)':<18} | {stats_tr['decoder_tgt']['sparsity']:<19.2f}% | {stats_te['decoder_tgt']['sparsity']:<19.2f}%")
    print("=" * 70)
    
    
    return nmse

def model_evaluate_errors(kspace, params, model_obj, slice_idx):
    p = {k: _unwrap(v) for k, v in params.items()}
    
    # Normalisation
    scaled = kspace.copy().astype(np.complex64)
    if p.get('outerbound') is not None:
        nor = params["outerbound"] / np.max(np.abs(scaled))
        scaled *= nor

    # On récupère l'ACS pour le training set
    _, acs = undersample(scaled, r=p['R'], num_acs=p['number_columns'])
    
    # Lancement de l'investigation
    return run_investigation(model_obj, acs, scaled, p['kernel'], p['R'], 
                             slice_idx=slice_idx, batch_size=4096, device=p['device'])



# ==================================================
# 13. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    base_save_path =  "D:/exp_RESULTS/CDT_exp/093026/"
    exp_folder = 'paper_imgszoom/dim2/'
    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR      = "D:/fastmri/multicoil_train/"
    # filename_list = [
    #     # "file_brain_AXT1POST_201_6002668.h5",
    #     "file_brain_AXT1POST_210_6001672.h5",
    #     # "file_brain_AXT1PRE_200_6002304.h5",
    #     # "file_brain_AXT1PRE_200_6002309.h5",
    #     #"file_brain_AXT1_202_2020146.h5",
    #     # "file_brain_AXT1_202_2020334.h5",
    #     # "file_brain_AXT2_200_2000485.h5",
    #     # "file_brain_AXT2_210_2100201.h5",
    #     #"file_brain_AXT2_200_6002197.h5",
    #     # "file_brain_AXFLAIR_201_6002934.h5",
    #     # "file_brain_AXFLAIR_201_6002914.h5",
    #     # "file_brain_AXFLAIR_200_6002512.h5",


    # ]
    # lambda_by_contrast = {
    #     "AXT1POST": 0.16,
    #     "AXT1PRE":  0.15,
    #     "AXT1":  0.15,
    #     "AXT2":     0.09,
    #     "AXFLAIR":  0.12,
    # }
    filename_list = [
        # "file_brain_AXT1POST_200_6002392.h5",
        # "file_brain_AXT1PRE_200_6002304.h5",
        # "file_brain_AXT2_200_2000600.h5",
        # "file_brain_AXFLAIR_200_6002469.h5",
        ## "file_brain_AXFLAIR_209_6001386.h5",
        # "file_brain_AXT1POST_200_6002122.h5"
        "file_brain_AXT2_210_6001534.h5"
    ]

    SAVE_PATH = os.path.join(base_save_path, exp_folder)

    # all_data needed for global normalisation across slices
    all_data  = process_fastmri_h5(DATA_DIR, filelist=filename_list,  all_slices=True)
    data_dict = process_fastmri_h5(DATA_DIR, filelist=filename_list, slice_idx=[0])

    # ── Baseline: GRAPPA + standard KAE ──────────────────────────────
    base_params = {
        'optimizer':         'adam',
        'momentum_rate':     None,
        'epochs':            4000,
        'lr_weight':         1e-3,
        'lr_bias':           1e-3,
        'l2_param_name':    'latent',
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
        save_path         = SAVE_PATH, # "./results/baseline/",
        save_path_raki_weights="D:/RAKI_weights/",
        save_path_koopman = "D:/Baseline_KAE_draft/",# "./weights/baseline/",
    )

    # ── Shared config for ablation experiments ────────────────────────
    COMMON_CONFIG = {
        **base_params,
        'epochs':            4000,
        'separate_lrs':      False,
        'lr_weight': 1e-3, 'lr_bias':1e-3,
        'optimizer':         'adam',
        "debug_training": False

    }

    # ── Experiments ───────────────────────────────────────────────────
    #
    # Keys you can vary:
    #   activation_name  : 'modrelu'|'cardioid'|'softshrink'|'swish'|'tanh'
    #   standardise_coils: True|False
    #   loss_type        : 'mse'|'nrmse'|'weighted'|'frob'|'weighted_frob'
    #   separate_lrs     : True|False
    #   lr_weight, lr_bias, epochs, reduc_d
    #   norm_type         : div_std, single, standardization
    # 'lambda_s', 'lambda_t'
    #         'l2_param_name':    'latent'|'encoder'|'decoder'|'encoderndecoder'|'all'|
    # ─────────────────────────────────────────────────────────────────

   
    # kernels = ["5x2", "5x4", "5x6", "3x4", "7x4", "9x4"]
    lambdas = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0]
    print(lambdas)
    my_experiments = {}

    for lmbd in lambdas:

        lmbd_str = f"{lmbd}".replace(".", "_")
        exp_name = f"{lmbd_str}"
        
        my_experiments[exp_name] = (
            cKAE_1nl_1l,
            Trainer,
            {
                **COMMON_CONFIG,
                'epochs':            4000,
                'reduc_d':           lmbd,
                'loss_type':         'weighted',
                'lambda_s': 0.12,
                'lambda_t': 1.0,
                'norm_type':         'global',
                'separate_lrs':      False,
                'lr_weight': 1e-3, 'lr_bias': 1e-3,


            }
       )






    print("\n" + "="*60 + "\nEXPERIMENTS\n" + "="*60)
    all_res, all_logs = run_koopman_experiments(
        all_data_dict    = all_data,
        data_dict        = data_dict,
        filenames        = filename_list,
        experiments_dict = my_experiments,
        save_path        = SAVE_PATH
    )

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    summary_df = evaluate_and_save_experiments(
        all_results     = all_res,
        master_baseline = master_baseline,
        filenames       = filename_list,
        save_path       = SAVE_PATH,
        zoom_x=170, zoom_y=200, box_size=(60, 60),
    )

    print("\nDone.")
    if summary_df is not None:
        print(summary_df.groupby("Experiment")[["NRMSE","MNRMSE"]].mean().round(4))


