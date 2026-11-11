
# Libraries & Modules
import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
from src.KAE_vec_investigation.networks import (cKAE)
from src.KAE_vec_investigation.training import (TrainerDebugVal)
from src.KAE_vec_investigation.run_koopman import (train_run_model_kae_debug, train_run_model_kae, load_run_kae, undersample, generate_data_vec, generate_test_data_vec)
from helpers.preprocessing.preprocessing import kspace_to_image


def plot_extensive_nmse(results_dict, epoch=-1, cmap='hot', vmax=None):
    """
    Génère des subplots de la carte NMSE pour chaque coil.
    Crée une figure pour le Training et une autre pour la Validation.
    
    Arguments:
    - results_dict : Le dictionnaire retourné par la méthode train()
    - epoch        : L'index de l'époque à afficher (par défaut -1, la dernière époque)
    - cmap         : La colormap (hot, viridis, magma...)
    - vmax         : Valeur maximale pour bloquer l'échelle des couleurs (utile pour comparer)
    """
    
    if "nmse_pos_train_history" not in results_dict:
        print("Erreur : Les données de tracking spatial ne sont pas dans le dictionnaire.")
        return

    # 1. Récupération des données pour l'époque ciblée (Shape: c, nx, ny)
    train_map = results_dict["nmse_pos_train_history"][epoch]
    c, nx, ny = train_map.shape
    
    has_val = "nmse_pos_val_history" in results_dict
    if has_val:
        val_map = results_dict["nmse_pos_val_history"][epoch]

    # 2. Définition de la disposition de la grille
    # On limite à 6 ou 8 colonnes max pour que ça reste lisible
    cols = min(c, 6) 
    rows = math.ceil(c / cols)
    
    # === FONCTION INTERNE POUR DESSINER UNE FIGURE ===
    def _draw_figure(data_map, title_prefix):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Gérer le cas où il n'y a qu'une seule ligne (axes n'est pas 2D)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes_flat = np.array(axes).flatten()
        
        # On calcule un vmax dynamique (99ème centile) pour ignorer les valeurs extrêmes isolées
        # si aucun vmax n'a été fourni manuellement
        current_vmax = vmax if vmax is not None else np.percentile(data_map, 99)
        
        fig.suptitle(f'{title_prefix} NMSE Map - Epoch {epoch if epoch >= 0 else "Finale"}', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        for i in range(c):
            ax = axes_flat[i]
            # origin='lower' est souvent préférable pour l'espace-k / images
            im = ax.imshow(data_map[i], cmap=cmap, origin='lower', vmin=0, vmax=current_vmax)
            ax.set_title(f'Coil {i}', fontsize=10)
            ax.axis('off')
            
            # Ajouter une barre de couleur individuelle, ajustée à la taille du subplot
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        # Masquer les subplots vides si le nombre de coils n'est pas un multiple de 'cols'
        for j in range(c, len(axes_flat)):
            axes_flat[j].axis('off')
            axes_flat[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()

    # 3. Génération des plots
    _draw_figure(train_map, "TRAINING")
    
    if has_val:
        _draw_figure(val_map, "VALIDATION")

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

def mplot(images, names, window_perc=99.9, subfactor=1,
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

def plot_save(data, save_bool=None, filepath=None, save_name=None):
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


# ==================================================
# 10. BASELINE ORCHESTRA  (GRAPPA + baseline KAE)
# ==================================================
def crop_to_signal_bbox(tensor, val_map, threshold=1e-12):
    """
    Crop spatial basé sur le support non nul du tensor (auto-détection du padding).
    """
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()

    # --- magnitude reference ---
    # On gère [C, H, W] ou [H, W] ou [C, H, W, 2]
    if tensor.ndim >= 3:
        # On fait la somme des magnitudes sur les axes non spatiaux pour être robuste
        ref = np.abs(tensor).sum(axis=0)
        if ref.ndim == 3: # cas complexe (H, W, 2)
            ref = ref.sum(axis=-1)
    else:
        ref = np.abs(tensor)

    mask = ref > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        return val_map, (0, val_map.shape[0], 0, val_map.shape[1])

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # crop (+1 pour inclure le dernier pixel)
    val_crop = val_map[x_min:x_max+1, y_min:y_max+1]

    return val_crop, (x_min, x_max, y_min, y_max)


def plot_experiment_comparison(
    logs_dict,
    results_dict,
    epoch=-1,
    cmap="jet",
    vmax_factor=0.99,
    use_log=True,
):
    flat_logs = []
    all_maps = []

    for fname, slices in logs_dict.items():
        for sl_idx, log_data in slices.items():
            if "nmse_pos_train_history" not in log_data:
                continue

            title = f"{fname}\nsl:{sl_idx}"

            train_map = np.array(log_data["nmse_pos_train_history"][epoch]).mean(axis=0)
            flat_logs.append((title, log_data, train_map))
            all_maps.append(train_map)

            gt_tensor = results_dict[fname][sl_idx]["gt_tensor"]
            if hasattr(gt_tensor, "cpu"):
                gt_tensor = gt_tensor.cpu().numpy()

            # --- VAL avec suppression du padding ---
            if "nmse_pos_val_history" in log_data:
                val_map = np.array(log_data["nmse_pos_val_history"][epoch]).mean(axis=0)

                val_map_crop, _ = crop_to_signal_bbox(
                                    gt_tensor, 
                                    val_map, 
                                    threshold=1e-12
                                )
                all_maps.append(val_map_crop)

    if not flat_logs:
        print("No spatial tracking data found.")
        return

    # --- GLOBAL SCALING ---
    all_vals = np.concatenate([m.flatten() for m in all_maps])
    all_vals = all_vals[np.isfinite(all_vals)]

    if use_log:
        positive_vals = all_vals[all_vals > 0]
        vmin = np.min(positive_vals) if len(positive_vals) > 0 else 1e-10
    else:
        vmin = 0.0

    vmax = np.percentile(all_vals, vmax_factor * 100) if len(all_vals) > 0 else 1.0

    if use_log and vmin >= vmax:
        vmin = vmax / 10

    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if use_log else colors.Normalize(vmin=vmin, vmax=vmax)

    # --- LAYOUT ---
    n_exps = len(flat_logs)
    has_val = any("nmse_pos_val_history" in l[1] for l in flat_logs)
    n_rows = 2 if has_val else 1

    fig, axes = plt.subplots(n_rows, n_exps, figsize=(4 * n_exps, 4 * n_rows), squeeze=False)

    im = None

    for i, (title, log, train_map) in enumerate(flat_logs):

        # TRAIN
        ax_t = axes[0, i]
        im = ax_t.imshow(train_map, cmap=cmap, origin="lower", norm=norm)
        ax_t.set_title(f"TRAIN: {title}", fontsize=10)
        ax_t.axis("off")

        # VAL
        if n_rows > 1:
            ax_v = axes[1, i]

            if "nmse_pos_val_history" in log:
                val_map = np.array(log["nmse_pos_val_history"][epoch]).mean(axis=0)

                val_map_crop, _ = crop_to_signal_bbox(gt_tensor, val_map)
                
                ax_v.imshow(val_map_crop, cmap=cmap, origin="lower", norm=norm)
                ax_v.set_title(f"VAL: {title}", fontsize=10)
            else:
                ax_v.set_visible(False)

            ax_v.axis("off")

    # --- COLORBAR ---
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("NMSE")

    plt.show()

def _unwrap(v):
    """[False] → False,  [1e-3] → 1e-3,  etc. — backward compat."""
    return v[0] if isinstance(v, list) and len(v) == 1 else v

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


    img_path = os.path.join(save_path, "imgs/RAKI/" )
    plot_save(raki_recon.conj(), save_bool=True, filepath=img_path, save_name=f"{filename}_sl_{slice_idx}")


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
                            params, model_class, trainer_class, save_path, exp_name=""):
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
            recon, log = train_run_model_kae_debug(
                tensor, p, model_class=model_class, trainer_class=trainer_class, global_norm=global_max)
            


            img_path_methodB = os.path.join(save_path, "imgs/KAE/")
            plot_save(recon.conj(), save_bool=True, filepath=img_path_methodB, save_name=f"{fname[:-3]}_sl_{slice_idx}")
            plot_loss_curve(log["losses"], title=f"Loss — {fname} sl{slice_idx}", save_path=save_path+"/losses/", filename=f"{exp_name}_{fname[:-3]}_sl_{slice_idx}")
            # investigation_results = model_evaluate_errors(tensor, params, log["model"], slice_idx)

            results[fname][slice_idx] = {
                "gt_tensor": tensor,
                "kspace": recon,
                "image":  kspace_to_image(recon.conj()),
            }
            logs[fname][slice_idx] = log
    
    plot_experiment_comparison(logs, results, cmap='jet')

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

            images_to_plot = [gt_img]
            methods_name   = ["Ground Truth"]

            # Baselines
            for ref in ["GRAPPA", "RAKI", "Koopman Baseline"]:
                if ref in bl_data:
                    img_ref         = np.abs(bl_data[ref]["image"])
                    n_std, n_masked, mask = nrmse_mask(gt_img, img_ref)
                    # s = ssim_func(gt_img, img_ref, data_range=gt_img.max() - gt_img.min())
                    s, sm = ssim_mssim(gt_img, img_ref, mask)

                    images_to_plot.append(img_ref)
                    methods_name.append(
                        f"{ref}\nNRMSE:{n_std:.4f}\nM:{n_masked:.4f}\n ssim:{s:.4f}\n mssim:{sm:.4f}")
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
                    f"{exp_name.replace('_',' ').upper()}"
                    f"\nNRMSE:{n_std:.4f}\nM:{n_masked:.4f}\n ssim:{s:.4f}\n mssim:{sm:.4f}")
                summary_stats.append({
                    "Experiment": exp_name, "File": fname, "Slice": sl,
                    "NRMSE": n_std, "MNRMSE": n_masked, "SSIM": s,  "MSSIM": sm})

            mplot(
                images_to_plot, methods_name,
                window_perc=99.9, subfactor=1,
                x=zoom_x, y=zoom_y, box_size=box_size,
                filename=f"Comp_{fname[:-3]}_sl{sl}",
                save_path=save_path)

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

    base_save_path =  "D:/exp_RESULTS/CDT_exp/050126/debug"
    exp_folder = "AXT2_200_2000485_slice_0_14/nmse/draft_kae_analysis_error_positional/"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR      = "D:/fastmri/multicoil_train/"
    filename_list = [
        # "file_brain_AXT1POST_201_6002668.h5",
        # "file_brain_AXT1POST_210_6001672.h5",
        # "file_brain_AXT1PRE_200_6002304.h5",
        # "file_brain_AXT1PRE_200_6002309.h5",
        # "file_brain_AXT1_202_2020146.h5",
        #"file_brain_AXT1_202_2020334.h5",
        "file_brain_AXT2_200_2000485.h5",
        #  "file_brain_AXT2_210_2100201.h5",
        # "file_brain_AXT2_200_6002197.h5",
        # "file_brain_AXFLAIR_201_6002934.h5",
        # "file_brain_AXFLAIR_201_6002914.h5",
        # "file_brain_AXFLAIR_200_6002512.h5",


    ]


    SAVE_PATH = os.path.join(base_save_path, exp_folder)

    # all_data needed for global normalisation across slices
    all_data  = process_fastmri_h5(DATA_DIR, filelist=filename_list, all_slices=True)
    data_dict = process_fastmri_h5(DATA_DIR, filelist=filename_list, slice_idx=[0, 14])

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
        'activation_name': 'modrelu',
        'norm_type':         'single',
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
    # ─────────────────────────────────────────────────────────────────

    my_experiments = {

        'kae_lr1e-4_nsme': (cKAE, TrainerDebugVal, {
            **COMMON_CONFIG,
            'epochs':            4000,
            'loss_type':         'nmse',
            'separate_lrs':      False,
            'lr_weight': 1e-4, 'lr_bias':1e-3,

        }),

    }



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
        zoom_x=170, zoom_y=160, box_size=(40, 40),
    )

    print("\nDone.")
    if summary_df is not None:
        print(summary_df.groupby("Experiment")[["NRMSE","MNRMSE"]].mean().round(4))


