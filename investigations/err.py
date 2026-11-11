
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
from src.KAE_vec_investigation.networks import (cKAE)
from src.KAE_vec_investigation.run_koopman import (train_run_model_kae, load_run_kae, generate_data_vec, generate_test_data_vec, undersample)
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

def ssim_mssim(gt_img, img, mask):

    s = ssim_func(gt_img, img, data_range=gt_img.max() - gt_img.min())
    mgt_img = gt_img[mask]
    mimg    = img[mask]
    ms = ssim_func(mgt_img, mimg, data_range=mgt_img.max() - mgt_img.min())
    return s, ms

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

    print("  [GRAPPA]")
    grappa_recon = retrospective_grappa(
        kspace.copy(), params["R"], params["number_columns"],
        params["kernel"], ACS_bool=True)
    

    
    print("  [RAKI]")
    raki_recon = raki_recon_algo(kspace.copy(), params["R"], params["number_columns"], ACS_flag=True, device='cuda', save_path=save_path_raki_weights, slice=run_dict["slice"], filename=run_dict["filename"])


    print("  [Koopman]")
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
    
    investigation_results = model_evaluate_errors(kspace, params, log["model"], run_dict)
    log["investigation"] = investigation_results
    


    kspace_recons = [kspace, grappa_recon, raki_recon, kae_recon]
    titles        = ["Ground Truth", "GRAPPA", "RAKI", "Koopman Baseline"]
    img_recons    = [kspace_to_image(k.conj()) for k in kspace_recons]
    return kspace_recons, img_recons, titles, log

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

            ksp_r, img_r, titles, log = train_run_baseline(
                tensor, base_params, run_dict,
                save_path_raki_weights=save_path_raki_weights,
                save_path_koopman=save_path_koopman,
                save_path=save_path, save_name="baseline_metrics.csv")
            
            


            slice_data = {}
            for i, title in enumerate(titles):
                slice_data[title] = {
                    "kspace":  ksp_r[i],
                    "image":   img_r[i],
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
            recon, log = train_run_model_kae(
                tensor, p, model_class=model_class, trainer_class=trainer_class, global_norm=global_max)


            results[fname][slice_idx] = {
                "kspace": recon,
                "image":  kspace_to_image(recon.conj()),
            }
            logs[fname][slice_idx] = log

    return results, logs

# ==================================================
# 12. EVALUATION & VISUALISATION
# ==================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
                
                debug_stats[key]["sparsity"].append((abs_val < 1e-20).float().mean().item())
                debug_stats[key]["mean"].append(abs_val.mean().item())

    final_stats = {k: {"sparsity": np.mean(v["sparsity"])*100, "mean": np.mean(v["mean"])} for k, v in debug_stats.items()}
    return np.concatenate(all_preds, axis=0), final_stats, samples

def plot_investigation(st_tr, sm_tr, st_te, sm_te, nmse, info):
    """Génère les plots de distribution des activations avec split Source/Target."""
    layers = [l for l in sm_tr.keys() if l != 'input']
    
    # Création de la figure en fonction du nombre de sous-graphiques (Encoder, Latent, Dec_Src, Dec_Tgt)
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4.5))
    if len(layers) == 1: axes = [axes]
    
    for i, layer in enumerate(layers):
        sns.kdeplot(sm_tr[layer], ax=axes[i], label=f"Train ({st_tr[layer]['sparsity']:.1f}% zeros)", fill=True)
        sns.kdeplot(sm_te[layer], ax=axes[i], label=f"Test ({st_te[layer]['sparsity']:.1f}% zeros)", fill=True)
        axes[i].set_title(f"{layer.upper()}\nMean Tr: {st_tr[layer]['mean']:.2e} | Te: {st_te[layer]['mean']:.2e}")
        axes[i].set_yscale('log')
        axes[i].legend(fontsize=9)

    title_str = (f"Investigation: {info['filename']} - Slice {info['slice']}\n"
                 f"NMSE Target (Lignes manquantes) -> Train: {nmse['tr_tgt']:.5f} | Test: {nmse['te_tgt']:.5f}")
    plt.suptitle(title_str, fontweight='bold')
    plt.tight_layout()
    plt.show()

def run_investigation(model, acs, kspace, kernel, R, run_dict, batch_size=4096, device='cuda'):
    print(f"\n" + "="*70)
    print(f" INVESTIGATION : {run_dict['filename']} - Slice {run_dict['slice']}")
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
    stats_tr["decoder_src"] = {"sparsity": (np.abs(pred_tr[:, :, :n_src]) < 1e-20).mean() * 100, "mean": np.abs(pred_tr[:, :, :n_src]).mean()}
    stats_tr["decoder_tgt"] = {"sparsity": (np.abs(pred_tr[:, :, n_src:]) < 1e-20).mean() * 100, "mean": np.abs(pred_tr[:, :, n_src:]).mean()}
    
    stats_te["decoder_src"] = {"sparsity": (np.abs(pred_te[:, :, :n_src]) < 1e-20).mean() * 100, "mean": np.abs(pred_te[:, :, :n_src]).mean()}
    stats_te["decoder_tgt"] = {"sparsity": (np.abs(pred_te[:, :, n_src:]) < 1e-20).mean() * 100, "mean": np.abs(pred_te[:, :, n_src:]).mean()}

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
    
    # 7. Génération des graphiques
    plot_investigation(stats_tr, samples_tr, stats_te, samples_te, nmse, run_dict)
    
    return nmse

def model_evaluate_errors(kspace, params, model_obj, run_dict):
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
                             run_dict=run_dict, batch_size=4096, device=p['device'])

# ==================================================
# 13. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    base_save_path =  "D:/exp_RESULTS/CDT_exp/042926/"
    exp_folder = "YO"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR      = "D:/fastmri/multicoil_train/"
    filename_list = [
        # "file_brain_AXT1POST_201_6002668.h5",
        # "file_brain_AXT1POST_210_6001672.h5",
        # "file_brain_AXT1PRE_200_6002304.h5",
        # "file_brain_AXT1PRE_200_6002309.h5",
        # "file_brain_AXT1_202_2020146.h5",
        #"file_brain_AXT1_202_2020334.h5",
        # "file_brain_AXT2_200_2000485.h5",
        # "file_brain_AXT2_210_2100201.h5",
        # "file_brain_AXT2_200_6002197.h5",
        # "file_brain_AXFLAIR_201_6002934.h5",
        # "file_brain_AXFLAIR_201_6002914.h5",
        "file_brain_AXFLAIR_200_6002512.h5",


    ]


    SAVE_PATH = os.path.join(base_save_path, exp_folder)

    # all_data needed for global normalisation across slices
    all_data  = process_fastmri_h5(DATA_DIR, filelist=filename_list, all_slices=True)
    data_dict = process_fastmri_h5(DATA_DIR, filelist=filename_list, slice_idx=[0, 8, 12, 14])

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

