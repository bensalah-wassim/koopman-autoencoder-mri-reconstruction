
# Libraries & Modules
import os
import sys

import numpy as np
import torch

# Internal Modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.downloaders.data_loaders import process_fastmri_h5
from helpers.plots.plots import (
    plot_loss_curve,
    plot_windowed_recon,
    plot_kspace_images,
    mplot,
    dynamic_windowing,
    get_zoom_region,
    grid_new,
)
from helpers.preprocessing.preprocessing import kspace_to_image
from helpers.utils.metrics import (
    nrmse,
    nrmse_mask,
    ssim_mssim,
    dict_metrics,
    print_metrics,
    round_sig,
    get_nrmse_metrics_new,
)
from baseline_models.GRAPPA_vectorized.vec_grappa import retrospective_grappa
from baseline_models.RAKI.main import raki_recon_algo
from src.KAE_vec_investigation.networks import cKAE
from src.KAE_vec_investigation.training import Trainer
from src.KAE_vec_investigation.run_koopman import (train_run_model_kae, load_run_kae, undersample, generate_data_vec, generate_test_data_vec)



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

    img_path = os.path.join(save_path, "imgs/GT/" )
    plot_windowed_recon(kspace.conj(), gt_data=kspace.conj(), save_bool=True, filepath=img_path, save_name=f"GT_{filename}_sl_{slice_idx}")


    kspace_recons = [kspace, acs, grappa_recon, raki_recon]
    titles        = ["Ground Truth", "ACS", "GRAPPA", "RAKI"]
    img_recons    = [kspace_to_image(k.conj()) for k in kspace_recons]
    metrics       = dict_metrics(
                        img_recons, 
                        titles,
                        run_dict, save_path, save_name)
    return kspace_recons, img_recons, titles, metrics, {}

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
                                                    model_class=cKAE, trainer_class=Trainer,
                                                    global_norm=global_max)
                    except Exception as exc:
                        print(f"    [KAE] Load failed ({exc}), retraining")
                        recon, log = train_run_model_kae(tensor, p,
                                                            model_class=cKAE, trainer_class=Trainer,
                                                            global_norm=global_max, debug_training=False)
                else:
                    recon, log = train_run_model_kae(tensor, p,
                                                        model_class=cKAE, trainer_class=Trainer,
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


            mplot(
                images_to_plot, methods_name,
                window_perc=99.9, subfactor=1,
                x=zoom_x, y=zoom_y, box_size=box_size,
                filename=f"Comp_{fname[:-3]}_sl{sl}",
                save_path=save_path, layout="horizontal")

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




# ================
# MAIN EXECUTION #
# ================

if __name__ == "__main__":

    base_save_path =  "D:/exp_RESULTS/CDT_exp/093026/"
    exp_folder = 'paper_imgszoom/dim2/'
    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data directory storing fastMRI data
    DATA_DIR      = "D:/fastmri/multicoil_train/"

    lambda_by_contrast = {
        "AXT1POST": 0.16,
        "AXT1PRE":  0.15,
        "AXT1":  0.15,
        "AXT2":     0.09,
        "AXFLAIR":  0.12,
    }
    filename_list = [

        # "file_brain_AXFLAIR_200_6002469.h5",
        # "file_brain_AXFLAIR_209_6001386.h5",
        # "file_brain_AXFLAIR_201_6002934.h5",
        # "file_brain_AXFLAIR_201_6002914.h5",
        # "file_brain_AXFLAIR_200_6002512.h5",
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
    #   activation_name  : 'modrelu'|
    #   loss_type        : 'mse'|'weighted'|
    #   separate_lrs     : True|False
    #   lr_weight, lr_bias, epochs, reduc_d
    #   norm_type         : single, global
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
            cKAE,
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


