import math
import os

import numpy as np
import pandas as pd
from skimage import filters
from skimage.metrics import structural_similarity as ssim_func


HAS_SKIMAGE = True


def nrmse(gt, pred):
    gt_m, pr_m = np.abs(gt), np.abs(pred)
    return (np.sqrt(np.mean((gt_m - pr_m) ** 2))
            / (np.sqrt(np.mean(gt_m ** 2)) + 1e-12))


def round_sig(x, sig=4):
    if x == 0:
        return 0.0
    d = math.ceil(math.log10(abs(x) + 1e-300))
    return round(float(x), sig - d)


def print_metrics(gt, pred, print_bool=True):
    n = nrmse(gt, pred)
    result = {"nrmse": n, "ssim": float("nan")}
    gt_m = np.abs(gt)
    pr_m = np.abs(pred)
    s = ssim_func(gt_m, pr_m, data_range=gt_m.max() - gt_m.min() + 1e-12)
    result["ssim"] = s
    if print_bool:
        print(f"    NRMSE={result['nrmse']:.4f}  SSIM={result['ssim']:.4f}")
    return result


def get_nrmse_metrics_new(gt, pred, method='mean', p_val=50):
    gt_m = np.abs(gt)
    n_std = nrmse(gt, pred)

    if method == 'percentile':
        thresh = np.percentile(gt_m, p_val)
    elif method == 'otsu':
        thresh = filters.threshold_otsu(gt_m)
    elif method == 'li':
        thresh = filters.threshold_li(gt_m)
    else:
        thresh = np.mean(gt_m)

    mask = gt_m > thresh
    n_masked = nrmse(gt[mask], pred[mask]) if np.any(mask) else n_std
    return n_std, n_masked, mask


def ssim_mssim(gt_img, img, mask):
    s = ssim_func(gt_img, img, data_range=gt_img.max() - gt_img.min())
    mgt_img = gt_img[mask]
    mimg = img[mask]
    ms = ssim_func(mgt_img, mimg, data_range=mgt_img.max() - mgt_img.min())
    return s, ms


def dict_metrics(images, titles, run_dict, save_path=None, save_name=None,
                 print_bool=True):
    all_m = []
    gt_img = images[0]
    for name, img in zip(titles[1:], images[1:]):
        m = print_metrics(gt_img, img, print_bool=print_bool)
        ssim_val = round_sig(m["ssim"])
        nrm_val = round_sig(m["nrmse"])
        all_m.append({"ssim": ssim_val, "nrmse": nrm_val})
        if save_path:
            row = {
                "filename": run_dict.get("filename", ""),
                "slice": run_dict.get("slice", ""),
                "model": name,
                "acc_rate": run_dict.get("R", ""),
                "ACS_lines": run_dict.get("number_columns", ""),
                "ssim": ssim_val,
                "nrmse": nrm_val,
            }
            csv_file = os.path.join(save_path, save_name or "results.csv")
            pd.DataFrame([row]).to_csv(
                csv_file, mode='a', index=False,
                header=not os.path.exists(csv_file)
            )
    return all_m


def nrmse_mask(gt, pred, method='triangle', p_val=50):
    gt_m = np.abs(gt)
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
