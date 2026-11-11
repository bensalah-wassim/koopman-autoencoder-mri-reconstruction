import argparse
import csv
import tempfile
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from baseline_models.GRAPPA_vectorized.vec_grappa import retrospective_grappa
from helpers.downloaders.data_loaders import process_fastmri_h5
from helpers.preprocessing.preprocessing import kspace_to_image
from baseline_models.RAKI.main import raki_recon_algo

from src.KAE_vec_investigation.networks import cKAE
from src.KAE_vec_investigation.run_koopman import train_run_model_kae, load_run_kae
from src.KAE_vec_investigation.training import Trainer
from src.reader_study import nrmse_mask, ssim_mssim, psnr_mpsnr


def parse_args():
    parser = argparse.ArgumentParser(description="Run a parameter sweep from a JSON config")
    parser.add_argument("--config", required=True, help="Path to the JSON config file")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    # Support JSONC-style // comments and trailing commas.
    in_string = False
    escaped = False
    cleaned = []
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            cleaned.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            cleaned.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        cleaned.append(ch)
        i += 1

    cleaned_text = "".join(cleaned)
    cleaned_text = re.sub(r",(\s*[}\]])", r"\1", cleaned_text)
    return json.loads(cleaned_text)


def parse_slice_selection(raw):
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(v) for v in raw]
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        return [int(v.strip()) for v in raw.split(",") if v.strip()]
    raise ValueError(f"Unsupported slice selection format: {raw}")


def normalize_token_list(raw):
    """Normalize a config token or token list into a list of uppercase strings.
    Accepts None, str (comma-separated), or list. Returns None if input is None.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [v.strip().upper() for v in raw.split(",") if v.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(v).strip().upper() for v in raw if str(v).strip()]
    return [str(raw).strip().upper()]


def format_value(value):
    if isinstance(value, list):
        return "x".join(map(str, value))
    if isinstance(value, float):
        return str(value).replace(".", "_")
    return str(value)


def compute_metrics(gt_img, recon_img):
    # Use reader_study's masked metrics: NRMSE (global), MNRMSE (masked by threshold), SSIM and PSNR
    n_std, n_masked, mask = nrmse_mask(gt_img, recon_img)
    s, ms = (float('nan'), float('nan'))
    ps, mps = (float('nan'), float('nan'))
    try:
        s, ms = ssim_mssim(gt_img, recon_img, mask)
    except Exception:
        s, ms = float('nan'), float('nan')
    try:
        ps, mps = psnr_mpsnr(gt_img, recon_img, mask)
    except Exception:
        ps, mps = float('nan'), float('nan')

    return {
        "NRMSE": float(n_std),
        "MNRMSE": float(n_masked),
        "SSIM": float(s),
        "MSSIM": float(ms),
        "PSNR": float(ps),
        "MPSNR": float(mps),
    }


def save_image(image, gt_img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(10, 10), frameon=False, dpi=96)
    if gt_img is None:
        vmin, vmax = np.quantile(image, (0.01, 0.99))
    else:
        vmin, vmax = np.quantile(gt_img, (0.01, 0.99))
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis("off")
    plt.margins(0, 0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def append_metrics_row(row, csv_path):
    file_exists = os.path.isfile(csv_path)
    new_fields = list(row.keys())

    if not file_exists:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=new_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    # File exists: read existing header
    with open(csv_path, mode="r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    # If new row introduces new fields, rewrite file with union of fields
    added_fields = [f for f in new_fields if f not in existing_fields]
    if added_fields:
        merged_fields = existing_fields + [f for f in new_fields if f not in existing_fields]
        # write to temp file then replace original
        dirpath = os.path.dirname(csv_path) or '.'
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_metrics_", dir=dirpath, text=True)
        os.close(fd)
        try:
            with open(tmp_path, mode="w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=merged_fields)
                writer.writeheader()
                for r in existing_rows:
                    out = {k: r.get(k, "") for k in merged_fields}
                    writer.writerow(out)
                out_new = {k: row.get(k, "") for k in merged_fields}
                writer.writerow(out_new)
            os.replace(tmp_path, csv_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    else:
        # No new fields: append using existing field order
        with open(csv_path, mode="a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_fields)
            writer.writerow({k: row.get(k, "") for k in existing_fields})


DEFAULT_LAMBDA_BY_CONTRAST = {
    "AXT1POST": 0.16,
    "AXT1PRE":  0.15,
    "AXT1":     0.15,
    "AXT2":     0.09,
    "AXFLAIR":  0.12,
}


def resolve_contrast_lambda(item_id, lambda_by_contrast, default=None):
    """Retourne le lambda_s associé au contraste détecté dans item_id.
    Priorité au token le plus spécifique (ex: AXT1POST avant AXT1)."""
    if not lambda_by_contrast:
        return default
    # Case-insensitive token matching to be robust to filename casing
    item_up = item_id.upper()
    matches = [tok for tok in lambda_by_contrast if tok.upper() in item_up]
    if not matches:
        return default
    matches.sort(key=len, reverse=True)
    return lambda_by_contrast[matches[0]]

def matches_contrast_filter(item_id, include_contrasts, exclude_contrasts):
    if include_contrasts:
        if not any(token in item_id for token in include_contrasts):
            return False
    if exclude_contrasts:
        if any(token in item_id for token in exclude_contrasts):
            return False
    return True


def run_from_config(config):
    study_name = config.get("study_name", "parametric_study")
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError("Missing output_dir in config")

    data_dir = config.get("data_dir", "D:/fastmri/multicoil_train/")
    entries_json = config.get("entries_json", "D:/Paper_reader_study_KAE/random_selected_files_mincoils16_v2.json")
    filename_tokens = normalize_token_list(config.get("filename_tokens", ["T1"])) or ["T1"]
    include_contrasts = normalize_token_list(config.get("include_contrasts") or config.get("contrasts") or config.get("contrast_include"))
    exclude_contrasts = normalize_token_list(config.get("exclude_contrasts") or config.get("contrast_exclude"))
    all_slices = bool(config.get("all_slices", False))
    slice_selection = parse_slice_selection(config.get("slice_list"))
    if slice_selection is None and not all_slices:
        slice_selection = [0]

    methods = config.get("methods", ["grappa", "kae"])
    if isinstance(methods, str):
        methods = [m.strip().lower() for m in methods.split(",") if m.strip()]
    methods = [m.lower() for m in methods]
    for method in methods:
        if method not in {"grappa", "kae", "raki"}:
            raise ValueError(f"Unsupported method: {method}")

    min_coils = int(config.get("min_coils", 15))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(config.get("epochs", 4000))
    base_r = int(config.get("base_r", 4))
    base_number_columns = int(config.get("base_number_columns", 40))
    base_kernel = config.get("base_kernel", "5x4")
    base_reduc_d = float(config.get("base_reduc_d", 0.8))
    base_lambda_s = float(config.get("base_lambda_s"), 0.12)
    # If the config explicitly sets null or an empty mapping, fall back to defaults
    lambda_by_contrast = config.get("lambda_by_contrast") or DEFAULT_LAMBDA_BY_CONTRAST
    weights_dir = config.get("weights_dir")
    raki_weights_dir = config.get("raki_weights_dir")

    with open(entries_json, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    selected_ids = []
    for item in entries:
        item_id = item.get("id")
        if not item_id:
            continue
        item_id_up = item_id.upper()
        if not any(token in item_id_up for token in filename_tokens):
            continue
        if not matches_contrast_filter(item_id_up, include_contrasts, exclude_contrasts):
            continue
        selected_ids.append(item_id)

    if not selected_ids:
        raise ValueError("No entries matched the requested filename tokens and contrast filters")

    print(f"Selected {len(selected_ids)} files")

    data_dict = process_fastmri_h5(
        data_dir,
        filelist=selected_ids,
        all_slices=True,
        slice_idx=None,
        min_coils=min_coils,
    )

    common_config = {
        "optimizer": "adam",
        "momentum_rate": None,
        "device": device,
        "epochs": epochs,
        "separate_lrs": False,
        "lr_weight": 1e-3,
        "lr_bias": 1e-3,
        "l2_param_name": "latent",
        "l2": False,
        "l2_lambda": None,
        "ACS_flag": True,
        "kernel": [int(p) for p in base_kernel.split("x") if p.strip()] if "x" in base_kernel else [5, 4],
        "number_columns": base_number_columns,
        "R": base_r,
        "outerbound": 1.0,
        "norm_type": "global",
        "bias": False,
        "beta_init": 0.0,
        "activation_name": "modrelu",
        "reduc_d": base_reduc_d,
        "loss_type": "weighted",
        "lambda_s": None if base_lambda_s is None else base_lambda_s,
        "lambda_t": 1.0,
        "batch_size_recon": 4096,
        "debug_training": False,
    }

    study_root = os.path.join(output_dir, study_name)
    os.makedirs(study_root, exist_ok=True)
    weights_root = os.path.join(study_root, "weights") if weights_dir is None else weights_dir
    os.makedirs(weights_root, exist_ok=True)

    raki_weights_root = raki_weights_dir or os.path.join(study_root, "RAKI_weights")
    os.makedirs(raki_weights_root, exist_ok=True)
    studies = config.get("studies", [])
    if not studies:
        raise ValueError("No studies defined in config")

    for study in studies:
        study_type = study.get("type")
        values = study.get("values")
        if not study_type or values is None:
            raise ValueError("Each study needs a 'type' and 'values'")

        param_name = {
            "kernel": "kernel",
            "reduc_d": "reduc_d",
            "lambda_s": "lambda_s",
            "number_columns": "number_columns",
        }.get(study_type)
        if param_name is None:
            raise ValueError(f"Unsupported study type: {study_type}")

        if isinstance(values, str):
            values = [v.strip() for v in values.split(",") if v.strip()]

        for value in values:
            if study_type == "kernel":
                if isinstance(value, str) and "x" in value:
                    parsed_value = [int(p.strip()) for p in value.split("x") if p.strip()]
                else:
                    parsed_value = value
            elif study_type == "number_columns":
                parsed_value = int(value)
            elif study_type == "reduc_d":
                parsed_value = float(value)
            elif study_type == "lambda_s":
                parsed_value = float(value)
            else:
                parsed_value = value

            val_str = format_value(parsed_value)
            exp_dir = os.path.join(study_root, f"{study_type}_{param_name}_{val_str}")
            os.makedirs(exp_dir, exist_ok=True)
            images_root = os.path.join(exp_dir, "images")
            os.makedirs(images_root, exist_ok=True)
            csv_path = os.path.join(exp_dir, "results.csv")

            config_run = dict(common_config)
            if study_type == "kernel":
                config_run["kernel"] = parsed_value
            elif study_type == "reduc_d":
                config_run["reduc_d"] = parsed_value
            elif study_type == "lambda_s":
                config_run["lambda_s"] = parsed_value
            elif study_type in {"acs", "number_columns"}:
                config_run["number_columns"] = parsed_value

            print(f"\n=== Running {study_type}={parsed_value} ===")

            for fname in selected_ids:
                if fname not in data_dict:
                    continue
                slices = data_dict[fname]
                if not slices:
                    continue
                if study_type != "lambda_s":
                    # Use uppercase name for matching inside  
                    resolved = resolve_contrast_lambda(fname.upper(), lambda_by_contrast, default=base_lambda_s)
                    config_run["lambda_s"] = resolved
                    print(f"  [lambda] {fname} -> {config_run['lambda_s']}")

                # Ignore None tensors when computing global normalization max
                vals = [t for t in slices.values() if t is not None]
                if not vals:
                    continue
                global_max = max(float(np.max(np.abs(t))) for t in vals)

                if all_slices:
                    selected_slices = sorted(slices.keys())
                else:
                    selected_slices = [sl for sl in slice_selection if sl in slices]
                    if not selected_slices and 0 in slices:
                        selected_slices = [0]

                if not selected_slices:
                    continue


                for slice_idx in selected_slices:
                    tensor = slices[slice_idx]
                    if tensor is None:
                        continue

                    filename = fname.replace(".h5", "")
                    gt_img = kspace_to_image(tensor.conj())

                    if "grappa" in methods and study_type in {"kernel", "number_columns"}:
                        t0 = time.time()
                        grappa_recon = retrospective_grappa(
                            tensor.copy(), config_run["R"], config_run["number_columns"], config_run["kernel"], ACS_bool=True
                        )
                        rec_time = time.time() - t0
                        grappa_img = kspace_to_image(grappa_recon.conj())
                        metrics = compute_metrics(gt_img, grappa_img)
                        save_image(grappa_img, gt_img, os.path.join(images_root, f"{filename}_slice{slice_idx}_grappa.png"))
                        row = {
                            "Study": study_type,
                            "Param": param_name,
                            "Value": parsed_value,
                            "Experiment": "GRAPPA",
                            "File": filename,
                            "Slice": slice_idx,
                            "RecTime_s": rec_time,
                            **metrics,
                        }
                        append_metrics_row(row, csv_path)
                    
                    if "raki" in methods and study_type in {"number_columns"}:
                        t0 = time.time()
                        raki_recon = raki_recon_algo(
                            tensor.copy(),
                            config_run["R"],
                            config_run["number_columns"],
                            ACS_flag=True,
                            device=config_run.get("device", device),
                            save_path=raki_weights_root, # "D:/RAKI_weights/"
                            slice=slice_idx,
                            filename=fname,
                        )
                        rec_time = time.time() - t0
                        raki_img = kspace_to_image(raki_recon.conj())
                        metrics = compute_metrics(gt_img, raki_img)
                        save_image(raki_img, gt_img, os.path.join(images_root, f"{filename}_slice{slice_idx}_raki.png"))
                        row = {
                            "Study": study_type,
                            "Param": param_name,
                            "Value": parsed_value,
                            "Experiment": "RAKI",
                            "File": filename,
                            "Slice": slice_idx,
                            "RecTime_s": rec_time,
                            **metrics,
                        }
                        append_metrics_row(row, csv_path)
                    if "kae" in methods:
                        # Build weight filename consistent with reader_study naming
                        lmbd_str = str(config_run.get("lambda_s", "NA")).replace('.', '_')
                        acs_str = str(config_run.get("number_columns", 'NA'))
                        dim_str = str(config_run.get("reduc_d", 'NA')).replace('.', '_')
                        kernel_val = config_run.get("kernel", 'NA')
                        kernel_str = "x".join(map(str, kernel_val)) if isinstance(kernel_val, list) else str(kernel_val)
                        kae_filename = f"{filename}_slice{slice_idx}_L{lmbd_str}_ACS{acs_str}_K{kernel_str}_D{dim_str}.pth"
                        kae_file = os.path.join(weights_root, kae_filename)

                        t0 = time.time()
                        if os.path.exists(kae_file):
                            print(f"  [KAE] Loading weights {kae_file}")
                            ckpt = torch.load(kae_file, map_location=config_run.get('device', 'cpu'))
                            try:
                                recon, log = load_run_kae(tensor, config_run, ckpt, model_class=cKAE_1nl_1l, trainer_class=Trainer, global_norm=global_max)
                            except Exception:
                                # Fallback to training if loading fails
                                recon, log = train_run_model_kae(tensor, config_run, model_class=cKAE_1nl_1l, trainer_class=Trainer, global_norm=global_max, debug_training=False)
                                torch.save({'model_state_dict': log.get('model').state_dict() if log.get('model') is not None else {}, 'params': config_run}, kae_file)
                        else:
                            recon, log = train_run_model_kae(tensor, config_run, model_class=cKAE_1nl_1l, trainer_class=Trainer, global_norm=global_max, debug_training=False)
                            try:
                                torch.save({'model_state_dict': log.get('model').state_dict() if log.get('model') is not None else {}, 'params': config_run}, kae_file)
                            except Exception:
                                pass
                        rec_time = time.time() - t0

                        kae_img = kspace_to_image(recon.conj())
                        metrics = compute_metrics(gt_img, kae_img)
                        save_image(kae_img, gt_img, os.path.join(images_root, f"{filename}_slice{slice_idx}_kae.png"))
                        row = {
                            "Study": study_type,
                            "Param": param_name,
                            "Value": parsed_value,
                            "Experiment": "KAE",
                            "File": filename,
                            "Slice": slice_idx,
                            "RecTime_s": rec_time,
                            **metrics,
                        }
                        append_metrics_row(row, csv_path)

    print(f"\nDone. Results saved under {study_root}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_from_config(config)
