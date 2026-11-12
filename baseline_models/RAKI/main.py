
# Libraires
import numpy as np
import torch
import os
import sys


# Modules
from .training import train_model
from .network import RAKI



# utilis


def normalize(kspace, outerbound=0.015):

    nor = outerbound / np.max(abs(kspace[:]))
    return nor, np.multiply(kspace, nor)


def undersample(kspace, r, number_columns):
    dim = kspace.shape
    mask = np.zeros(dim)
    mask[:, :, ::r] = True
    Udata = kspace * mask
    start = dim[2]//2 - number_columns//2
    end = dim[2]//2 + number_columns//2

    # Get the ACS lines and put it back in the undersampled data
    ACS = kspace[:, :, start: end]
    Udata[:, :, start:end] = ACS

    return Udata, ACS


# Train - Save - Run - Load - RAKI functions

def RAKI_run_save(
        kspace, acc_rate, number_columns,
        ACS_flag, device,
        save_path, slice, save_model=True,
        lr=3e-3, epochs=3000, prefix=""):

    dim = kspace.shape
    torch.cuda.empty_cache()
    print("Acceleration rate :", acc_rate)

    nor, normalized_kspace = normalize(kspace, outerbound=0.015)
    cc_kspace = np.concatenate(
        (np.real(normalized_kspace), np.imag(normalized_kspace)), axis=0)
    # Undersampled data
    Udata, ACS = undersample(
        kspace=cc_kspace, r=acc_rate, number_columns=number_columns
        )

    # RAKI
    print(30*"*" + " RAKI " + 30*"*")

    RAKI_model = RAKI(R=acc_rate, nc=dim[0]).to(device)
    RAKI_model, raki_recon = train_model(
        model=RAKI_model,
        calib=ACS,
        Udata=Udata,
        epochs=epochs,
        R=acc_rate,
        draw_curve=False,
        early_stop=False,
        lr=lr,
        silent=True,
        save_bool=save_model,
        save_path=save_path,
        prefix=prefix,
    )
    # IF ACS_flag is True, then we put back the calibration lines.
    if ACS_flag:
        start = dim[2]//2 - number_columns//2
        end = dim[2]//2 + number_columns//2
        raki_recon[:, :, start: end] = ACS

    raki_recon = raki_recon / nor
    return raki_recon


def infer_raki_NEW(
        kspace, path, slice,
        device, R, number_columns,
        ACS_flag, epochs, lr, prefix=""):
    """
    You need to complete that... >> To be continued >>
    """
    dim = kspace.shape
    # Undersampled data
    nor, normalized_kspace = normalize(kspace, outerbound=0.015)
    cc_kspace = np.concatenate(
        (np.real(normalized_kspace), np.imag(normalized_kspace)), axis=0)
    # Undersampled data
    Udata, ACS = undersample(
        kspace=cc_kspace, r=R, number_columns=number_columns
        )
    complete_path = path

    [kernel_x_1, kernel_y_1] = [5, 2]
    [kernel_x_2, kernel_y_2] = [1, 1]
    [kernel_x_3, kernel_y_3] = [3, 2]

    target_x_start = np.int32(
        np.ceil(kernel_x_1 / 2)
        + np.floor(kernel_x_2 / 2)
        + np.floor(kernel_x_3 / 2)
        - 1
    )
    target_x_end = np.int32(dim[1] - target_x_start)
    target_y_start = (
        np.int32(
            (np.ceil(kernel_y_1 / 2) - 1)
            + (np.ceil(kernel_y_2 / 2) - 1)
            + (np.ceil(kernel_y_3 / 2) - 1)
        )
        * R
        + 1
    )
    target_y_end = (
        dim[2]
        - np.int32(
            (
                np.floor(kernel_y_1 / 2)
                + np.floor(kernel_y_2 / 2)
                + np.floor(kernel_y_3 / 2)
            )
        )
        * R
    )
    target_dim_X = target_x_end - target_x_start
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_Z = R - 1

    recon = Udata.copy()
    for coil in range(Udata.shape[0]):

        # Convert to PyTorch tensor and move to device
        input_tensor = torch.tensor(Udata, dtype=torch.float).to(device)
        RAKI_model = RAKI(R=R, nc=dim[0])
        weight_path = os.path.join(complete_path, f"{prefix}coil_{coil}.pth")
        RAKI_model.load_state_dict(torch.load(weight_path, map_location=device))
        RAKI_model = RAKI_model.to(device)
        RAKI_model.eval()
        with torch.no_grad():
            # Infer missing lines
            predictions = RAKI_model(input_tensor)
            predictions = predictions.cpu().clone().detach()
            predictions = predictions.view((1,
                                            target_dim_Z,
                                            target_dim_X,
                                            target_dim_Y))
            target_x_end_kspace = dim[1] - target_x_start
            for t in range(0, R - 1):

                target_y_start = (
                    np.int32(
                        (np.ceil(kernel_y_1 / 2) - 1)
                        + np.int32((np.ceil(kernel_y_2 / 2) - 1))
                        + np.int32(np.ceil(kernel_y_3 / 2) - 1)
                    )
                    * R
                    + t
                    + 1
                )
                target_y_end_kspace = (
                    dim[2]
                    - np.int32(
                        (np.floor(kernel_y_1 / 2))
                        + (np.floor(kernel_y_2 / 2))
                        + np.floor(kernel_y_3 / 2)
                    )
                    * R
                    + t
                )
                recon[
                    coil,
                    target_x_start: target_x_end_kspace,
                    target_y_start: target_y_end_kspace + 1: R,
                ] = predictions[0, t, :, ::R]

    # IF ACS_flag is True, then we put back the calibration lines.
    if ACS_flag:
        start = dim[2]//2 - number_columns//2
        end = dim[2]//2 + number_columns//2
        recon[:, :, start: end] = ACS

    recon = recon / nor
    return recon


def raki_recon_algo(
        kspace, acc_rate, number_columns,
        ACS_flag, device,
        save_path, slice, filename="Brain_rand",
        lr=3e-3, epochs=3000):
    save_dir = save_path
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    txt = filename + f"_Slice_{slice}_R_{acc_rate}_ACS_{number_columns}_ACS_flag_{ACS_flag}_epochs_{epochs}_lr_{lr}_"
    prefix = txt
    filepath = os.path.join(save_dir, f"{prefix}coil_{0}.pth") if save_dir is not None else None

    if save_dir is not None and os.path.exists(filepath):
        print(f"Loading RAKI weights from {save_dir}")
        recon = infer_raki_NEW(
                kspace, save_dir, slice,
                device, acc_rate, number_columns,
                ACS_flag, epochs, lr, prefix=prefix)
    else:
        print(f"Training model and saving the weights at {save_dir}")
        recon = RAKI_run_save(
            kspace, acc_rate, number_columns,
            ACS_flag, device,
            save_dir, slice=slice,
            lr=lr, epochs=epochs, prefix=prefix)

    channels = kspace.shape[0]
    recon = recon[:channels, :, :] + 1j * recon[channels:, :, :]

    return recon