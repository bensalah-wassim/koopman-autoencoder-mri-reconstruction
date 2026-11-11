# External modules
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
# Libraires
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

import torch
import torch.nn as nn
import torch.nn.init as init



# Modules
from .network import RAKI
from .training import train_model


def kspace_to_image(kspace):
    """Converts k-space data to image space."""
    img = np.squeeze(
        np.sqrt(
            np.sum(np.abs(np.fft.ifftshift(np.fft.ifftn(kspace), axes=(1, 2)
                                           )) ** 2, axis=0)))
    return img


def show_quad(data_list, text_list, x=200, y=108, box_size=30, coil=0, txt=""):
    """
    Generalized function to display k-space, images, and zoomed-in region for multiple sets of data.
    
    Parameters:
    - data_list: List of data arrays [X, Y, Z, ...]
    - text_list: List of text labels corresponding to each data set
    - x, y: Coordinates in the image to center the zoomed-in box
    - box_size: Half-length of the box around (x, y) to zoom into
    - coil: Coil axis to move (default is 0)
    
    Notes:
    K-space data needs to have the coils dimension as the first dimension,
    followed by read-out and phase-encoding dimensions.
    """
    
    # Ensure the input lists are consistent in length
    if len(data_list) != len(text_list):
        raise ValueError("Length of data_list and text_list must be the same.")
    
    num_data = len(data_list)  # Number of datasets (X, Y, Z, etc.)
    
    if coil != 0:
        data_list = [np.moveaxis(data, [coil, 0], [0, coil]) for data in data_list]
    
    fig, axs = plt.subplots(3, num_data, figsize=(5 * num_data, 15), constrained_layout=True)
    
    for i, data in enumerate(data_list):
        # K-space visualization
        axs[0, i].imshow(np.squeeze(np.log10(np.abs(data[0, :, :]) + 1e-15)), cmap="gray")
        axs[0, i].set_title(f"{text_list[i]} k-space")

        # Image reconstruction
        reconstructed_image = np.squeeze(
            np.sqrt(np.sum(np.abs(np.fft.ifftshift(np.fft.ifftn(data), axes=(1, 2))) ** 2, axis=0))
        )
        
        axs[1, i].imshow(reconstructed_image, cmap="gray")
        axs[1, i].set_title(f"{text_list[i]} Image")
        
        # Add a box to the image at (x, y)
        rect = plt.Rectangle((y - box_size, x - box_size), 2 * box_size, 2 * box_size, 
                             linewidth=1, edgecolor='r', facecolor='none')
        axs[1, i].add_patch(rect)
        
        # Zoomed-in area
        zoomed_image = reconstructed_image[
            max(0, x - box_size):min(reconstructed_image.shape[0], x + box_size),
            max(0, y - box_size):min(reconstructed_image.shape[1], y + box_size)
        ]
        axs[2, i].imshow(zoomed_image, cmap="gray")
        axs[2, i].set_title(f"Zoomed-in {text_list[i]} Image")
    if txt == "":
        fig.suptitle("Comparison of k-space, reconstructed images, and zoomed-in areas", fontsize=16)
    else:
        fig.suptitle(txt, fontsize=16)
    plt.show()


def normalize(kspace, outerbound=0.015):

    nor = outerbound / np.max(abs(kspace[:]))
    return np.multiply(kspace, nor)


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


def crop_in_image_space_multi_coil(kspace, target_height, target_width):
    """
    Crop an image after transforming from k-space to image space, 
    and then return the cropped image back to k-space.

    Parameters:
    kspace (numpy.ndarray): The input k-space data (3D complex array: coils, readout, phase encoding).
    target_height (int): The desired height of the cropped image.
    target_width (int): The desired width of the cropped image.

    Returns:
    numpy.ndarray: The cropped k-space data.
    """
    num_coils, original_readout, original_phase_encoding = kspace.shape

    if target_height > original_readout or target_width > original_phase_encoding:
        raise ValueError("Target dimensions must be smaller than or equal to the original dimensions.")

    cropped_kspace = np.zeros((num_coils, target_height, target_width), dtype=complex)

    readout_start = (original_readout - target_height) // 2
    readout_end = readout_start + target_height

    phase_encoding_start = (original_phase_encoding - target_width) // 2
    phase_encoding_end = phase_encoding_start + target_width

    for coil in range(num_coils):
        image_space = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace[coil])))
        cropped_image = image_space[readout_start:readout_end, phase_encoding_start:phase_encoding_end]
        cropped_kspace[coil] = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cropped_image)))

    return cropped_kspace


def run_save_plot(
        kspace, acc_rate, number_columns,
        ACS_flag, lr, epochs, device,
        draw_curve, save_model, kspace_name, save_path):

    dim = kspace.shape
    txt = f"{kspace_name}_R_{acc_rate}_ACS_{number_columns}_ACS_flag_{ACS_flag}_epochs_{epochs}_lr_{lr}_"
    torch.cuda.empty_cache()
    print("Acceleration rate :", acc_rate)

    # Undersampled data
    Udata, ACS = undersample(
        kspace=kspace, r=acc_rate, number_columns=number_columns
        )

    # Load model

    # RAKI
    print(30*"*" + " RAKI " + 30*"*")

    RAKI_model = RAKI(R=acc_rate, nc=dim[0]//2).to(device)

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
        save_path=save_path+txt
    )
    # IF ACS_flag is True, then we put back the calibration lines.
    if ACS_flag:
        start = dim[2]//2 - number_columns//2
        end = dim[2]//2 + number_columns//2
        raki_recon[:, :, start: end] = ACS

    # Save
    if save_model:
        np.save(save_path+txt+"_kspace.npz",
                arr=raki_recon, allow_pickle=False)

    # Plot
    if draw_curve:

        raki_recon = (raki_recon[: dim[0] // 2, :, :]
                      + 1j * raki_recon[dim[0] // 2:, :, :])
        truth = (kspace[: dim[0] // 2, :, :]
                 + 1j * kspace[dim[0] // 2:, :, :])

        show_quad(
            X=truth,
            Y=raki_recon,
            text=txt,
            coil=0,
            textX="Ground truth",
            textY="Reconstructed",
        )

    del raki_recon
    del RAKI_model
    del Udata
    del ACS


def compute_metric(metric, gt_img, image):
    # Compute the corresponding metric
    if metric == "ssim":
        metric_value = ssim(gt_img, image,
                            data_range=gt_img.max()-gt_img.min())
    elif metric == "mse":
        metric_value = mse(gt_img, image)
    else:
        metric_value = nrmse(gt_img, image)
    return metric_value


def print_metrics(gt,  recons, titles):
    for i, recon in enumerate(recons):
        img = kspace_to_image(recon)
        gt_img = kspace_to_image(gt)
        metrics = {
            "ssim": [], "mse": [], "nrmse": []
        }

        for metric in ["ssim", "mse", "nrmse"]:
            metric_value = compute_metric(metric, gt_img, img)
            metrics[metric] = metric_value
        print(titles[i])
        print(
            f"Metric: ssim ={metrics['ssim']}, "
            f"mse ={metrics['mse']}, "
            f"nrmse ={metrics['nrmse']}")


def infer_raki(
        kspace, path,
        device, R, number_columns,
        ACS_flag, epochs_path, lr_path, kspace_name):
    dim = kspace.shape
    kernel = [5, 4]
    # Undersampled data
    Udata, ACS = undersample(
        kspace=kspace, r=R, number_columns=number_columns
        )
    txt = f"{kspace_name}_R_{R}_ACS_{number_columns}_ACS_flag_{ACS_flag}_epochs_{epochs_path}_lr_{lr_path}_"
    complete_path = path + txt

    dim = Udata.shape
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
    for coil in range(dim[0]):

        # Convert to PyTorch tensor and move to device
        input_tensor = torch.tensor(Udata, dtype=torch.float).to(device)
        RAKI_model = RAKI(R=R, nc=dim[0]//2)
        RAKI_model.load_state_dict(
            torch.load(complete_path+f"coil_{coil}.pth"))
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

    recon = (recon[: dim[0] // 2, :, :]
             + 1j * recon[dim[0] // 2:, :, :])
    kspace_complex = (kspace[: dim[0] // 2, :, :]
                      + 1j * kspace[dim[0] // 2:, :, :])

    # Grappa
    print(30*"*" + " Grappa " + 30*"*")
    # Grappa needs complexe kspaces
    Udata_complex = Udata[:dim[0]//2, :, :] + 1j * Udata[dim[0]//2:, :, :]
    ACS_complex = ACS[:dim[0]//2, :, :] + 1j * ACS[dim[0]//2:, :, :]
    grappa_recon = grappa(
        Udata=Udata_complex, calib=ACS_complex, kernel=[5, 4], R=[1, acc_rate]
        )

    # AE
    # n1 = (kernel[0]*kernel[1]+acc_rate-1)*dim[0]
    # n2 = int(n1*0.8)
    # n3 = int(n1*0.4)
    # model = AE(
    #     n1=n1, n2=n2, n3=n3)

    lr = 1e-4
    epochs = 1000
    ae_recon = run_ae(
        ModifiedAutoencoder, kspace,
        lr, epochs, acc_rate, kernel,
        number_columns, device,
        pretrain=True, static_weights=False)

    show_quad(
        data_list=[kspace_complex, grappa_recon, recon, ae_recon],
        text_list=["Ground Truth", "GRAPPA", "RAKI", "AE"],
        txt=txt)

    print_metrics(
        gt=kspace_complex,
        recons=[grappa_recon, recon, ae_recon],
        titles=["GRAPPA", "RAKI", "AE"])


if __name__ == "__main__":
    # Parameters

    epochs = 3000
    lr = 3e-3
    draw_curve = True
    early_stop = False
    ACS_flag = False
    save_model = False
    r_range = [5] #np.arange(2, 7, dtype="int")
    number_columns = [10*r for r in r_range]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Load the MATLAB .mat file
    # with h5py.File(
    #     "C:/Users/ws24/Documents/Wassim_GIT/data/data.mat", "r"
    # ) as file:
    #     # Access a dataset in the file
    #     calib = file["calib"][:]
    #     truth = file["truth"][:]
    #     # normalisation
    #     tt = normalize(truth["real"] + 1j * truth["imag"], outerbound=0.015)
    #     tt = np.moveaxis(tt, [2, 0], [0, 2])

    #     truth = np.concatenate((np.real(tt), np.imag(tt)), axis=0)

    # print(truth.shape, "Ground Truth dimension\n",
    #       calib.shape, "Calib dimension")

    # kspace_name = "data_br"
    # save_path = "D:/results/raki/single_data/"

    # for i, acc_rate in enumerate(r_range):
    #     kspace = truth.copy()
    #     run_save_plot(
    #         kspace=kspace, acc_rate=acc_rate, number_columns=number_columns[i],
    #         ACS_flag=ACS_flag, lr=lr, epochs=epochs, device=device,
    #         draw_curve=draw_curve, save_model=save_model,
    #         kspace_name=kspace_name, save_path=save_path)
    #     recon = infer_raki(
    #         kspace=kspace, path=save_path,
    #         device=device, R=acc_rate, number_columns=number_columns[i],
    #         ACS_flag=ACS_flag, epochs=epochs, lr=lr, kspace_name=kspace_name)

    # Directory containing the .h5 files
    directory = 'D:/fastmri/multicoil_train/'
    for filename in os.listdir(directory)[5:6]:
        if filename.endswith('.h5'):
            filepath = os.path.join(directory, filename)
            with h5py.File(filepath, 'r') as file:
                print(f"Processing file: {filename[:-3]}")
                print("Keys: %s" % list(file.keys()))

                # Get the k-space data
                data = file['kspace'][:]
            # Shape should be
            # (num_slices, num_coils, readout, phase_encoding)
            print(f"Data shape: {data.shape}")
            # Limiting to the first 6 slices for display
            for i in range(min(1, data.shape[0])):

                original_kspace = data[i]
                org_shape = original_kspace.shape
                cropped_kspace = crop_in_image_space_multi_coil(
                    original_kspace,
                    target_height=org_shape[2],
                    target_width=org_shape[2]
                    )[:, :, 49:-49]
                truth = cropped_kspace.copy()
                truth = normalize(truth, outerbound=1)
                truth = np.concatenate(
                    (np.real(truth), np.imag(truth)), axis=0)

                kspace = truth.copy()

                # Raki
                kspace_name = filename[:-3]+f"_slice_{i}"
                save_path = "D:/results/raki/fastmri/"
                path = save_path

                for i, acc_rate in enumerate(r_range):
                    kspace = truth.copy()
                    number_columns = 10*acc_rate
                    infer_raki(
                        kspace, path,
                        device, acc_rate, number_columns,
                        ACS_flag, epochs, lr, kspace_name)
