import numpy as np

def normalize(kspace, outerbound=0.015):
    nor = outerbound / np.max(abs(kspace[:]))
    return np.multiply(kspace, nor)

def kspace_to_image(kspace):
    """Converts k-space data to image space."""
    img = np.squeeze(
        np.sqrt(
            np.sum(
                np.abs(
                    np.fft.ifftshift(
                        np.fft.ifftn(kspace), axes=(1, 2))) ** 2, axis=0
            )
        )
    )
    return img

def uniform_restrospective_undersampling(kspace, r, number_columns):
    dim = kspace.shape
    mask = np.zeros(dim, dtype=kspace.dtype)

    mask[:, :, ::r] = True
    Udata = kspace * mask

    start = dim[2] // 2 - number_columns // 2
    end = dim[2] // 2 + number_columns // 2

    # Get the ACS lines and put it back in the undersampled data
    ACS = kspace[:, :, start:end]
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
        raise ValueError(
            "Target dimensions must be smaller than or equal to the original dimensions."
        )

    cropped_kspace = np.zeros((num_coils, target_height, target_width), dtype=np.complex64)

    readout_start = (original_readout - target_height) // 2
    readout_end = readout_start + target_height

    phase_encoding_start = (original_phase_encoding - target_width) // 2
    phase_encoding_end = phase_encoding_start + target_width

    for coil in range(num_coils):
        image_space = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace[coil])))
        cropped_image = image_space[
            readout_start:readout_end, phase_encoding_start:phase_encoding_end
        ]
        cropped_kspace[coil] = np.fft.ifftshift(
            np.fft.fft2(np.fft.fftshift(cropped_image))
        )

    return cropped_kspace

