
import numpy as np

from ..helpers.preprocessing.preprocessing import uniform_restrospective_undersampling as undersample

def pad_size(kernel, R):
    """
    inputs:R       -   [Rx, Ry] undersampling factors

            kernel  -   [sx, sy] kernel size in each dimension
    output:
            pad     -   [pad_x, pad_y] size of padding in each direction
    """

    #   Compute size of padding needed in each direction
    pad = np.floor(np.multiply(R, kernel) / 2).astype(int)
    return pad


def pad_data(data, pad):
    """ "
    inputs:
            data    -   (c, kx, ky) complex k-space data
            pad     -   [pad_x, pad_y] size of padding in each direction

    output:
            pdata   -   (c, kx+pad, ky+pad) complex padded k-space data
    """

    #   Zero-Pad
    #   Apply zero-padding to kx, ky directions (don't pad coil dimension)
    padded_tensor = np.pad(
        data,
        ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])),
        mode="constant",
        constant_values=0,
    )

    return padded_tensor


def unpad_data(pdata, pad):
    """
    Remove padding from padded k-space data.

    Parameters:
    - pdata: complex padded k-space data (c, kx+pad_x, ky+pad_y)
    - pad: size of padding in each direction [pad_x, pad_y]

    Returns:
    - data: complex k-space data (c, kx, ky)
    """
    # Subselect inner data absent the padded points
    data = pdata[:, pad[0]: pdata.shape[1] - pad[0],
                 pad[1]: pdata.shape[2] - pad[1]]

    return data


def src_trg(calib, kernel, nx, ny, t, R):
    """
    Inputs:
        calib   -   (c, cx, cy) complex calibration k-space data
        kernel  -   [kx, ky] kernel size (kx odd, ky even)
        nx      -   (cx')   vector of possible read out positions
        ny      -   (cy')   vector of possible phase encoding positions
        t       -   int, position of the line to be reconstructed

    output:
        src_m   -   (kernelx*kernely, cx'*cy')
                    Matrix made of sources (or neighbors)
        trg_m   -   (c, cx'*cy')     Matrix made of targets
    """
    trg = []
    src = []
    # Calibration for a specific position
    for i in nx:
        for j in ny:
            start = j - t
            end = j - t + R[1]
            num = kernel[1]//2-1
            y_list = (
                [start - R[1]*s for s in range(1, num + kernel[1] % 2+1)]
                + [start, end]
                + [end + R[1]*s for s in range(1, num+1)])
            trg.append(calib[:, i, j].reshape(-1, 1))
            neighbors = calib[
                :,
                i - kernel[0] // 2: i + kernel[0] // 2 + 1,
                y_list
            ].reshape(-1, 1)

            # print(calib[
            #     0,
            #     i - kernel[0] // 2: i + kernel[0] // 2 + 1,
            #     y_list
            # ])

            src.append(neighbors.reshape(-1, 1))

    # Get indices for a specific position t
    trg_m = np.concatenate(trg, axis=1)
    src_m = np.concatenate(src, axis=1)
    return trg_m, src_m


def estimate_weights(trg_m, src_m):
    weights = trg_m @ np.linalg.pinv(src_m)
    return weights


def src(pUdata, dimT, dimU, R, t, pad, kernel):

    mask = np.zeros(dimT)
    mask[:, :, t:: R[1]] = True
    condition = mask == True
    coils, rows, cols = np.where(condition)

    coils, rows, cols = np.unique(coils), np.unique(rows), np.unique(cols)
    src_truth = []

    # GET SOURCES

    kx = np.arange(start=pad[0], stop=dimU[1] - pad[0], dtype="int")

    for i in kx:
        for j in cols:
            start = j - t + pad[1]
            end = j - t + R[1] + pad[1]
            num = kernel[1]//2-1
            y_list = (
                [start - R[1]*s for s in range(1, num + kernel[1] % 2+1)]
                + [start, end]
                + [end + R[1]*s for s in range(1, num+1)])

            neighbors = pUdata[
                :,
                i - kernel[0] // 2: i + kernel[0] // 2 + 1,
                y_list
            ].reshape(-1, 1)

            src_truth.append(neighbors.reshape(-1, 1))

    src_m_truth = np.concatenate(src_truth, axis=1)

    del neighbors
    del mask
    del src_truth

    return src_m_truth, coils, rows, cols


def apply_weights(weights, src_m_truth):
    return np.dot(weights, src_m_truth)


def grappa(Udata, calib, kernel, R, ACS_flag=True):
    """
    Inputs:
          Udata    -   (c, nx, ny) complex undersampled k-space data
          calib   -   (c, cx, cy) complex calibration k-space data
          R       -   [Rx, Ry] 1x2 array or just Ry scalar
          kernel  -   [kx, ky] kernel size (kx odd, ky even)

    output:
          recon   -   (c, nx, ny) complex reconstructed k-space data
    """
    print(30*"*" + " Grappa " + 30*"*")
    recon = Udata.copy()
    dimT = Udata.shape

    # Padding
    pad = pad_size(kernel, R)
    pUdata = pad_data(Udata, pad)
    dimU = pUdata.shape

    dimC = calib.shape
    number_columns = dimC[-1]
    padC = pad_size(kernel, R)

    nx = np.arange(start=padC[0], stop=dimC[1] - padC[0], dtype="int")
    ny = np.arange(start=padC[1], stop=dimC[2] - padC[1], dtype="int")

    # weight dictionnary

    weights_dict = {}

    for t in range(1, R[1]):

        # Get the targets and sources
        trg_m, src_m = src_trg(calib, kernel, nx, ny, t, R)

        # Estimate weights
        weights = estimate_weights(trg_m, src_m)
        weights_dict[f'line {t-1}'] = weights
        src_m_truth, coils, rows, cols = src(pUdata,
                                             dimT,
                                             dimU,
                                             R, t, pad, kernel)
        estimated_trg = apply_weights(weights, src_m_truth)
        recon[np.ix_(coils, rows, cols)] = estimated_trg.reshape(
            dimT[0], dimT[1], -1)


    if ACS_flag:
            recon[
                :,
                :,
                dimT[2]//2 - number_columns//2: dimT[2]//2 + number_columns//2
                ] = calib
    return recon, weights_dict


def retrospective_grappa(kspace, r, number_columns, kernel, ACS_bool=True):
    """
    Get fully-sampled kspace data, undersampled it and reconstruct
    """
    print(30*"*" + " Grappa " + 30*"*")
    dim = kspace.shape
    Udata, ACS = undersample(
        kspace=kspace.copy(), r=r, number_columns=number_columns
        )

    grappa_recon = grappa(
        Udata=Udata, calib=ACS, kernel=kernel, R=[1, r]
        )

    if ACS_bool:
        grappa_recon[
            :,
            :,
            dim[2]//2 - number_columns//2: dim[2]//2 + number_columns//2
            ] = ACS

    return grappa_recon
