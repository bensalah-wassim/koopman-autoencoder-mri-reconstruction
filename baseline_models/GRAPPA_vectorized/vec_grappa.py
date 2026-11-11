"""

MRI reconstruction pipeline with a vectorized GRAPPA

"""

# ==================================================
#           IMPORTS & PATH CONFIGURATION           #
# ==================================================

import numpy as np



# ==================================================
#     VECTORISED DATA PREPARATION  (GPU-READY)     #
# ==================================================

def undersample(kspace, r, num_acs):
    """Uniform Ry=r undersampling with central ACS region."""
    Ry  = _ry(r)
    res = np.zeros_like(kspace)
    res[:, :, ::Ry] = kspace[:, :, ::Ry]
    mid = kspace.shape[2] // 2
    hw  = num_acs // 2
    acs = kspace[:, :, mid - hw: mid + hw]
    res[:, :, mid - hw: mid + hw] = acs
    return res, acs


def _ry(R):
    """
    Extract scalar Ry from R.

    R can be:
      - int / float          → used directly  (KAE pipeline convention)
      - list / array [Rx,Ry] → Ry = R[1]      (GRAPPA pipeline convention)

    This is the single fix point for the TypeError:
        unsupported operand type(s) for -: 'int' and 'list'
    """
    if isinstance(R, (list, np.ndarray)):
        return int(R[1])
    return int(R)


def _src_y_offsets(kernel, R, t):
    """
    Relative y-index offsets for the GRAPPA/KAE kernel.

    For kernel=[kx, ky], acceleration Ry:
        neighbours left  : [y-1 - Ry·s  for s in 1..num+ky%2]
        immediate pair   : [y-1,  y-1+Ry]
        neighbours right : [y-1+Ry + Ry·s  for s in 1..num]

    Offsets are expressed relative to y (target position), so they
    can be broadcast over any array of target positions.

    Returns: np.ndarray of shape (ky_pts,) = (kernel[1],)
    Wassim you need to introduce t in the comments above as it was missing in the first implementation
    """
    Ry  = _ry(R)                    # ← always scalar, regardless of R type
    num = kernel[1] // 2 - 1
    return np.array(
        [-t - Ry * s for s in range(1, num + kernel[1] % 2 + 1)]
        + [-t, -t + Ry]
        + [-t + Ry + Ry * s for s in range(1, num + 1)],
        dtype=int,
    )   # shape: (kernel[1],)


def generate_data_vec(kspace: np.ndarray, kernel: list, R):
    """
    Vectorised full-batch training data generation from ACS.

    Replaces the nested Python loops in the original generate_data() with
    numpy advanced indexing — no Python loop overhead.

    Training regime
    ---------------
    The returned (X, Y) arrays are passed ENTIRELY to the Trainer in one
    shot: full-batch gradient descent, not mini-batch SGD.
    This is appropriate because:
      - The ACS is small (typically 24 × ~300 positions ≈ 7200 patches).
      - Full-batch has stable, deterministic gradient → faster convergence
        for this problem size.
      - No data loader / shuffling overhead.

    Layout
    ------
    X[n, c, :n_src]  = acquired neighbour patch   (source, known)
    X[n, c, n_src:]  = zeros                       (missing slots, unknown)
    Y[n, c, :n_src]  = same neighbour patch        (target, src consistency)
    Y[n, c, n_src:]  = actual missing line values  (target, reconstruction)

    Returns
    -------
    X, Y : (N, c, n_feat)  complex64,   N = len(kx) × len(ky)
    """
    Ry    = _ry(R)
    c, nx, ny = kspace.shape
    padx  = kernel[0] // 2
    num   = kernel[1] // 2 - 1
    l_pady = 1 + (num + kernel[1] % 2) * Ry
    r_pady = Ry - 1 + num * Ry
    n_src  = kernel[0] * kernel[1]
    n_miss = Ry - 1

    kx = np.arange(padx,   nx - padx,   dtype=int)
    ky = np.arange(l_pady, ny - r_pady, dtype=int)
    N  = len(kx) * len(ky)

    src_y_off  = _src_y_offsets_kae(kernel, Ry)           # (kernel[1],)
    x_off      = np.arange(-padx, padx + 1, dtype=int)  # (ker_x,)
    miss_y_off = np.arange(0, n_miss, dtype=int)        # (Ry-1,)

    # Absolute index arrays
    src_x_abs  = kx[:, None] + x_off[None, :]         # (n_kx, ker_x)
    src_y_abs  = ky[:, None] + src_y_off[None, :]     # (n_ky, kernel[1])
    miss_y_abs = ky[:, None] + miss_y_off[None, :]    # (n_ky, Ry-1)

    # Source patches — shape (c, n_kx, n_ky, ker_x, kernel[1])
    xi = src_x_abs[:, np.newaxis, :, np.newaxis]       # (n_kx, 1, ker_x, 1)
    yi = src_y_abs[np.newaxis,  :, np.newaxis, :]      # (1, n_ky, 1, kernel[1])
    src_patches = kspace[:, xi, yi]
    src_patches = src_patches.transpose(1, 2, 0, 3, 4).reshape(N, c, n_src)

    # Missing values — shape (c, n_kx, n_ky, Ry-1)
    xi_m    = kx[:, np.newaxis, np.newaxis]
    yi_m    = miss_y_abs[np.newaxis, :, :]
    miss_vals = kspace[:, xi_m, yi_m]
    miss_vals = miss_vals.transpose(1, 2, 0, 3).reshape(N, c, n_miss)

    X = np.zeros((N, c, n_src + n_miss), dtype=kspace.dtype)
    Y = np.zeros((N, c, n_src + n_miss), dtype=kspace.dtype)
    X[:, :, :n_src] = src_patches
    Y[:, :, :n_src] = src_patches
    Y[:, :, n_src:] = miss_vals
    return X, Y


# ==================================================
#                 GRAPPA BASELINE                  #
# ==================================================

def _pad_size(kernel, R):
    """R can be scalar or [Rx, Ry] — always returns [pad_x, pad_y]."""
    Ry = _ry(R)
    return np.floor(np.multiply([1, Ry], kernel) / 2).astype(int)

def _pad_data(data, pad):
    return np.pad(data, ((0,0),(pad[0],pad[0]),(pad[1],pad[1])),
                  mode='constant', constant_values=0)

def _src_trg_grappa(calib, kernel, nx, ny, t, R):
    """Vectorised calibration patch extraction for GRAPPA."""
    c       = calib.shape[0]
    kx_h    = kernel[0] // 2
    offsets = _src_y_offsets(kernel, R, t)      # handles list R internally

    xi = (nx[:, None] + np.arange(-kx_h, kx_h+1)[None, :])[:, np.newaxis, :, np.newaxis]
    yi = (ny[:, None] + offsets[None, :])[np.newaxis, :, np.newaxis, :]
    patches = calib[:, xi, yi].transpose(1, 2, 0, 3, 4)
    src_m   = patches.reshape(len(nx) * len(ny), -1).T
    xi_t, yi_t = nx[:, np.newaxis], ny[np.newaxis, :]
    trg_m   = calib[:, xi_t, yi_t].reshape(c, -1)
    return trg_m, src_m

def _src_grappa(pdata, dimT, R, t, pad, kernel):
    """Vectorised application patch extraction for GRAPPA."""
    c, nx, ny = dimT
    Ry        = _ry(R)
    offsets   = _src_y_offsets(kernel, R, t)
    kx_h      = kernel[0] // 2
    cols      = np.arange(t, ny, Ry)
    rows_pad  = np.arange(nx) + pad[0]
    x_abs     = rows_pad[:, None] + np.arange(-kx_h, kx_h+1)[None, :]
    y_abs     = (cols + pad[1])[:, None] + offsets[None, :]
    xi        = x_abs[:, np.newaxis, :, np.newaxis]
    yi        = y_abs[np.newaxis,   :, np.newaxis, :]
    src_m     = pdata[:, xi, yi].transpose(1, 2, 0, 3, 4).reshape(nx * len(cols), -1).T
    return src_m, cols

def grappa(Udata, calib, kernel, R, debug=False):
    """
    GRAPPA reconstruction (vectorised).

    R is expected as [1, Ry] following your convention
    (Rx=1 means no undersampling in readout direction).
    Both scalar R and list [Rx, Ry] are handled via _ry().
    """
    recon     = Udata.copy().astype(complex)
    c, nx, ny = Udata.shape
    Ry        = _ry(R)
    pad       = _pad_size(kernel, R)
    pdata     = _pad_data(Udata, pad)
    padC      = _pad_size(kernel, R)
    nx_c      = np.arange(padC[0], calib.shape[1] - padC[0])
    ny_c      = np.arange(padC[1], calib.shape[2] - padC[1])

    for t in range(1, Ry):
        trg_m, src_m  = _src_trg_grappa(calib, kernel, nx_c, ny_c, t, R)
        W             = trg_m @ np.linalg.pinv(src_m)
        src_app, cols = _src_grappa(pdata, (c, nx, ny), R, t, pad, kernel)
        estimated     = W @ src_app                              # (c, nx*n_cols)
        recon[:, :, cols] = estimated.reshape(c, nx, len(cols))

    return recon

def retrospective_grappa(kspace, r, number_columns, kernel, ACS_bool=True):
    """
    Retrospective GRAPPA.
    r is the phase-encoding acceleration factor (Ry).
    Passed to grappa() as R=[1, r] following your convention.
    """
    dim        = kspace.shape
    Udata, ACS = undersample(kspace.copy(), r, number_columns)
    recon      = grappa(Udata=Udata, calib=ACS, kernel=kernel, R=[1, r])
    if ACS_bool:
        mid = dim[2] // 2
        hw  = number_columns // 2
        recon[:, :, mid - hw: mid + hw] = ACS
    return recon

