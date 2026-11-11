import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


def kspace_to_image(kspace):
    """Converts k-space data to image space."""
    img = np.squeeze(
        np.sqrt(
            np.sum(np.abs(np.fft.ifftshift(np.fft.ifftn(kspace), axes=(1, 2)
                                           )) ** 2, axis=0)))
    return img

def plot_kspace_images(
    cropped_kspace,
    plot_type="image",
    title="K-space Visualization",
    cols=5,
    filename=None,
    slice_index=None,
    save_dir=None
):
    """
    Plot either reconstructed images or k-space magnitude for each coil in a grid layout.

    Parameters:
    - cropped_kspace: numpy.ndarray, shape (coils, height, width)
    - plot_type: str, "image" or "kspace"
    - title: str, base title of the plot
    - cols: int, number of columns in the subplot grid
    - filename: str, optional, filename for title and saving
    - slice_index: int, optional, slice index for title and saving
    - save_dir: str or None, if provided, the plot is saved as PNG
    """
    num_coils = cropped_kspace.shape[0]
    rows = math.ceil(num_coils / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.75), constrained_layout=True)
    axs = axs.flatten()

    for coil in range(num_coils):
        data = cropped_kspace[coil]

        if plot_type == "image":
            img = np.squeeze(
                np.abs(np.fft.ifftshift(np.fft.ifftn(data), axes=(0, 1)))
            )
            axs[coil].imshow(img, cmap="gray")
            axs[coil].set_title(f"Coil {coil} Image")
        elif plot_type == "kspace":
            kspace_log = np.squeeze(np.log10(np.abs(data) + 1e-15))
            axs[coil].imshow(kspace_log, cmap="gray")
            axs[coil].set_title(f"Coil {coil} K-space")
        else:
            raise ValueError("plot_type must be either 'image' or 'kspace'")

        axs[coil].axis("off")  # nettoyage des axes

    # Masquer les subplots non utilisés
    for i in range(num_coils, len(axs)):
        axs[i].axis("off")

    # Compose le titre
    full_title = title
    if filename is not None:
        full_title += f"\nFile: {filename}"
    if slice_index is not None:
        full_title += f" | Slice: {slice_index}"

    fig.suptitle(full_title, fontsize=16)

    # Sauvegarde si save_dir fourni
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_safe = (filename.replace('.h5','') if filename else 'kspace')
        slice_str = f"_slice{slice_index}" if slice_index is not None else ""
        save_path = os.path.join(save_dir, f"{file_safe}_{plot_type}{slice_str}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_slices_comparison(
    recon_dict,
    filename,
    line_key=None,
    slice_indices=None,
    plot_type="recon",      # "recon" for GRAPPA recon, "abs_diff" for |GT - Recon|
    global_min=None,
    global_max=None,
    cols=5,
    save_dir=None
):
    """
    Plot multiple slices of GRAPPA reconstruction or absolute differences vs. ground truth.

    Parameters
    ----------
    recon_dict : dict
        Keys are (filename, slice_index), values are dict with 'ground_truth' and 'grappa_recon'
    filename : str
        Name of the file to plot slices for
    line_key : str or None
        Optional line key if plotting per GRAPPA line
    slice_indices : list[int] or None
        Which slices to plot. If None, plot all slices in recon_dict for this filename.
    plot_type : str
        "recon" for GRAPPA reconstruction, "abs_diff" for absolute difference |GT - Recon|
    global_min : float or None
        Minimum value for colormap (optional)
    global_max : float or None
        Maximum value for colormap (optional)
    cols : int
        Number of columns in the subplot grid
    save_dir : str or None
        Directory to save the figure, if None will just show
    """
    
    # Select slices for this filename
    selected_slices = [key for key in recon_dict.keys() if key[0] == filename]
    if slice_indices is not None:
        selected_slices = [(fn, idx) for fn, idx in selected_slices if idx in slice_indices]
    
    num_slices = len(selected_slices)
    rows = math.ceil(num_slices / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), constrained_layout=True)
    axs = axs.flatten()
    
    # Collect images for global min/max calculation
    all_imgs = []
    for i, (fn, sl_idx) in enumerate(selected_slices):
        data_dict = recon_dict[(fn, sl_idx)]

        
        gt_img = np.abs(np.fft.ifftshift(np.fft.ifftn(data_dict["ground_truth"], axes=(0,1))))
        recon_img = np.abs(np.fft.ifftshift(np.fft.ifftn(data_dict["grappa_recon"], axes=(0,1))))
        
        if plot_type == "recon":
            img = recon_img
        elif plot_type == "abs_diff":
            img = np.abs(gt_img - recon_img)
        else:
            raise ValueError("plot_type must be 'recon' or 'abs_diff'")
        
        all_imgs.append(img)
    
    # Determine global window if not given
    if global_min is None:
        global_min = min(img.min() for img in all_imgs)
    if global_max is None:
        global_max = max(img.max() for img in all_imgs)
    
    # Plot each slice
    for i, img in enumerate(all_imgs):
        axs[i].imshow(img, cmap="gray", vmin=global_min, vmax=global_max)
        axs[i].set_title(f"Slice {selected_slices[i][1]}")
        axs[i].axis("off")
    
    # Hide unused axes
    for j in range(num_slices, len(axs)):
        axs[j].axis("off")
    
    full_title = f"{filename} - {plot_type.replace('_', ' ').title()}"
    if line_key is not None:
        full_title += f" | {line_key}"
    fig.suptitle(full_title, fontsize=16)
    
    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_safe = filename.replace(".h5","")
        line_str = f"_{line_key}" if line_key else ""
        save_path = os.path.join(save_dir, f"{file_safe}{line_str}_{plot_type}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# 16/02/2026 - plots

def show_single(original, recon, title, coil=0, save_bool=None, folder=None, filename_str=None, window_perc = 99.5):
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
    original_image = kspace_to_image(original)
    vmin = 0
    vmax = np.percentile(original_image, window_perc)
    reconstructed_image = kspace_to_image(recon)

    # Create a figure for the reconstructed image
    plt.figure(figsize=(10, 10), constrained_layout=True)

    plt.imshow(reconstructed_image, cmap="gray",  vmin=vmin, vmax=vmax)
    plt.title(f"{title} Image")

    if save_bool:
        filepath = os.path.join(folder, filename_str)
        plt.savefig(filepath)
    else:
        plt.show()


def plot_loss_curve(losses, title="Training Loss", save_path=None, filename="loss"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(losses, lw=1.2, color='steelblue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.4)
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
    n = len(images)
    if n <= 1:
        return

    gt_img = np.abs(images[0])
    methods_imgs = [np.abs(im) for im in images[1:]]
    plot_images = methods_imgs + [gt_img]
    plot_names = names[1:] + [names[0]]
    ncols = len(plot_images)

    vmin_img, vmax_img = dynamic_windowing(gt_img, window_perc)
    diff_images = [np.abs(gt_img - im) for im in methods_imgs]
    diff_max = np.percentile(np.concatenate([d.ravel() for d in diff_images]), window_perc)
    xbox, ybox = (box_size, box_size) if isinstance(box_size, int) else box_size

    fig = plt.figure(figsize=(ncols * 3.5 + 0.6, 10))
    gs = fig.add_gridspec(3, ncols + 1, width_ratios=[3.5] * ncols + [0.25], wspace=0.02, hspace=0.05)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(3)])
    cax = fig.add_subplot(gs[2, ncols])
    diff_ref = None

    for col in range(ncols):
        img = plot_images[col]
        is_gt = (col == ncols - 1)

        axs[0, col].imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img, interpolation='nearest')
        axs[0, col].set_title(plot_names[col], fontsize=9, fontweight='bold', pad=3)
        if is_gt:
            axs[0, col].add_patch(Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                                           linewidth=1.5, edgecolor='r', facecolor='none'))
        axs[0, col].axis('off')

        axs[1, col].imshow(get_zoom_region(img, x, y, xbox, ybox), cmap='gray', vmin=vmin_img, vmax=vmax_img,
                           interpolation='nearest')
        axs[1, col].axis('off')

        if not is_gt:
            diff_ref = axs[2, col].imshow(diff_images[col] * subfactor, cmap='gray', vmin=0, vmax=diff_max)
            axs[2, col].text(0.02, 0.97, 'Diff', transform=axs[2, col].transAxes,
                             color='cyan', fontsize=8, fontweight='bold', va='top')
        else:
            axs[2, col].axis('off')

    if diff_ref is not None:
        cb = plt.colorbar(diff_ref, cax=cax)
        cb.set_label('|GT − recon|', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cax.text(0.5, -0.04, 'shared scale', transform=cax.transAxes, fontsize=6, ha='center', color='gray')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"), bbox_inches='tight', dpi=200)
    else:
        plt.show()
    plt.close(fig)


def mplot(images, names, window_perc=99.9, subfactor=1,
          x=50, y=50, box_size=(20, 20), show_mask=False,
          wspace=0.0, hspace=0.0, filename="comparison", save_path=None,
          layout="vertical"):
    n = len(images)
    if n <= 1:
        return

    gt_img = np.abs(images[0])
    methods_imgs = [np.abs(im) for im in images[1:]]

    if layout == "horizontal":
        plot_images = methods_imgs + [gt_img]
        plot_names = names[1:] + [names[0]]
    else:
        plot_images = [gt_img] + methods_imgs
        plot_names = [names[0]] + names[1:]

    n_images = len(plot_images)

    if layout == "horizontal":
        nrows, ncols = 3, n_images
    elif layout == "vertical":
        nrows, ncols = n_images, 3
    else:
        raise ValueError("layout must be 'horizontal' or 'vertical'.")

    vmin_img, vmax_img = dynamic_windowing(gt_img, window_perc)
    diff_images = [np.abs(gt_img - im) for im in plot_images]
    diff_max = np.percentile(np.concatenate([d.ravel() for d in diff_images]), window_perc)
    xbox, ybox = (box_size, box_size) if isinstance(box_size, int) else box_size

    cell_size = 3.0
    fig = plt.figure(figsize=(ncols * cell_size, nrows * cell_size))
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        pass

    col_w = (1 - wspace * (ncols - 1)) / ncols
    row_h = (1 - hspace * (nrows - 1)) / nrows

    axs = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            left = c * (col_w + wspace)
            bottom = 1 - (r + 1) * row_h - r * hspace
            axs[r, c] = fig.add_axes([left, bottom, col_w, row_h])

    current_mask = None
    for i in range(n_images):
        img = plot_images[i]

        if layout == "horizontal":
            is_gt = (i == n_images - 1)
            ax_img, ax_zoom, ax_diff = axs[0, i], axs[1, i], axs[2, i]
        else:
            is_gt = (i == 0)
            ax_img, ax_zoom, ax_diff = axs[i, 0], axs[i, 1], axs[i, 2]

        ax_img.imshow(img, cmap='gray', vmin=vmin_img, vmax=vmax_img, aspect='auto')
        if layout == "horizontal":
            ax_img.set_title(plot_names[i], fontsize=9, fontweight='bold', pad=3)
            ax_img.axis('off')
        else:
            ax_img.set_ylabel(plot_names[i], fontsize=15, fontweight='bold', rotation=0, ha='right', va='center')
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_visible(False)
        if is_gt:
            ax_img.add_patch(Rectangle((y - ybox, x - xbox), 2 * ybox, 2 * xbox,
                                       linewidth=1.5, edgecolor='r', facecolor='none'))

        ax_zoom.imshow(get_zoom_region(img, x, y, xbox, ybox), cmap='gray', vmin=vmin_img, vmax=vmax_img,
                       aspect='auto')
        ax_zoom.axis('off')

        if not is_gt:
            current_mask = np.abs(gt_img - img) > 0
            ax_diff.imshow(diff_images[i] * subfactor, cmap='gray', vmin=0, vmax=diff_max, aspect='auto')
            if subfactor != 1:
                ax_diff.text(0.98, 0.03, f'×{subfactor}', transform=ax_diff.transAxes,
                             color='yellow', fontsize=7, va='bottom', ha='right')
        else:
            if show_mask and current_mask is not None:
                ax_diff.imshow(current_mask, cmap='bone')
            else:
                ax_diff.imshow(diff_images[i] * subfactor, cmap='gray', vmin=0, vmax=diff_max, aspect='auto')
        ax_diff.axis('off')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{filename}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_windowed_recon(data, gt_data=None, save_bool=False, filepath=None, save_name=None, window_perc=(0.01, 0.99)):
    reconstructed_image = kspace_to_image(data)
    if gt_data is not None:
        gt_img = kspace_to_image(gt_data)
        vmin, vmax = np.quantile(gt_img, window_perc)
    else:
        vmin, vmax = np.quantile(reconstructed_image, window_perc)

    fig = plt.figure(figsize=(10, 10), frameon=False, dpi=96)
    plt.imshow(reconstructed_image, vmin=vmin, vmax=vmax, cmap='gray')
    plt.axis('off')
    plt.margins(0, 0)

    if save_bool and filepath and save_name:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(os.path.join(filepath, save_name), bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close(fig)


def dynamic_windowing(orig, plot_type = "kspace", window_perc = 99.9):    
    if plot_type == "image":
        # Transform to image space
        ref_img = orig
        vmin = 0
        vmax = np.percentile(ref_img, window_perc)
    elif plot_type == "kspace":
        # Logic for k-space log-magnitude windowing
        ref_k = np.log10(np.abs(orig) + 1e-18)
        vmin = np.percentile(ref_k, 1) # bottom 1%
        vmax = np.percentile(ref_k, window_perc)
    else:
        raise ValueError(f"Unsupported plot-type: {plot_type}")

    calc_window = (vmin, vmax)
    return calc_window

def get_zoom_region(img, x, y, xbox_size, ybox_size):
    """Helper to extract zoom region."""
    return img[x - xbox_size : x + xbox_size, y - ybox_size : y + ybox_size]


def grid(
        images,
        names,
        window_perc=99.9, subfactor=1,
        x=50, y=50, box_size=(20, 20),
        filename="images_comparison", save_path=None
):
    """
    Docstring for grid
    
    :param images: Description
    :param names: Description
    :param x: Description
    :param y: Description
    :param boxsize: Description
    :param filename: Description
    :param save_path: Description
    """

    num_imgs = len(images)
    num_methods = len(names)
    
    if num_imgs == 0 or num_methods == 0:
        print("No parameters or methods to plot.")
        return

    if num_imgs != num_methods:
        raise ValueError("The number of images does not correspond to the number of labels.")

    if num_imgs <= 1:
        print("Missing reconstructed images.")
        return
    
    # --- Global windowing ---
    gt_img = images[0]
    global_min, global_max = dynamic_windowing(gt_img, plot_type="image", window_perc=window_perc)
    
    # --- Diff windowing ---
    diff_images = [np.abs(gt_img - img) for img  in images]
    diff_min = 0
    diff_max = np.max(diff_images, window_perc)

    # --- Figure Layout ---
    nrows = 3
    ncols = num_imgs

    _, axs = plt.subplot(nrows, ncols, figsize=(), squeeze=False, gridspec_kw={'wspace': 0, 'hspace': 0})

    if isinstance(box_size, (list, tuple)):
        xbox_size, ybox_size = box_size
    else:
        xbox_size = ybox_size = box_size

    for m_idx, img in enumerate(images[1:]):
        col = m_idx

        # Image
        ax_img = axs[0, col]
        ax_img.imshow(img, cmap='gray', vmin=global_min, vmax=global_max)

        # Zoom
        ax_zoom = axs[1, col]
        zoom_img = get_zoom_region(img, x, y, xbox_size, ybox_size)
        ax_zoom.imshow(zoom_img, cmap='gray', vmin=global_min, vmax=global_max)

        # Diff
        ax_diff = axs[2, col]
        ax_diff.imshow(diff_images[m_idx] * subfactor, cmap='gray', 
                        vmin=diff_min, vmax=diff_max)
        ax_diff.text(0.05, 0.05, f'x{subfactor}', transform=ax_diff.transAxes,
                        fontsize=10, fontweight='bold', color='white')
    
    # Ground truth

    gt_col = ncols - 1

    # Image
    ax_img = axs[0, gt_col]
    
    ax_img.set_title('Ground truth', fontsize=12, pad=10,  fontweight='bold')
    ax_img.imshow(gt_img, cmap='gray', vmin=global_min, vmax=global_max)
    rect = Rectangle((y - ybox_size, x - xbox_size), 2 * ybox_size, 2 * xbox_size,
                        linewidth=1, edgecolor='r', facecolor='none')
    ax_img.add_patch(rect)

    # Zoom
    axs[1, gt_col].imshow(get_zoom_region(gt_img, x, y, xbox_size, ybox_size), 
                                    cmap='gray', vmin=global_min, vmax=global_max)

    # Diff (Zeros for GT)
    axs[2, gt_col].imshow(np.zeros_like(gt_img), cmap='gray', vmin=0, vmax=1)

    



    return
