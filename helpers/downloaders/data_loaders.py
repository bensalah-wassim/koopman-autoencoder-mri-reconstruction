import os
import sys

import h5py
import numpy as np

path = "E:/KAE"
sys.path.append(os.path.dirname(os.path.abspath(path)))

from KAE.helpers.preprocessing.preprocessing import (crop_in_image_space_multi_coil) # noqa: E402


# Load function
def process_fastmri_h5(
    directory,
    filelist=None,
    slice_idx=None,
    all_slices=False,
    min_coils=15
):
    processed = {}

    filenames = (
        filelist
        if filelist is not None
        else [f for f in os.listdir(directory) if f.endswith(".h5")]
    )

    for filename in filenames:
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        with h5py.File(filepath, "r") as file:
            kspace = file["kspace"]
            num_slices, num_coils = kspace.shape[:2]

            if num_coils < min_coils:
                print(f"Skipping {filename}: insufficient coils ({num_coils})")
                continue

            if all_slices:
                slice_range = range(num_slices)
            elif isinstance(slice_idx, int):
                # Ensures the index is within the valid range
                slice_range = [slice_idx] if 0 <= slice_idx < num_slices else []
            elif isinstance(slice_idx, list):
                # Filters the list to only include valid indices
                slice_range = [i for i in slice_idx if 0 <= i < num_slices]
            else:
                # Default fallback
                slice_range = [0] if num_slices > 0 else []

            processed[filename] = {}

            for sl in slice_range:
                kspace_slice = kspace[sl]
                org_shape = kspace_slice.shape
                

                cropped = crop_in_image_space_multi_coil(
                    kspace_slice,
                    target_height=org_shape[1] // 2,
                    target_width=org_shape[2]
                )

                processed[filename][sl] = cropped.astype(np.complex64)
            print(f"{filename}", org_shape)
    return processed

