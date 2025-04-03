"""Fractal task to apply illumination profiles calculated by BaSiC."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from ngio import open_ome_zarr_container
from pydantic import validate_call


@validate_call
def basic_apply_illumination_profile(
    *,
    zarr_url: str,
    illumination_profiles_folder: str,
    subtract_median_baseline: bool = False,
    new_well_sub_group: Optional[str] = None,
) -> dict:
    """Applies illumination correction to the OME-Zarr.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path of folder of illumination profiles.
        subtract_median_baseline: If True, subtract the median of all baseline
            values from the corrected image.
        new_well_sub_group: Name of new well-subgroup. If this is set,
            overwrite_input needs to be False.
            Example: `0_illumination_corrected`.
    """
    omezarr = open_ome_zarr_container(zarr_url)

    if new_well_sub_group is not None:
        # TODO: see how to return new image-list
        raise ValueError("new_well_sub_group is not implemented yet.")
        # new_zarr_url = Path(zarr_url).parent / new_well_sub_group
        # output_omezarr = omezarr.derive_image(new_zarr_url, overwrite=True)
        # # copy all tables
        # for table_name in omezarr.list_tables():
        #     output_omezarr.add_table(table_name, omezarr.get_table(table_name))
        # # TODO: copy all labels?
    else:
        output_omezarr = omezarr

    source_image = omezarr.get_image()
    output_image = output_omezarr.get_image()

    roi_table = omezarr.get_table("FOV_ROI_table", check_type="roi_table")

    # TODO: handle case where no channel names are available?
    channels = source_image.channel_labels

    # Process each channel & FOV
    for channel in channels:
        # load illumination profiles
        channel_idx = source_image.channel_labels.index(channel)
        folder_path = Path(illumination_profiles_folder) / channel
        flatfield = np.load(folder_path / "flatfield.npy")
        darkfield = np.load(folder_path / "darkfield.npy")
        if subtract_median_baseline:
            baseline_array = np.load(folder_path / "baseline.npy")
            baseline = int(np.median(baseline_array))
        else:
            baseline = 0
        # Correct each FOV
        for roi in roi_table.rois():
            patch = source_image.get_roi(roi, c=channel_idx)
            patch_corrected = correct(patch, flatfield, darkfield, baseline)
            output_image.set_roi(patch=patch_corrected, roi=roi, c=channel_idx)

    output_image.consolidate()


def correct(
    img: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray,
    baseline: int,
):
    """Apply illumination correction to an image.

    Corrects an image, using a given illumination profile (e.g. bright
    in the center of the image, dim outside).

    Args:
        img: 4D numpy array (czyx), with dummy size along c.
        flatfield: 2D numpy array (yx)
        darkfield: 2D numpy array (yx)
        baseline: baseline value to be subtracted from the image
    """
    # Check shapes
    if flatfield.shape != img.shape[2:] or img.shape[0] != 1:
        raise ValueError(
            f"Error in illumination_correction:\n{img.shape=}\n{flatfield.shape=}"
        )

    # Store info about dtype
    dtype = img.dtype
    dtype_max = np.iinfo(dtype).max

    #  Apply the normalized correction matrix (requires a float array)
    # img_stack = img_stack.astype(np.float64)
    new_img = (img - darkfield) / flatfield

    # Background subtraction
    if baseline != 0:
        new_img = np.where(
            new_img > baseline,
            new_img - baseline,
            0,
        )

    # Handle edge case: corrected image may have values beyond the limit of
    # the encoding, e.g. beyond 65535 for 16bit images. This clips values
    # that surpass this limit and triggers a warning
    if np.sum(new_img > dtype_max) > 0:
        logging.warning(
            "Illumination correction created values beyond the max range of "
            f"the current image type. These have been clipped to {dtype_max=}."
        )
        new_img[new_img > dtype_max] = dtype_max

    # Cast back to original dtype and return
    return new_img.astype(dtype)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=basic_apply_illumination_profile)
