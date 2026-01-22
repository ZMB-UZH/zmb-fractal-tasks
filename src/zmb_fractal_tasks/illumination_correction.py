"""Fractal task to apply illumination correction."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ngio import open_ome_zarr_container
from pydantic import BaseModel, validate_call


@validate_call
def illumination_correction(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    illumination_profiles_folder: str,
    illumination_profiles: dict[str, str],
    background: int = 0,
    input_ROI_table: str = "FOV_ROI_table",
    overwrite_input_image: bool = True,
    # Advanced parameters
    new_well_subgroup_suffix: str = "_illum_corr",
) -> dict[str, Any]:
    """
    Applies illumination correction to the images in the OME-Zarr.

    Assumes that the illumination correction profiles were generated before
    separately and that the same background subtraction was used during
    calculation of the illumination correction (otherwise, it will not work
    well & the correction may only be partial).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path of folder of illumination profiles.
        illumination_profiles: Dictionary where keys match the `wavelength_id`
            attributes of existing channels (e.g.  `A01_C01` ) and values are
            the filenames of the corresponding illumination profiles.
        background: Background value that is subtracted from the image before
            the illumination correction is applied. Set it to `0` if you don't
            want any background subtraction.
        input_ROI_table: Name of the ROI table that contains the information
            about the location of the individual field of views (FOVs) to
            which the illumination correction shall be applied. Defaults to
            "FOV_ROI_table", the default name Fractal converters give the ROI
            tables that list all FOVs separately. If you generated your
            OME-Zarr with a different converter and used Import OME-Zarr to
            generate the ROI tables, `image_ROI_table` is the right choice if
            you only have 1 FOV per Zarr image and `grid_ROI_table` if you
            have multiple FOVs per Zarr image and set the right grid options
            during import.
        overwrite_input_image: If `True`, the results of this task will
            overwrite the input image data. If false, a new image is generated
            and the illumination corrected data is saved there.
        new_well_subgroup_suffix: What suffix to append to the illumination
            corrected images. Only relevant if `overwrite_input=False`.
    """
    omezarr = open_ome_zarr_container(zarr_url)

    if overwrite_input_image:
        output_omezarr = omezarr
    else:
        new_zarr_url = Path(zarr_url).parent / (
            Path(zarr_url).stem + "_" + new_well_subgroup_suffix
        )
        output_omezarr = omezarr.derive_image(new_zarr_url, overwrite=True)
        # copy all tables
        for table_name in omezarr.list_tables():
            output_omezarr.add_table(table_name, omezarr.get_table(table_name))
        # TODO: copy all labels? -> how best?

    source_image = omezarr.get_image()
    output_image = output_omezarr.get_image()

    roi_table = omezarr.get_table("FOV_ROI_table")

    channels = source_image.wavelength_ids

    # Process each channel & FOV
    for channel, file_name in illumination_profiles.items():
        # load illumination profiles
        channel_idx = source_image.wavelength_ids.index(channel)
        file_path = Path(illumination_profiles_folder) / file_name
        flatfield = np.load(file_path)
        XXX
        else:
            baseline = 0
        # Correct each FOV
        for roi in roi_table.rois():
            patch = source_image.get_roi(
                roi, c=channel_idx, axes_order=["c", "z", "y", "x"]
            )
            patch_corrected = correct(patch, flatfield, darkfield, baseline)
            output_image.set_roi(
                patch=patch_corrected,
                roi=roi,
                c=channel_idx,
                axes_order=["c", "z", "y", "x"],
            )

    output_image.consolidate()

    if init_args.overwrite_input_image:
        image_list_updates = {"image_list_updates": [{"zarr_url": zarr_url}]}
    else:
        image_list_updates = {
            "image_list_updates": [{"zarr_url": new_zarr_url, "origin": zarr_url}]
        }
    return image_list_updates


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
