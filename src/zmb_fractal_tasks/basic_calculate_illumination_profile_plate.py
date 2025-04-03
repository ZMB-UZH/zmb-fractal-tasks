"""Fractal task to calculate illumination profiles for a plate using BaSiC."""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
from basicpy import BaSiC
from ngio import open_ome_zarr_container
from pydantic import validate_call


@validate_call
def basic_calculate_illumination_profile_plate(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    illumination_profiles_folder: str,
    n_images: int = 128,
    overwrite: bool = False,
    random_seed: Optional[int] = None,
    basic_smoothness: float = 1,
    get_darkfield: bool = True,
) -> dict:
    """Calculate illumination profiles for all channels in a plate using BaSiC.

    Calculates illumination correction profiles based on a random sample
    of FOVs for each channel.
    NOTE: This assumes that all FOVs in the plate have the same dimensions.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr images to
            be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: Not used for this task.
            (Standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path to folder where illumination
            profiles will be saved.
        n_images: Number of images to sample for illumination correction.
        overwrite: If True, overwrite existing illumination profiles.
        random_seed: integer random seed to initialize random number generator.
            None will result in non-reproducibel outputs.
        basic_smoothness: Smoothing parameter for BaSiC (used for both flat-
            and dark-field).
        get_darkfield: If True, calculate darkfield correction.
    """
    random.seed(random_seed)

    logging.info(f"Processing {len(zarr_urls)} images")

    omezarrs = [open_ome_zarr_container(zarr_url) for zarr_url in zarr_urls]
    ngio_images = [omezarr.get_image() for omezarr in omezarrs]

    # check if all FOVs have the same dimensions
    roi_dims = []
    for omezarr in omezarrs:
        roi_table = omezarr.get_table("FOV_ROI_table")
        for roi in roi_table.rois():
            roi_dims.append((roi.z_length, roi.y_length, roi.x_length))

    if not all(dim == roi_dims[0] for dim in roi_dims):
        raise ValueError("FOVs have differing dimensions")

    # get list of all channels
    # TODO: handle case where no channel names are available?
    channels = [ngio_image.channel_labels for ngio_image in ngio_images]
    channels = {channel for sublist in channels for channel in sublist}
    logging.info(f"Processing {len(channels)} channels: {channels}")

    # process each channel
    basic_dict = {}
    for i, channel in enumerate(channels):
        logging.info(f"Processing channel {i}/{len(channels)}: {channel}")
        fov_data_all = []
        for omezarr in omezarrs:
            ngio_image = omezarr.get_image()
            if channel in ngio_image.channel_labels:
                channel_idx = ngio_image.channel_labels.index(channel)
                roi_table = omezarr.get_table("FOV_ROI_table")
                for roi in roi_table.rois():
                    roi_data = ngio_image.get_roi(roi, c=channel_idx, mode="dask")
                    fov_data_all.append(roi_data)
        if len(fov_data_all) >= n_images:
            logging.info(f"Using {n_images} random images out of {len(fov_data_all)}.")
            fov_data_sample = random.sample(fov_data_all, n_images)
        else:
            logging.warning(
                f"{n_images} images requested, but only {len(fov_data_all)} available. "
                + f"Using all {len(fov_data_all)} images."
            )
            fov_data_sample = fov_data_all
        if fov_data_sample[0].shape[1] > 1:
            # take random slice along z-axis
            logging.info("Image is z-stack, taking random slices along z-axis.")
            fov_data_sample = [
                img[0, random.randint(0, img.shape[1] - 1), ...]
                for img in fov_data_sample
            ]
        else:
            fov_data_sample = [img[0, 0, ...] for img in fov_data_sample]
        logging.info("Loading data...")
        basic_data = da.stack(fov_data_sample).compute()

        # calculate illumination correction profile
        logging.info("Calculating illumination correction profile...")
        basic = BaSiC(
            get_darkfield=get_darkfield,
            smoothness_flatfield=basic_smoothness,
            smoothness_darkfield=basic_smoothness,
        )
        basic.fit(basic_data)

        # save illumination correction profile
        logging.info("Saving illumination correction profile...")
        folder_path = Path(illumination_profiles_folder) / f"{channel}"
        if overwrite:
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=False)
        # basic.save_model(model_dir=filename, overwrite=overwrite)
        np.save(folder_path / "flatfield.npy", basic.flatfield)
        np.save(folder_path / "darkfield.npy", basic.darkfield)
        np.save(folder_path / "baseline.npy", basic.baseline)
        basic_dict[channel] = basic


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=basic_calculate_illumination_profile_plate)
