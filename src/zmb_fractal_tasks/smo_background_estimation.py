"""Fractal task to estimate background using SMO."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.tables import FeatureTable
from pydantic import validate_call
from smo import SMO


@validate_call
def smo_background_estimation(
    *,
    zarr_url: str,
    pyramid_level: str = "0",
    sigma: float = 0.0,
    size: int = 7,
    subtract_background: bool = False,
    overwrite_input_image: bool = True,
    new_well_subgroup_suffix: str = "_BG_subtracted",
) -> dict[str, Any]:
    """Estimates background of each FOV using SMO.

    Uses the SMO algorithm to estimate background for each FOV & channel. In
    short, SMO uses local gradient amount to identify background pixels. See
    the SMO publication for details: https://doi.org/10.1364/JOSAA.477468

    Limitation: Currently, does not support 3D or time-lapse images.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        pyramid_level: Pyramid level to use for background estimation. Choose
            `0` to process at full resolution.
        sigma: Standard deviation for Gaussian pre-filter to reduce noise.
        size: Window size in pixels to average gradient. Should be smaller than
            foreground objects & background regions.
        subtract_background: If True, subtract the estimated background from
            the image (clipping at zero).
        overwrite_input_image: If True, overwrite the input image. If False,
            create a new well sub-group to store the corrected image. Only used
            if subtract_background is True.
        new_well_subgroup_suffix: Suffix to add to the new well sub-group
            name. Only used if overwrite_input_image is False.
    """
    omezarr = open_ome_zarr_container(zarr_url)
    source_image = omezarr.get_image(path=pyramid_level)

    # TODO: support time-lapses?
    if source_image.is_time_series:
        if source_image.dimensions.get("t") > 1:
            raise ValueError(
                "SMO background estimation does not yet support time-lapse images."
            )
    # TODO: Add options for iterating & masking
    roi_table = omezarr.get_table("FOV_ROI_table")

    channels = source_image.channel_labels

    # Estimate BG for each FOV & channel
    list_of_dfs = []
    for r, roi in enumerate(roi_table.rois()):
        roi_df = pd.DataFrame(data=[{"label": r, "ROI": roi.name}])
        for channel in channels:
            channel_idx = source_image.channel_labels.index(channel)
            patch = source_image.get_roi(roi, c=channel_idx)
            bg_value = estimate_BG_smo(patch, sigma, size)
            roi_df[f"BG_{channel}"] = bg_value
        list_of_dfs.append(roi_df)
    # create feature table
    feat_df = pd.concat(list_of_dfs, ignore_index=True)
    feat_table = FeatureTable(feat_df, reference_label=None)
    omezarr.add_table("BG_feature_table", feat_table, overwrite=True)

    # Apply BG subtraction
    if subtract_background:
        # open new ome-zarr
        if overwrite_input_image:
            output_omezarr = omezarr
        else:
            new_zarr_url = Path(zarr_url).parent / (
                Path(zarr_url).stem + new_well_subgroup_suffix
            )
            output_omezarr = omezarr.derive_image(new_zarr_url, overwrite=True)
            # copy all tables
            for table_name in omezarr.list_tables():
                output_omezarr.add_table(table_name, omezarr.get_table(table_name))
            # TODO: copy all labels? -> how best?
        output_image = output_omezarr.get_image()

        # cycle through FOVs and channels and subtract BG
        for roi in roi_table.rois():
            for channel in channels:
                channel_idx = source_image.channel_labels.index(channel)
                bg = feat_df.loc[feat_df["ROI"] == roi.name, f"BG_{channel}"].values[0]
                patch = source_image.get_roi(roi, c=channel_idx)
                patch_corrected = subtract_BG(patch, bg)
                output_image.set_roi(patch=patch_corrected, roi=roi, c=channel_idx)

        output_image.consolidate()

        if overwrite_input_image:
            image_list_updates = {}
        else:
            image_list_updates = {
                "image_list_updates": [{"zarr_url": new_zarr_url, "origin": zarr_url}]
            }
    else:
        image_list_updates = {}

    return image_list_updates


def estimate_BG_smo(patch: np.ndarray, sigma: float, size: int) -> float:
    """Estimate background using SMO.

    Args:
        patch: nD numpy array (image to estimate BG for)
        sigma : Standard deviation for Gaussian kernel of pre-filter.
        size : Averaging window size in pixels. Should be smaller than
            foreground objects.
    """
    # remove singleton dimensions
    image = np.squeeze(patch)
    # initialize SMO
    smo = SMO(sigma=sigma, size=size, shape=image.shape)
    # estimate BG
    # TODO: expose threshold as parameter?
    bg_value = np.median(smo.bg_mask(image, threshold=0.05).compressed())
    return bg_value


def subtract_BG(patch: np.ndarray, bg_value: float) -> np.ndarray:
    """Subtract background from an image, clipping at zero.

    Args:
        patch: nD numpy array (image to subtract BG from)
        bg_value: background value to subtract
    """
    dtype = patch.dtype
    new_img = np.where(
        patch > bg_value,
        patch - bg_value,
        0,
    )
    return new_img.astype(dtype)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=smo_background_estimation)
