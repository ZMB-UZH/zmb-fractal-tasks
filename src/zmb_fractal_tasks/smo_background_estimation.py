"""Fractal task to estimate background using SMO."""

from typing import Optional

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
    sigma: float = 0.0,
    size: int = 7,
    subtract_background: bool = False,
    new_well_sub_group: Optional[str] = None,
) -> dict:
    """Estimates background of each FOV using SMO.

    Only works with 2D data at the moment.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        sigma : Standard deviation for Gaussian kernel of pre-filter.
        size : Averaging window size in pixels. Should be smaller than
            foreground objects.
        subtract_background : If True, subtract the estimated background from
            the image (clipping at zero).
        new_well_sub_group: Name of new well-subgroup. If None, the input image
            is overwritten. This is only needed if subtract_background is True.
    """
    omezarr = open_ome_zarr_container(zarr_url)
    source_image = omezarr.get_image()
    roi_table = omezarr.get_table("FOV_ROI_table", check_type="roi_table")

    # TODO: handle case where no channel names are available?
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
