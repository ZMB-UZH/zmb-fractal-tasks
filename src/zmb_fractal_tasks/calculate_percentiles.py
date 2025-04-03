"""Fractal task to calculate channel percentiles of image."""

from collections.abc import Sequence

import dask.array as da
import numpy as np
import zarr
from ngio import open_ome_zarr_container
from pydantic import validate_call


@validate_call
def calculate_percentiles(
    *,
    zarr_url: str,
    level: str = "0",
    percentiles: Sequence[float] = (1, 99),
) -> dict:
    """Calculate percentiles of image and write them to omero-channels.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Resolution level to calculate percentiles on.
        percentiles: lower and upper percentiles to calculate
    """
    # Preliminary checks
    for percentile in percentiles:
        if not (0 <= percentile <= 100):
            raise ValueError("percentiles need to be between 0 and 100")
    if len(percentiles) != 2:
        raise ValueError("Percentiles needs to be of lenth 2")

    omezarr = open_ome_zarr_container(zarr_url)

    image = omezarr.get_image(path=level)

    roi_table = omezarr.get_table("FOV_ROI_table", check_type="roi_table")

    # TODO: handle case where no channel names are available?
    channels = image.channel_labels

    percentile_values = {}
    for channel in channels:
        channel_idx = image.channel_labels.index(channel)
        dask_arrays = []
        for roi in roi_table.rois():
            dask_arrays.append(image.get_roi(roi, c=channel_idx, mode="dask"))
        percentile_values[channel] = get_percentiles(
            dask_arrays, percentiles=percentiles
        )

    # write omero metadata
    with zarr.open(zarr_url, mode="a") as zarr_file:
        omero_dict = zarr_file.attrs["omero"]
        for channel_dict in omero_dict["channels"]:
            channel_name = channel_dict["label"]
            channel_dict["window"]["start"] = percentile_values[channel_name][0]
            channel_dict["window"]["end"] = percentile_values[channel_name][1]
        zarr_file.attrs["omero"] = omero_dict


def get_percentiles(
    dask_arrays: Sequence[da.Array],
    percentiles: Sequence[float] = (1, 99),
    bin_width: float = 1,
) -> Sequence[float]:
    """Calculate percentiles of one or more dask arrays."""
    dtypes = [dask_array.dtype for dask_array in dask_arrays]
    if len(set(dtypes)) > 1:
        raise ValueError("All dask arrays must have the same dtype")
    dtype = dtypes[0]

    # check if int or float & calculate bin_edges
    if np.issubdtype(dtype, np.integer):
        mn = np.iinfo(dtype).min
        mx = np.iinfo(dtype).max
        step = round(bin_width)
        bin_edges = np.arange(mn, mx + step, step)
    elif np.issubdtype(dtype, np.float):
        mn = np.finfo(dtype).min
        mx = np.finfo(dtype).max
        step = bin_width
        bin_edges = np.arange(mn, mx + step, step)
    else:
        raise TypeError("dtype is neither int nor float")

    hist_da = None
    for dask_array in dask_arrays:
        if hist_da is not None:
            hist_da += da.histogram(dask_array, bins=bin_edges)[0]
        else:
            hist_da = da.histogram(dask_array, bins=bin_edges)[0]

    cumulative_hist = np.cumsum(hist_da.compute())
    total_points = cumulative_hist[-1]
    percentile_indices = np.searchsorted(
        cumulative_hist, np.array(percentiles) * 0.01 * total_points, side="right"
    )
    percentile_values = bin_edges[percentile_indices]
    return percentile_values


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=calculate_percentiles)
