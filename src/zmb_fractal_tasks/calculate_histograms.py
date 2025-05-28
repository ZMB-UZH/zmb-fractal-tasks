"""Fractal task to calculate channel histograms of image."""

from collections.abc import Sequence
from typing import Optional

import zarr
from ngio import open_ome_zarr_container
from ngio.tables import GenericTable
from pydantic import validate_call

from zmb_fractal_tasks.utils.histogram import Histogram, histograms_to_anndata


@validate_call
def calculate_histograms(
    *,
    zarr_url: str,
    level: str = "0",
    input_ROI_table: str = "FOV_ROI_table",
    bin_width: float = 1,
    omero_percentiles: Optional[Sequence[float]] = None,
    histogram_name: str = "channel_histograms",
) -> dict:
    """Calculate channel histograms of image.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Resolution level to calculate histograms on.
        input_ROI_table: Name of the ROI table over which the task loops
        bin_width: Width of the histogram bins. Default is 1.
        omero_percentiles: Percentiles to calculate and add to the omero metadata.
            If None, no percentiles are calculated.
    """
    omezarr = open_ome_zarr_container(zarr_url)

    image = omezarr.get_image(path=level)

    roi_table = omezarr.get_table(input_ROI_table, check_type="roi_table")

    channels = image.channel_labels

    channel_histos = {}
    for channel in channels:
        channel_idx = image.channel_labels.index(channel)
        channel_histo = Histogram(bin_width=bin_width)
        for roi in roi_table.rois():
            data_da = image.get_roi(roi, c=channel_idx, mode="dask")
            channel_histo.add_histogram(Histogram(data_da, bin_width=bin_width))
        channel_histos[channel] = channel_histo

    adata = histograms_to_anndata(channel_histos)
    adata.uns["level"] = level
    generic_table = GenericTable(anndata=adata)
    omezarr.add_table("channel_histograms", generic_table)

    if omero_percentiles is not None:
        if len(omero_percentiles) != 2:
            raise ValueError(
                "omero_percentiles should be a list of two values: [lower, upper]"
            )
        percentile_values = {}
        for channel in channels:
            percentile_values[channel] = channel_histos[
                channel
            ].get_quantiles([p / 100 for p in omero_percentiles])

        # write omero metadata
        with zarr.open(zarr_url, mode="a") as zarr_file:
            omero_dict = zarr_file.attrs["omero"]
            for channel_dict in omero_dict["channels"]:
                channel_name = channel_dict["label"]
                channel_dict["window"]["start"] = percentile_values[channel_name][0]
                channel_dict["window"]["end"] = percentile_values[channel_name][1]
            zarr_file.attrs["omero"] = omero_dict


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=calculate_histograms)
