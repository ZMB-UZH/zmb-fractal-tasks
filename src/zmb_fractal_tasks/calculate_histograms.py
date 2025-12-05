"""Fractal task to calculate channel histograms of image."""

from collections.abc import Sequence

import zarr
from ngio import open_ome_zarr_container
from ngio.tables import GenericTable
from pydantic import validate_call

from zmb_fractal_tasks.utils.histogram import Histogram, histograms_to_anndata


@validate_call
def calculate_histograms(
    *,
    zarr_url: str,
    pyramid_level: str = "0",
    input_ROI_table: str = "FOV_ROI_table",
    bin_width: float = 1,
    update_display_range: bool = True,
    display_range_percentiles: Sequence[float] = (0.5, 99.5),
    histogram_name: str = "channel_histograms",
) -> None:
    """Calculate channel histograms of image.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        pyramid_level: Resolution level to calculate histograms on. Choose `0`
            for full resolution.
        input_ROI_table: Name of the ROI table over which the task loops
        bin_width: Width of the histogram bins. A bin-width of 1 is suitable
            for integer-valued images (e.g. 8-bit or 16-bit images).
        update_display_range: If True, update the display range of the image.
            (Saved in the omero metadata of the zarr file).
        display_range_percentiles: Percentiles (e.g. [0.5, 99.5]) to use
            for display range calculation. (Only used if update_display_range
            is True).
        histogram_name: Name of the output histogram table.
    """
    omezarr = open_ome_zarr_container(zarr_url)

    image = omezarr.get_image(path=pyramid_level)

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
    adata.uns["pyramid_level"] = pyramid_level
    generic_table = GenericTable(table_data=adata)
    omezarr.add_table(histogram_name, generic_table)

    if update_display_range:
        if len(display_range_percentiles) != 2:
            raise ValueError(
                "display_range_percentiles should be a list of two values: "
                "[lower, upper]"
            )
        percentile_values = {}
        for channel in channels:
            percentile_values[channel] = channel_histos[channel].get_quantiles(
                [p / 100 for p in display_range_percentiles]
            )

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
