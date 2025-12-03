"""Fractal task to combine channel histograms of images."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import zarr
from ngio import open_ome_zarr_container
from ngio.tables import GenericTable
from pydantic import validate_call

from zmb_fractal_tasks.utils.histogram import (
    Histogram,
    anndata_to_histograms,
    histograms_to_anndata,
)


@validate_call
def aggregate_plate_histograms(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    histogram_input_name: str = "channel_histograms",
    omero_percentiles: Optional[Sequence[float]] = None,
) -> None:
    """Find all channel histograms in a plate and combine them.

    In each image, a new table is created with the combined histograms.
    The new table is named as the input histogram table with suffix "_plate".

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr images to
            be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: Not used for this task.
            (Standard argument for Fractal tasks, managed by Fractal server).
        histogram_input_name: Table name of the histogram table to be combined.
        omero_percentiles: Percentiles to calculate from the combine histogram
            and add to the omero metadata.
            If None, no percentiles are calculated.
    """
    # identify plates
    plate_to_urls = {}
    for zarr_url in zarr_urls:
        plate_path = Path(zarr_url).as_posix().split(".zarr/")[0] + ".zarr"
        if plate_path not in plate_to_urls:
            plate_to_urls[plate_path] = []
        plate_to_urls[plate_path].append(zarr_url)

    for plate_path, plate_zarr_urls in plate_to_urls.items():
        combined_channel_histogram = {}
        levels = []
        # load and combine histograms from all images in the plate
        for zarr_url in plate_zarr_urls:
            ome_zarr_container = open_ome_zarr_container(zarr_url)
            table = ome_zarr_container.get_table(histogram_input_name)
            adata = table.anndata
            levels.append(adata.uns["level"])
            histo_dict = anndata_to_histograms(adata)
            for channel, histo in histo_dict.items():
                if channel not in combined_channel_histogram:
                    combined_channel_histogram[channel] = Histogram()
                combined_channel_histogram[channel].add_histogram(histo)

        # check if all histograms have the same level
        if len(set(levels)) != 1:
            logging.warning(
                f"Histograms from different levels found in {plate_path}:"
                f" {set(levels)}. Combining anyway."
            )
            level = None
        else:
            level = levels[0]

        # write combined histograms to each image in the plate
        adata = histograms_to_anndata(combined_channel_histogram)
        adata.uns["level"] = level
        generic_table = GenericTable(table_data=adata)
        for zarr_url in plate_zarr_urls:
            omezarr = open_ome_zarr_container(zarr_url)
            omezarr.add_table(histogram_input_name+"_plate", generic_table)

        # calculate percentiles & write omero metadata
        if omero_percentiles is not None:
            if len(omero_percentiles) != 2:
                raise ValueError(
                    "omero_percentiles should be a list of two values: [lower, upper]"
                )
            percentile_values = {}
            for channel, histo in combined_channel_histogram.items():
                percentile_values[channel] = histo.get_quantiles(
                    [p / 100 for p in omero_percentiles]
                )
            # write omero metadata for all images in the plate
            for zarr_url in plate_zarr_urls:
                with zarr.open(zarr_url, mode="a") as zarr_file:
                    omero_dict = zarr_file.attrs["omero"]
                    for channel_dict in omero_dict["channels"]:
                        channel_name = channel_dict["label"]
                        channel_dict["window"]["start"] = percentile_values[channel_name][0]
                        channel_dict["window"]["end"] = percentile_values[channel_name][1]
                    zarr_file.attrs["omero"] = omero_dict


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=aggregate_plate_histograms)
