"""Fractal task to combine multiple acquisitions."""

import logging
import shutil
from typing import Any

import dask.array as da
import numpy as np
from ngio import create_ome_zarr_from_array, open_ome_zarr_container
from pydantic import BaseModel, validate_call


class InitArgsCombineAcquisitionsParallel(BaseModel):
    """Init Args for combine_acquisitions_parallel task.

    Args:
        zarr_urls_to_combine: List of urls to the individual OME-Zarr images to
            be combined.
        keep_individual_acquisitions: If True, keep the individual acquisitions
            and add combined acquisition. If False, delete them and only keep
            the combined acquisition.
    """

    zarr_urls_to_combine: list[str]
    keep_individual_acquisitions: bool = False


@validate_call
def combine_acquisitions_parallel(
    *,
    zarr_url: str,
    init_args: InitArgsCombineAcquisitionsParallel,
) -> dict[str, Any]:
    """Combine multiple acquisitions into a single acquisition.

    Args:
        zarr_url: Absolute path to the new OME-Zarr image.
        init_args: Initialization arguments from the init task.
    """
    combined_image_data = []
    channel_labels = []
    channel_wavelengths = []
    channel_colors = []
    for url in init_args.zarr_urls_to_combine:
        acquisition = url.split("/")[-1]
        # get data
        omezarr = open_ome_zarr_container(url)
        image_data = omezarr.get_image().get_as_dask(axes_order="tczyx")
        combined_image_data.append(image_data)
        # get channel info
        channels = omezarr.get_image().channels_meta.channels
        channel_labels.extend([(acquisition, c.label) for c in channels])
        channel_wavelengths.extend([(acquisition, c.wavelength_id) for c in channels])
        channel_colors.extend([c.channel_visualisation.color for c in channels])

    # process channel labels & wavelength_ids to ensure uniqueness
    channel_labels = np.array(channel_labels)
    if len(set(channel_labels[:, 1])) == len(channel_labels):
        # all channel names are unique, so we can just use them
        channel_labels = channel_labels[:, 1].tolist()
    else:
        # channel names are not unique, so we add acquisition suffix
        channel_labels = [f"{label}_acq{acq}" for acq, label in channel_labels]
    channel_wavelengths = np.array(channel_wavelengths)
    if len(set(channel_wavelengths[:, 1])) == len(channel_wavelengths):
        # all wavelength_ids are unique, so we can just use them
        channel_wavelengths = channel_wavelengths[:, 1].tolist()
    else:
        # wavelength_ids are not unique, so we add acquisition suffix
        channel_wavelengths = [
            f"{wid}_acq{acq}" for acq, wid in channel_wavelengths
        ]

    # concatenate data along channel axis
    combined_image_data = da.concatenate(combined_image_data, axis=1)

    # use first acquisition as reference for metadata
    ref_img = open_ome_zarr_container(init_args.zarr_urls_to_combine[0]).get_image()
    # set up chunks based on reference
    chunks = {'t': 1, 'c': 1, 'z':1, 'y': 1, 'x': 1}
    for axis, size in zip(ref_img.axes, ref_img.chunks, strict=True):
        chunks[axis] = size
    chunks = (chunks['t'], chunks['c'], chunks['z'], chunks['y'], chunks['x'])
    # create new OME-Zarr
    new_omezarr = create_ome_zarr_from_array(
        zarr_url,
        combined_image_data,
        xy_pixelsize=ref_img.pixel_size.x,
        z_spacing=ref_img.pixel_size.z,
        time_spacing=ref_img.pixel_size.t,
        space_unit=ref_img.pixel_size.space_unit,
        time_unit=ref_img.pixel_size.time_unit,
        axes_names=["t", "c", "z", "y", "x"],
        channel_labels=channel_labels,
        channel_wavelengths=channel_wavelengths,
        channel_colors=channel_colors,
        chunks=chunks,
    )

    # copy tables from first acquisition
    ref_omezarr = open_ome_zarr_container(init_args.zarr_urls_to_combine[0])
    for table_name in ref_omezarr.list_tables():
        table = ref_omezarr.get_table(table_name)
        new_omezarr.add_table(table_name, table)

    # TODO: handle label images?

    image_list_updates = [
        {
            "zarr_url": zarr_url,
            "origin": init_args.zarr_urls_to_combine[0],
        }
    ]

    if init_args.keep_individual_acquisitions:
        return {"image_list_updates": image_list_updates}
    else:
        for url in init_args.zarr_urls_to_combine:
            logging.info(f"Deleting individual acquisition at {url}")
            shutil.rmtree(url)
        return {
            "image_list_updates": image_list_updates,
            "image_list_removals": init_args.zarr_urls_to_combine,
        }


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=combine_acquisitions_parallel)
