"""Fractal task to extract a single image from a plate into a flat hierarchy."""

import logging
from typing import Any

import dask.array as da
from ngio import open_ome_zarr_container
from pydantic import BaseModel, validate_call


class InitArgsExtractImagesFromPlateParallel(BaseModel):
    """Init args for extract_images_from_plate_parallel.

    Args:
        zarr_url_source: Path to the source OME-Zarr image inside the plate.
        extract_label_images: Whether to copy label images to the output.
        extract_tables: Whether to copy tables to the output.
    """

    zarr_url_source: str
    extract_label_images: bool = False
    extract_tables: bool = False


@validate_call
def extract_images_from_plate_parallel(
    *,
    zarr_url: str,
    init_args: InitArgsExtractImagesFromPlateParallel,
) -> dict[str, Any]:
    """Extract a single image from a plate into a flat hierarchy.

    Args:
        zarr_url: Path to the output OME-Zarr image.
            (Standard argument for Fractal tasks, managed by Fractal server).
        init_args: Initialization arguments from the init task.
    """
    logging.info(f"Extracting image from {init_args.zarr_url_source} to {zarr_url}.")
    source_omezarr = open_ome_zarr_container(init_args.zarr_url_source)
    new_omezarr = source_omezarr.derive_image(
        zarr_url,
        copy_labels=init_args.extract_label_images,
        copy_tables=init_args.extract_tables,
        overwrite=True,
    )

    for level_path in source_omezarr.levels_paths:
        src_img = source_omezarr.get_image(path=level_path)
        dst_img = new_omezarr.get_image(path=level_path)
        logging.info(f"Copying level {level_path} (shape {src_img.shape}).")
        da.store(da.from_zarr(src_img.zarr_array), dst_img.zarr_array, lock=False)

    return {
        "image_list_updates": [
            {
                "zarr_url": zarr_url,
                "origin": init_args.zarr_url_source,
            }
        ]
    }


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=extract_images_from_plate_parallel)
