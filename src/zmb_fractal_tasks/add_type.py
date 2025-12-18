"""Fractal task to add an arbitrary type to each image in an image list."""

from typing import Any

from pydantic import validate_call


@validate_call
def add_type(
    *,
    zarr_url: str,
    output_types: dict[str, bool],
) -> dict[str, Any]:
    """Add an arbitrary type to each selected image in an image list.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_types: Dictionary where keys are the types to add and values
            are booleans indicating wheter the type is set to True or False.
    """
    image_list_updates = {
        "image_list_updates": [{"zarr_url": zarr_url}],
        "types": output_types,
    }
    return image_list_updates
