"""Get image URLs from a Zarr plate."""

from pathlib import Path
from typing import Union

import zarr


def get_image_urls(
    zarr_path: Union[Path, str],
) -> list[str]:
    """Get image URLs from a Zarr plate.

    Args:
        zarr_path: Path to the Zarr plate.
    """
    if isinstance(zarr_path, str):
        zarr_path = Path(zarr_path)
    zarrurl = zarr_path.as_posix()
    group = zarr.open_group(zarrurl, mode="r+")
    images = []
    for well in group.attrs["plate"]["wells"]:
        well_path = f"{zarrurl}/{well['path']}"
        well_group = zarr.open_group(well_path, mode="r+")
        for image in well_group.attrs["well"]["images"]:
            images.append(f"{well_path}/{image['path']}")
    return images
