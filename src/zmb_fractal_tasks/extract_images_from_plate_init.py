"""Fractal init task to extract images from a plate."""

import json
import logging
from pathlib import Path
from typing import Literal

from ngio import open_ome_zarr_plate
from pydantic import BaseModel, Field, field_validator, model_validator, validate_call


class AcquisitionSelectionModel(BaseModel):
    """Select acquisitions by acquisition ID, acquisition name, or folder name."""

    mode: Literal["acquisition_id", "acquisition_name", "folder_name"] = (
        "acquisition_id"
    )
    """Mode in which identifiers are interpreted."""
    identifiers: list[str] = Field(default_factory=list)
    """Identifiers of the acquisitions to select (according to the selected
    mode). If empty, all acquisitions will be selected."""

    @field_validator("identifiers", mode="after")
    @classmethod
    def validate_identifiers(cls, value: list[str]) -> list[str]:
        """Ensure identifiers are non-empty strings."""
        for identifier in value:
            if not identifier:
                raise ValueError("Identifiers must be non-empty strings.")
        return value

    @model_validator(mode="after")
    def validate_mode_specific_identifiers(self) -> "AcquisitionSelectionModel":
        """Validate identifiers for the selected mode."""
        if self.mode == "acquisition_id":
            for identifier in self.identifiers:
                try:
                    int(identifier)
                except ValueError as exc:
                    raise ValueError(
                        "acquisition_id identifiers must be convertible to int"
                    ) from exc
        return self


def _get_plate_images_by_acquisition_id(ome_zarr_plate, acquisition_id):
    return ome_zarr_plate.images_paths(acquisition=acquisition_id)


def _get_plate_images_by_acquisition_name(ome_zarr_plate, acquisition_name):
    ids = ome_zarr_plate.acquisition_ids
    names = ome_zarr_plate.acquisitions_names
    if acquisition_name not in names:
        return []
    if names.count(acquisition_name) > 1:
        raise ValueError(
            f"Acquisition name '{acquisition_name}' is not unique in the plate. "
            "Cannot select acquisition by name."
        )
    acquisition_id = ids[names.index(acquisition_name)]
    return ome_zarr_plate.images_paths(acquisition=acquisition_id)


def _get_plate_images_by_folder_name(ome_zarr_plate, folder_name):
    images_paths = []
    for image_path in ome_zarr_plate.images_paths():
        if image_path.split("/")[-1] == folder_name:
            images_paths.append(image_path)
    return images_paths


def _detect_zarr_format(plate_root: Path) -> int:
    """Detect zarr format version (2 or 3) from the plate root directory."""
    if (plate_root / ".zgroup").exists():
        return 2
    if (plate_root / "zarr.json").exists():
        return 3
    raise ValueError(f"Cannot detect zarr format version at {plate_root}")


def _write_zarr_group_metadata(folder_path: Path, zarr_format: int) -> None:
    """Write minimal zarr group metadata files for the given format version."""
    if zarr_format == 2:
        (folder_path / ".zattrs").write_text("{}\n")
        (folder_path / ".zgroup").write_text(
            json.dumps({"zarr_format": 2}, indent=2) + "\n"
        )
    elif zarr_format == 3:
        (folder_path / "zarr.json").write_text(
            json.dumps(
                {"zarr_format": 3, "node_type": "group", "attributes": {}},
                indent=2,
            )
            + "\n"
        )


def _iter_acquisitions(ome_zarr_plate, selection: AcquisitionSelectionModel):
    """Yield (identifier, image_paths) pairs for each acquisition to process."""
    if selection.mode == "acquisition_id":
        identifiers = selection.identifiers or [
            str(aid) for aid in ome_zarr_plate.acquisition_ids
        ]
        for identifier in identifiers:
            images = _get_plate_images_by_acquisition_id(
                ome_zarr_plate, int(identifier)
            )
            if not images:
                logging.warning(f"No images found for acquisition_id={identifier!r}.")
            yield identifier, images

    elif selection.mode == "acquisition_name":
        identifiers = selection.identifiers or ome_zarr_plate.acquisitions_names
        for identifier in identifiers:
            images = _get_plate_images_by_acquisition_name(ome_zarr_plate, identifier)
            if not images:
                logging.warning(f"No images found for acquisition_name={identifier!r}.")
            yield identifier, images

    elif selection.mode == "folder_name":
        identifiers = selection.identifiers or list(
            {p.split("/")[-1] for p in ome_zarr_plate.images_paths()}
        )
        for identifier in identifiers:
            images = _get_plate_images_by_folder_name(ome_zarr_plate, identifier)
            if not images:
                logging.warning(f"No images found for folder_name={identifier!r}.")
            yield identifier, images


@validate_call
def extract_images_from_plate_init(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    acquisitions_to_extract: AcquisitionSelectionModel,
    extract_label_images: bool = False,
    extract_tables: bool = False,
):
    """Extract images from a plate.

    Extracts images from plate and puts them in a flat hierarchy. This is
    mainly intended for use in e.g. QuPath, which currently does not support
    the plate hierarchy.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr images to
            be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: Directory where the extracted OME-Zarr images will be stored.
            (Standard argument for Fractal tasks, managed by Fractal server).
        acquisitions_to_extract: Select acquisitions to extract. If left empty,
            all acquisitions found in the plate will be extracted.
        extract_label_images: Whether to extract label images. (Disable if the
            images will be used in QuPath.)
        extract_tables: Whether to extract tables. (Disable if the images will
            be used in QuPath.)
    """
    zarr_paths = [Path(url) for url in zarr_urls]
    # OME-Zarr HCS hierarchy: plate/{row}/{col}/{image} → 3 levels up to plate root
    plate_roots = {p.parent.parent.parent for p in zarr_paths}
    parallelization_list = []
    for plate_root in plate_roots:
        ome_zarr_plate = open_ome_zarr_plate(plate_root)
        plate_name = plate_root.stem  # e.g. "MyPlate" from "MyPlate.zarr"
        zarr_format = _detect_zarr_format(plate_root)
        for identifier, image_paths in _iter_acquisitions(
            ome_zarr_plate, acquisitions_to_extract
        ):
            out_folder = f"{plate_name}_{identifier}.zarr"
            out_folder_path = Path(zarr_dir) / out_folder
            out_folder_path.mkdir(parents=True, exist_ok=True)
            _write_zarr_group_metadata(out_folder_path, zarr_format)
            for image_path in image_paths:
                # image_path is e.g. "B/03/0"; well folder is "B03"
                parts = image_path.split("/")
                well_folder = parts[0] + parts[1]
                zarr_url_out = Path(zarr_dir) / out_folder / well_folder
                parallelization_list.append(
                    {
                        "zarr_url": zarr_url_out.as_posix(),
                        "init_args": {
                            "zarr_url_source": (plate_root / image_path).as_posix(),
                            "extract_label_images": extract_label_images,
                            "extract_tables": extract_tables,
                        },
                    }
                )

    logging.info(
        f"Returning parallelization list with {len(parallelization_list)} items "
        "for extract_images_from_plate_parallel."
    )
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=extract_images_from_plate_init)
