"""Fractal init task to extract images from a plate."""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from ngio import open_ome_zarr_container, open_ome_zarr_plate
from pydantic import BaseModel, Field, field_validator, model_validator, validate_call

_NUMPY_TO_OME_TYPE = {
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "float32": "float",
    "float64": "double",
}


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


def _hex_color_to_ome_int(hex_color: str) -> int:
    """Convert 6-char hex RGB string to OME-XML 32-bit signed RGBA int (alpha=255)."""
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    rgba = (r << 24) | (g << 16) | (b << 8) | 255
    if rgba >= 2**31:
        rgba -= 2**32
    return rgba


def _collect_image_info_for_omexml(well_folder: str, zarr_url_source: str) -> dict:
    """Read image metadata from a source OME-Zarr for OME-XML generation."""
    src_container = open_ome_zarr_container(zarr_url_source)
    src_img = src_container.get_image()
    sh = src_img.shape
    ps = src_img.pixel_size
    return {
        "name": well_folder,
        "size_c": sh[0],
        "size_z": sh[-3] if len(sh) >= 4 else 1,
        "size_y": sh[-2],
        "size_x": sh[-1],
        "pixel_type": _NUMPY_TO_OME_TYPE.get(str(src_img.dtype), str(src_img.dtype)),
        "physical_size_x": ps.x,
        "physical_size_y": ps.y,
        "channels": [
            {
                "name": ch.label,
                "color_int": _hex_color_to_ome_int(
                    ch.channel_visualisation.color
                    if ch.channel_visualisation is not None
                    else "FFFFFF"
                ),
            }
            for ch in src_img.channels_meta.channels
        ],
    }


def _generate_ome_xml(images_info: list[dict]) -> str:
    """Generate OME-XML with channel metadata for all images.

    Args:
        images_info: list of dicts with keys name, size_c, size_z, size_y,
            size_x, pixel_type, physical_size_x, physical_size_y, and
            channels (list of {name, color_int}).
    """
    ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xsi_ns = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("", ome_ns)
    ET.register_namespace("xsi", xsi_ns)

    ome = ET.Element(f"{{{ome_ns}}}OME")
    ome.set(
        f"{{{xsi_ns}}}schemaLocation",
        f"{ome_ns} {ome_ns}/ome.xsd",
    )
    for i, info in enumerate(images_info):
        img_el = ET.SubElement(ome, f"{{{ome_ns}}}Image")
        img_el.set("ID", f"Image:{i}")
        img_el.set("Name", info["name"])
        pix = ET.SubElement(img_el, f"{{{ome_ns}}}Pixels")
        pix.set("ID", f"Pixels:{i}")
        pix.set("DimensionOrder", "XYZCT")
        pix.set("BigEndian", "false")
        pix.set("Interleaved", "false")
        pix.set("Type", info["pixel_type"])
        pix.set("SizeX", str(info["size_x"]))
        pix.set("SizeY", str(info["size_y"]))
        pix.set("SizeZ", str(info["size_z"]))
        pix.set("SizeC", str(info["size_c"]))
        pix.set("SizeT", "1")
        pix.set("PhysicalSizeX", str(info["physical_size_x"]))
        pix.set("PhysicalSizeXUnit", "µm")
        pix.set("PhysicalSizeY", str(info["physical_size_y"]))
        pix.set("PhysicalSizeYUnit", "µm")
        for j, ch in enumerate(info["channels"]):
            ch_el = ET.SubElement(pix, f"{{{ome_ns}}}Channel")
            ch_el.set("ID", f"Channel:{i}:{j}")
            ch_el.set("Name", ch["name"])
            ch_el.set("Color", str(ch["color_int"]))
            ch_el.set("SamplesPerPixel", "1")

    ET.indent(ome, space="  ")
    xml_body = ET.tostring(ome, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body + "\n"


def _detect_zarr_format(plate_root: Path) -> int:
    """Detect zarr format version (2 or 3) from the plate root directory."""
    if (plate_root / ".zgroup").exists():
        return 2
    if (plate_root / "zarr.json").exists():
        return 3
    raise ValueError(f"Cannot detect zarr format version at {plate_root}")


def _write_zarr_group_metadata(
    folder_path: Path, zarr_format: int, attrs: dict | None = None
) -> None:
    """Write minimal zarr group metadata files for the given format version."""
    attrs = attrs or {}
    if zarr_format == 2:
        (folder_path / ".zattrs").write_text(json.dumps(attrs) + "\n")
        (folder_path / ".zgroup").write_text(
            json.dumps({"zarr_format": 2}, indent=2) + "\n"
        )
    elif zarr_format == 3:
        (folder_path / "zarr.json").write_text(
            json.dumps(
                {"zarr_format": 3, "node_type": "group", "attributes": attrs},
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
    create_omexml: bool = True,
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
        create_omexml: Whether to create an OME-XML file with channel names and
            colors for all extracted images. This follows the bioformats2raw
            layout and allows e.g. QuPath to read channel metadata.
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
            out_folder_attrs = {"bioformats2raw.layout": 3} if create_omexml else None
            _write_zarr_group_metadata(
                out_folder_path, zarr_format, attrs=out_folder_attrs
            )
            images_info = []
            for image_path in image_paths:
                # image_path is e.g. "B/03/0"; well folder is "B03"
                parts = image_path.split("/")
                well_folder = parts[0] + parts[1]
                zarr_url_source = (plate_root / image_path).as_posix()
                zarr_url_out = Path(zarr_dir) / out_folder / well_folder
                if create_omexml:
                    images_info.append(
                        _collect_image_info_for_omexml(well_folder, zarr_url_source)
                    )
                parallelization_list.append(
                    {
                        "zarr_url": zarr_url_out.as_posix(),
                        "init_args": {
                            "zarr_url_source": zarr_url_source,
                            "extract_label_images": extract_label_images,
                            "extract_tables": extract_tables,
                        },
                    }
                )
            if create_omexml and images_info:
                ome_dir = out_folder_path / "OME"
                ome_dir.mkdir(exist_ok=True)
                _write_zarr_group_metadata(ome_dir, zarr_format)
                (ome_dir / "METADATA.ome.xml").write_text(
                    _generate_ome_xml(images_info)
                )
                logging.info(
                    f"Written OME-XML at {ome_dir / 'METADATA.ome.xml'} "
                    f"with {len(images_info)} image(s)."
                )

    logging.info(
        f"Returning parallelization list with {len(parallelization_list)} items "
        "for extract_images_from_plate_parallel."
    )
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=extract_images_from_plate_init)
