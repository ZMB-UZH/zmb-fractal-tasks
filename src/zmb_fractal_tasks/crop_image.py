"""Fractal task to crop an OME-Zarr image along arbitrary dimensions."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from ngio import open_ome_zarr_container, open_ome_zarr_plate
from ngio.tables import GenericRoiTable
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    validate_call,
)

if TYPE_CHECKING:
    from ngio import Image, Label, OmeZarrContainer, PixelSize, Roi


class AxisCrop(BaseModel):
    """Crop range along one axis of an OME-Zarr image."""

    axis: str = "x"
    """Name of the axis to crop, as it is named in the OME-Zarr metadata
    (typically `x`, `y`, `z`, `t` or `c`)."""

    unit: Literal["pixel", "micrometer", "fraction"] = "pixel"
    """How `start` and `end` are interpreted. `pixel`: indices at the
    highest-resolution level. `micrometer`: distance from the image origin, in
    the physical unit of the image (only for the spatial axes `x`, `y`, `z`).
    `fraction`: relative position along the axis, between 0.0 and 1.0."""

    start: Optional[float] = None
    """Where the crop starts along this axis. Leave empty to start at the
    beginning of the axis."""

    end: Optional[float] = None
    """Where the crop ends along this axis (exclusive). Leave empty to crop
    until the end of the axis."""

    @field_validator("axis", mode="after")
    @classmethod
    def validate_axis(cls, value: str) -> str:
        """Ensure the axis name is a non-empty string."""
        value = value.strip()
        if not value:
            raise ValueError("The axis name must be a non-empty string.")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> "AxisCrop":
        """Ensure that start and end describe a non-empty range."""
        if self.start is None and self.end is None:
            raise ValueError(
                f"The crop of axis '{self.axis}' sets neither start nor end. "
                "Set at least one of them, or remove the entry."
            )
        for name, value in (("start", self.start), ("end", self.end)):
            if value is None:
                continue
            if value < 0:
                raise ValueError(
                    f"The {name} of axis '{self.axis}' must not be negative."
                )
            if self.unit == "fraction" and value > 1:
                raise ValueError(
                    f"The {name} of axis '{self.axis}' must be between 0.0 and "
                    "1.0 if the unit is 'fraction'."
                )
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError(
                f"The end of axis '{self.axis}' must be larger than its start."
            )
        return self


class AxisChunkSize(BaseModel):
    """Chunk size along one axis of the cropped image."""

    axis: str = "x"
    """Name of the axis, as it is named in the OME-Zarr metadata (typically
    `x`, `y`, `z`, `t` or `c`)."""

    size: int = Field(gt=0)
    """Chunk size along this axis, in pixels."""

    @field_validator("axis", mode="after")
    @classmethod
    def validate_axis(cls, value: str) -> str:
        """Ensure the axis name is a non-empty string."""
        value = value.strip()
        if not value:
            raise ValueError("The axis name must be a non-empty string.")
        return value


class AdvancedOptions(BaseModel):
    """Advanced options of the crop task."""

    crop_labels: bool = True
    """If `True`, all label images are cropped to the same region and stored in
    the cropped image. If `False`, the label images are not copied to the
    cropped image."""

    copy_tables: bool = True
    """If `True`, all tables are copied to the cropped image. ROI tables are
    clipped to the crop region and ROIs that lie completely outside of it are
    dropped. Feature tables are copied unchanged, so they may still contain
    rows of labels that were cropped away."""

    new_image_suffix: str = "_cropped"
    """Suffix that is appended to the name of the input image to name the
    cropped image. If an image with that name already exists, it is
    overwritten. Only relevant if `overwrite_input_image=False`."""

    new_chunk_sizes: Optional[list[AxisChunkSize]] = None
    """New chunk sizes of the cropped image, one entry per axis. Axes that are
    not listed keep the chunk size of the input image. Leave empty to keep the
    chunking of the input image."""


# Module-level singleton, so that the default does not need a call in the
# signature of the task. It is never mutated.
_DEFAULT_ADVANCED_OPTIONS = AdvancedOptions()


def _resolve_chunk_sizes(
    new_chunk_sizes: Optional[list[AxisChunkSize]], image: "Image"
) -> dict[str, int]:
    """Validate the requested chunk sizes against the axes of the image."""
    requested: dict[str, int] = {}
    for chunk_size in new_chunk_sizes or []:
        axis = chunk_size.axis
        if axis in requested:
            raise ValueError(
                f"Axis '{axis}' is given more than once in `new_chunk_sizes`. "
                "Specify at most one chunk size per axis."
            )
        if image.dimensions.get(axis) is None:
            raise ValueError(
                f"Axis '{axis}' does not exist in the image. Available axes: "
                f"{list(image.axes)}."
            )
        requested[axis] = chunk_size.size
    return requested


def _chunks_for(
    image: "Image | Label", new_chunk_sizes: dict[str, int]
) -> Optional[tuple[int, ...]]:
    """Return the chunk shape of the cropped image, or None to inherit it.

    Axes that are not listed keep the chunk size of the input image, and axes
    that do not exist in the image (e.g. `c` for a label image) are ignored.
    Note that zarr caps every chunk at the size of the array, so a chunk can
    end up smaller than requested along a heavily cropped axis.
    """
    if not new_chunk_sizes:
        return None
    return tuple(
        new_chunk_sizes.get(axis, chunk)
        for axis, chunk in zip(image.axes, image.chunks, strict=True)
    )


def _resolve_crops(
    crops: list[AxisCrop], image: "Image"
) -> dict[str, tuple[float, float]]:
    """Resolve the user-specified crops into a region in world coordinates.

    The returned coordinates are relative to the image origin and expressed in
    the physical unit of the image. Axes without a physical scale (e.g. `c`)
    have a scale of 1, so their world coordinates equal their pixel indices.

    Args:
        crops: Crop ranges specified by the user.
        image: Highest-resolution image the crops are specified against.

    Returns:
        Mapping of axis name to (start, end) in world coordinates.
    """
    region: dict[str, tuple[float, float]] = {}
    for crop in crops:
        axis = crop.axis
        if axis in region:
            raise ValueError(
                f"Axis '{axis}' is cropped more than once. Specify at most one "
                "crop per axis."
            )
        n_pixels = image.dimensions.get(axis)
        if n_pixels is None:
            raise ValueError(
                f"Axis '{axis}' does not exist in the image. Available axes: "
                f"{list(image.axes)}."
            )
        scale = image.pixel_size.get(axis, default=1.0)
        if crop.unit == "micrometer":
            if axis not in ("x", "y", "z"):
                raise ValueError(
                    "Unit 'micrometer' is only supported for the spatial axes "
                    f"'x', 'y' and 'z', but axis '{axis}' was given. Use "
                    "'pixel' or 'fraction' instead."
                )
            start = 0.0 if crop.start is None else crop.start / scale
            end = float(n_pixels) if crop.end is None else crop.end / scale
        elif crop.unit == "fraction":
            start = 0.0 if crop.start is None else crop.start * n_pixels
            end = float(n_pixels) if crop.end is None else crop.end * n_pixels
        else:
            start = 0.0 if crop.start is None else crop.start
            end = float(n_pixels) if crop.end is None else crop.end

        start_pixel = min(max(round(start), 0), n_pixels - 1)
        end_pixel = min(max(round(end), start_pixel + 1), n_pixels)
        logging.info(
            f"Cropping axis '{axis}' to pixels {start_pixel}:{end_pixel} "
            f"(of {n_pixels})."
        )
        region[axis] = (start_pixel * scale, end_pixel * scale)
    return region


def _region_to_slices(
    region: dict[str, tuple[float, float]], image: "Image | Label"
) -> dict[str, slice]:
    """Convert a world-coordinate region into pixel slices for a given image.

    Axes of the region that do not exist in the image (e.g. `c` for a label
    image) are ignored.
    """
    slices: dict[str, slice] = {}
    for axis, (start_world, end_world) in region.items():
        n_pixels = image.dimensions.get(axis)
        if n_pixels is None:
            continue
        scale = image.pixel_size.get(axis, default=1.0)
        start = min(max(round(start_world / scale), 0), n_pixels - 1)
        end = min(max(round(end_world / scale), start + 1), n_pixels)
        slices[axis] = slice(start, end)
    return slices


def _cropped_shape_and_translation(
    image: "Image | Label", slices: dict[str, slice]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return the shape and the translation of the cropped image.

    The translation is shifted by the crop offset, so that the cropped image
    stays at the same position in world coordinates as the original data.
    """
    shape = tuple(
        slices[axis].stop - slices[axis].start if axis in slices else size
        for axis, size in zip(image.axes, image.shape, strict=True)
    )
    translation = tuple(
        offset + (slices[axis].start * scale if axis in slices else 0.0)
        for axis, offset, scale in zip(
            image.axes,
            image.dataset.translation,
            image.dataset.scale,
            strict=True,
        )
    )
    return shape, translation


def _copy_cropped_array(
    source: "Image | Label", destination: "Image | Label", slices: dict[str, slice]
) -> None:
    """Copy the cropped region of `source` into `destination`.

    The patch is rechunked to the chunks of the destination, so that every dask
    task writes to exactly one chunk. Without this, several tasks can write to
    the same chunk concurrently, which makes the atomic rename that zarr uses
    fail with a PermissionError on Windows.
    """
    patch = source.get_array(mode="dask", **slices)
    destination.set_array(patch.rechunk(destination.chunks))


def _crop_image_data(
    omezarr: "OmeZarrContainer",
    source_image: "Image",
    region: dict[str, tuple[float, float]],
    zarr_url_new: Path,
    new_chunk_sizes: dict[str, int],
) -> "OmeZarrContainer":
    """Derive a new OME-Zarr image containing only the cropped region."""
    slices = _region_to_slices(region, source_image)
    shape, translation = _cropped_shape_and_translation(source_image, slices)

    channels_meta = None
    if "c" in slices:
        channels_meta = list(source_image.channels_meta.channels)[slices["c"]]

    logging.info(f"Creating cropped image with shape {shape} at {zarr_url_new}.")
    new_omezarr = omezarr.derive_image(
        store=zarr_url_new,
        shape=shape,
        translation=translation,
        channels_meta=channels_meta,
        dtype=source_image.dtype,
        chunks=_chunks_for(source_image, new_chunk_sizes),
        overwrite=True,
    )
    new_image = new_omezarr.get_image()
    _copy_cropped_array(source_image, new_image, slices)
    new_image.consolidate()
    return new_omezarr


def _crop_labels(
    omezarr: "OmeZarrContainer",
    new_omezarr: "OmeZarrContainer",
    region: dict[str, tuple[float, float]],
    new_chunk_sizes: dict[str, int],
) -> None:
    """Crop all label images of the input and store them in the new image."""
    for label_name in omezarr.list_labels():
        source_label = omezarr.get_label(label_name)
        slices = _region_to_slices(region, source_label)
        shape, translation = _cropped_shape_and_translation(source_label, slices)
        logging.info(f"Cropping label '{label_name}' to shape {shape}.")
        new_label = new_omezarr.derive_label(
            name=label_name,
            ref_image=source_label,
            shape=shape,
            translation=translation,
            dtype=source_label.dtype,
            chunks=_chunks_for(source_label, new_chunk_sizes),
            overwrite=True,
        )
        _copy_cropped_array(source_label, new_label, slices)
        new_label.consolidate()


def _crop_roi(roi: "Roi", region: dict[str, tuple[float, float]]) -> "Roi | None":
    """Clip an ROI to the crop region and shift it to the new image origin.

    Args:
        roi: ROI in world coordinates.
        region: Crop region in world coordinates.

    Returns:
        The cropped ROI, or None if it does not overlap with the crop region.
    """
    cropped_roi = roi
    for axis, (region_start, region_end) in region.items():
        roi_slice = roi.get(axis)
        if roi_slice is None:
            # The ROI does not constrain this axis, so it also spans the full
            # extent of the cropped image.
            continue
        start = region_start if roi_slice.start is None else roi_slice.start
        end = region_end if roi_slice.end is None else roi_slice.end
        start = max(start, region_start)
        end = min(end, region_end)
        if end <= start:
            return None
        cropped_roi = cropped_roi.update_slice(
            axis, (start - region_start, end - start)
        )
    return cropped_roi


def _crop_roi_table(
    table: GenericRoiTable,
    region: dict[str, tuple[float, float]],
    pixel_size: "PixelSize",
) -> Optional[GenericRoiTable]:
    """Clip all ROIs of a table to the crop region.

    Returns None if no ROI of the table overlaps with the crop region.
    """
    cropped_rois = []
    for roi in table.rois():
        cropped_roi = _crop_roi(roi.to_world(pixel_size), region)
        if cropped_roi is not None:
            cropped_rois.append(cropped_roi)
    if not cropped_rois:
        return None
    return type(table)(rois=cropped_rois, meta=table.meta.model_copy(deep=True))


def _copy_tables(
    omezarr: "OmeZarrContainer",
    new_omezarr: "OmeZarrContainer",
    region: dict[str, tuple[float, float]],
    pixel_size: "PixelSize",
) -> None:
    """Copy all tables to the new image, adapting the ROI tables to the crop."""
    for table_name in omezarr.list_tables():
        table = omezarr.get_table(table_name)
        if not isinstance(table, GenericRoiTable):
            new_omezarr.add_table(table_name, table)
            continue
        cropped_table = _crop_roi_table(table, region, pixel_size)
        if cropped_table is None:
            logging.warning(
                f"All ROIs of table '{table_name}' lie outside of the crop "
                "region. The table is not copied to the cropped image."
            )
            continue
        n_dropped = len(table.rois()) - len(cropped_table.rois())
        if n_dropped:
            logging.info(
                f"Dropped {n_dropped} ROI(s) of table '{table_name}' that lie "
                "outside of the crop region."
            )
        new_omezarr.add_table(table_name, cropped_table)


def _replace_image(zarr_url: Path, zarr_url_tmp: Path) -> None:
    """Replace the input image by the cropped image stored at `zarr_url_tmp`."""
    backup_path = zarr_url.parent / f"{zarr_url.name}__crop_backup"
    if backup_path.exists():
        shutil.rmtree(backup_path)
    zarr_url.rename(backup_path)
    try:
        zarr_url_tmp.rename(zarr_url)
    except OSError:
        backup_path.rename(zarr_url)
        raise
    shutil.rmtree(backup_path)


def _register_image_in_plate(zarr_url: Path, zarr_url_new: Path) -> None:
    """Add the new image to the plate metadata, if the input is part of a plate."""
    # OME-Zarr HCS hierarchy: plate/{row}/{col}/{image} -> 3 levels up to plate root
    plate_root = zarr_url.parent.parent.parent
    row = zarr_url.parent.parent.name
    column = zarr_url.parent.name
    try:
        ome_zarr_plate = open_ome_zarr_plate(plate_root)
        acquisition_id = ome_zarr_plate.get_image_acquisition_id(
            row=row, column=column, image_path=zarr_url.name
        )
    except Exception:
        logging.info(
            f"{zarr_url} does not seem to be part of an OME-Zarr plate. The "
            "cropped image is not added to any plate metadata."
        )
        return
    if zarr_url_new.name in ome_zarr_plate.get_well(row=row, column=column).paths():
        # The task was run before with the same suffix. The image data has just
        # been overwritten, so the existing plate entry is still correct.
        logging.info(
            f"{zarr_url_new.name} is already registered in well {row}/{column} "
            "of the plate."
        )
        return
    ome_zarr_plate.atomic_add_image(
        row=row,
        column=column,
        image_path=zarr_url_new.name,
        acquisition_id=acquisition_id,
    )
    logging.info(f"Added {zarr_url_new.name} to well {row}/{column} of the plate.")


@validate_call
def crop_image(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    crops: list[AxisCrop],
    overwrite_input_image: bool = False,
    # Advanced parameters
    advanced_options: AdvancedOptions = _DEFAULT_ADVANCED_OPTIONS,
) -> dict[str, Any]:
    """Crop an OME-Zarr image along any of its dimensions.

    Add one entry to `crops` per axis you want to crop (e.g. `x`, `y`, `z`, `t`
    or `c`); all other axes are kept in full. Every entry has its own unit, so
    you can e.g. crop `x` and `y` in micrometer while cropping `z` in pixels
    (i.e. planes).

    The position of the crop is stored in the OME-Zarr metadata, so that the
    cropped image stays at its original position in world coordinates.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        crops: Ranges to crop to, one entry per axis. Axes that are not listed
            here are kept in full. Within an entry, leaving `start` or `end`
            empty means "from the beginning" / "until the end" of that axis.
        overwrite_input_image: If `True`, the input image is replaced by the
            cropped image. If `False`, a new image (with the same acquisition
            ID) is created next to the input image and the input image is left
            untouched. NOTE: cropping cannot be undone, so only overwrite the
            input image if you are sure about the crop region.
        advanced_options: Options that rarely need to be changed: which labels
            and tables to carry over, how the new image is named and how it is
            chunked.
    """
    if not crops:
        raise ValueError(
            "No crop was specified. Add at least one axis to `crops`, "
            "otherwise the task would only duplicate the input image."
        )
    if not overwrite_input_image and not advanced_options.new_image_suffix:
        raise ValueError(
            "`new_image_suffix` must not be empty if the input image is not "
            "overwritten, since the cropped image would otherwise overwrite it."
        )

    zarr_url_path = Path(zarr_url)
    omezarr = open_ome_zarr_container(zarr_url)
    source_image = omezarr.get_image()
    region = _resolve_crops(crops, source_image)
    chunk_sizes = _resolve_chunk_sizes(advanced_options.new_chunk_sizes, source_image)

    if overwrite_input_image:
        zarr_url_new = zarr_url_path.parent / f"{zarr_url_path.name}__crop_tmp"
    else:
        zarr_url_new = (
            zarr_url_path.parent
            / f"{zarr_url_path.name}{advanced_options.new_image_suffix}"
        )

    new_omezarr = _crop_image_data(
        omezarr, source_image, region, zarr_url_new, chunk_sizes
    )
    if advanced_options.crop_labels:
        _crop_labels(omezarr, new_omezarr, region, chunk_sizes)
    if advanced_options.copy_tables:
        _copy_tables(omezarr, new_omezarr, region, source_image.pixel_size)

    if overwrite_input_image:
        _replace_image(zarr_url_path, zarr_url_new)
        logging.info(f"Replaced {zarr_url} by its cropped version.")
        return {"image_list_updates": [{"zarr_url": zarr_url}]}

    _register_image_in_plate(zarr_url_path, zarr_url_new)
    return {
        "image_list_updates": [
            {"zarr_url": zarr_url_new.as_posix(), "origin": zarr_url}
        ]
    }


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=crop_image)
