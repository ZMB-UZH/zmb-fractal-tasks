"""Tests for the crop_image task."""

from pathlib import Path

import numpy as np
import pytest
from ngio import Image, open_ome_zarr_container, open_ome_zarr_plate
from pydantic import ValidationError

from zmb_fractal_tasks.crop_image import (
    AdvancedOptions,
    AxisChunkSize,
    AxisCrop,
    crop_image,
)


def test_crop_image_new_image(zarr_MIP_path):
    """Cropping x/y creates a new image with the cropped data."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_image = open_ome_zarr_container(zarr_url).get_image()
    source_data = source_image.get_array(x=slice(100, 300), y=slice(50, 150))

    result = crop_image(
        zarr_url=zarr_url,
        crops=[
            AxisCrop(axis="x", start=100, end=300),
            AxisCrop(axis="y", start=50, end=150),
        ],
    )

    update = result["image_list_updates"][0]
    assert update["origin"] == zarr_url
    zarr_url_new = update["zarr_url"]
    assert Path(zarr_url_new).name == "0_cropped"
    assert Path(zarr_url_new).exists()

    new_image = open_ome_zarr_container(zarr_url_new).get_image()
    assert new_image.dimensions.get("x") == 200
    assert new_image.dimensions.get("y") == 100
    assert new_image.dimensions.get("c") == source_image.dimensions.get("c")
    assert new_image.dtype == source_image.dtype
    np.testing.assert_array_equal(new_image.get_array(), source_data)

    # The input image is untouched
    assert open_ome_zarr_container(zarr_url).get_image().shape == source_image.shape


def test_crop_image_registers_in_plate(zarr_MIP_path):
    """The new image is added to the well metadata of the plate."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    crop_image(zarr_url=zarr_url, crops=[AxisCrop(axis="x", end=200)])

    plate = open_ome_zarr_plate(zarr_MIP_path)
    assert "B/03/0_cropped" in plate.images_paths()


def test_crop_image_rerun_is_idempotent(zarr_MIP_path):
    """Re-running the task overwrites the cropped image without failing."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    crop_image(zarr_url=zarr_url, crops=[AxisCrop(axis="x", end=200)])

    # A second run must not fail on the plate metadata, which already lists the
    # cropped image.
    result = crop_image(zarr_url=zarr_url, crops=[AxisCrop(axis="x", end=123)])

    zarr_url_new = result["image_list_updates"][0]["zarr_url"]
    new_image = open_ome_zarr_container(zarr_url_new).get_image()
    assert new_image.dimensions.get("x") == 123

    # The image is registered exactly once
    plate = open_ome_zarr_plate(zarr_MIP_path)
    assert plate.images_paths().count("B/03/0_cropped") == 1


def test_crop_image_pyramid_and_translation(zarr_MIP_path):
    """All pyramid levels are rebuilt and the crop keeps its world position."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_omezarr = open_ome_zarr_container(zarr_url)
    source_image = source_omezarr.get_image()
    pixel_size_x = source_image.pixel_size.x
    x_index = source_image.axes.index("x")
    source_translation = source_image.dataset.translation[x_index]

    result = crop_image(
        zarr_url=zarr_url, crops=[AxisCrop(axis="x", start=100, end=356)]
    )
    zarr_url_new = result["image_list_updates"][0]["zarr_url"]

    new_omezarr = open_ome_zarr_container(zarr_url_new)
    assert new_omezarr.level_paths == source_omezarr.level_paths
    for level_path in new_omezarr.level_paths:
        source_level = source_omezarr.get_image(path=level_path)
        new_level = new_omezarr.get_image(path=level_path)
        ratio = source_level.pixel_size.x / pixel_size_x
        assert new_level.dimensions.get("x") == round(256 / ratio)

    new_image = new_omezarr.get_image()
    assert new_image.dataset.translation[x_index] == pytest.approx(
        source_translation + 100 * pixel_size_x
    )


def test_crop_image_micrometer_and_fraction(zarr_MIP_path):
    """Crops can be given in physical units and as a fraction of the axis."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_image = open_ome_zarr_container(zarr_url).get_image()
    pixel_size_x = source_image.pixel_size.x
    n_y = source_image.dimensions.get("y")

    result = crop_image(
        zarr_url=zarr_url,
        crops=[
            AxisCrop(axis="x", start=0, end=100 * pixel_size_x, unit="micrometer"),
            AxisCrop(axis="y", start=0.25, end=0.75, unit="fraction"),
        ],
    )
    new_image = open_ome_zarr_container(
        result["image_list_updates"][0]["zarr_url"]
    ).get_image()
    assert new_image.dimensions.get("x") == 100
    assert new_image.dimensions.get("y") == round(0.75 * n_y) - round(0.25 * n_y)


def test_crop_image_channels(zarr_MIP_path):
    """Cropping the channel axis also subsets the channel metadata."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_image = open_ome_zarr_container(zarr_url).get_image()
    source_labels = source_image.channel_labels

    result = crop_image(zarr_url=zarr_url, crops=[AxisCrop(axis="c", start=1, end=3)])
    new_image = open_ome_zarr_container(
        result["image_list_updates"][0]["zarr_url"]
    ).get_image()
    assert new_image.dimensions.get("c") == 2
    assert new_image.channel_labels == source_labels[1:3]


def test_crop_image_labels_and_tables(zarr_MIP_path):
    """Labels are cropped along and ROI tables are clipped to the crop."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_omezarr = open_ome_zarr_container(zarr_url)
    source_label_names = source_omezarr.list_labels()
    assert source_label_names, "test dataset is expected to contain labels"
    pixel_size_x = source_omezarr.get_image().pixel_size.x

    result = crop_image(
        zarr_url=zarr_url,
        crops=[AxisCrop(axis="x", start=100, end=300)],
        advanced_options=AdvancedOptions(crop_labels=True, copy_tables=True),
    )
    new_omezarr = open_ome_zarr_container(result["image_list_updates"][0]["zarr_url"])

    assert new_omezarr.list_labels() == source_label_names
    for label_name in source_label_names:
        assert new_omezarr.get_label(label_name).dimensions.get("x") == 200

    roi_table = new_omezarr.get_table("FOV_ROI_table")
    assert len(roi_table.rois()) > 0
    for roi in roi_table.rois():
        x_slice = roi["x"]
        assert x_slice.start >= 0
        assert x_slice.end <= 200 * pixel_size_x + 1e-6


def test_crop_image_overwrite(zarr_MIP_path):
    """With overwrite_input_image, the input image is replaced in place."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_image = open_ome_zarr_container(zarr_url).get_image()
    source_data = source_image.get_array(x=slice(0, 200))

    result = crop_image(
        zarr_url=zarr_url,
        crops=[AxisCrop(axis="x", end=200)],
        overwrite_input_image=True,
    )

    assert result["image_list_updates"] == [{"zarr_url": zarr_url}]
    assert not (Path(zarr_url).parent / "0__crop_tmp").exists()
    assert not (Path(zarr_url).parent / "0__crop_backup").exists()

    new_image = open_ome_zarr_container(zarr_url).get_image()
    assert new_image.dimensions.get("x") == 200
    np.testing.assert_array_equal(new_image.get_array(), source_data)


def test_crop_image_unknown_axis(zarr_MIP_path):
    """Cropping an axis that does not exist raises an informative error."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="does not exist in the image"):
        crop_image(zarr_url=zarr_url, crops=[AxisCrop(axis="q", end=10)])


def test_crop_image_no_crops(zarr_MIP_path):
    """An empty crop list is rejected."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="No crop was specified"):
        crop_image(zarr_url=zarr_url, crops=[])


def test_crop_image_duplicate_axis(zarr_MIP_path):
    """The same axis cannot be cropped twice."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="cropped more than once"):
        crop_image(
            zarr_url=zarr_url,
            crops=[AxisCrop(axis="x", end=100), AxisCrop(axis="x", start=50)],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": "x"},
        {"axis": "x", "start": -1},
        {"axis": "x", "start": 100, "end": 50},
        {"axis": "x", "start": 0.2, "end": 1.5, "unit": "fraction"},
        {"axis": " ", "end": 10},
    ],
)
def test_axis_crop_validation(kwargs):
    """Invalid crop entries are rejected by the input model."""
    with pytest.raises(ValidationError):
        AxisCrop(**kwargs)


def test_axis_crop_micrometer_on_channel(zarr_MIP_path):
    """Physical units are rejected for non-spatial axes."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="only supported for the spatial axes"):
        crop_image(
            zarr_url=zarr_url,
            crops=[AxisCrop(axis="c", end=2, unit="micrometer")],
        )


def test_crop_image_one_dask_block_per_chunk(zarr_MIP_path, monkeypatch):
    """Every dask block must be written into a single destination chunk.

    If several blocks land in the same chunk, they are written concurrently
    (ngio calls `da.store` with `lock=False`), which makes the atomic rename
    that zarr uses fail with a PermissionError on Windows.
    """
    recorded = []
    original_set_array = Image.set_array

    def recording_set_array(self, patch, *args, **kwargs):
        recorded.append((patch.chunks, self.chunks))
        return original_set_array(self, patch, *args, **kwargs)

    monkeypatch.setattr(Image, "set_array", recording_set_array)

    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    # A fractional crop along y spans several chunks of the source image
    crop_image(
        zarr_url=zarr_url,
        crops=[AxisCrop(axis="y", start=0.25, end=0.75, unit="fraction")],
        advanced_options=AdvancedOptions(crop_labels=False, copy_tables=False),
    )

    assert recorded, "no array was written"
    for block_sizes, dest_chunks in recorded:
        for sizes, chunk in zip(block_sizes, dest_chunks, strict=True):
            assert all(size <= chunk for size in sizes)
            offset = 0
            for size in sizes[:-1]:
                offset += size
                assert offset % chunk == 0, (
                    f"block boundary at {offset} is not on a chunk boundary "
                    f"(chunk size {chunk})"
                )


def test_crop_image_new_chunk_sizes(zarr_MIP_path):
    """Listed axes get the requested chunk size, the others keep the input's."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_omezarr = open_ome_zarr_container(zarr_url)
    source_chunks = source_omezarr.get_image().chunks
    axes = source_omezarr.get_image().axes

    result = crop_image(
        zarr_url=zarr_url,
        crops=[AxisCrop(axis="x", start=100, end=1100)],
        advanced_options=AdvancedOptions(
            new_chunk_sizes=[
                AxisChunkSize(axis="x", size=256),
                AxisChunkSize(axis="y", size=512),
            ]
        ),
    )
    new_omezarr = open_ome_zarr_container(result["image_list_updates"][0]["zarr_url"])

    expected = tuple(
        {"x": 256, "y": 512}.get(axis, chunk)
        for axis, chunk in zip(axes, source_chunks, strict=True)
    )
    assert new_omezarr.get_image().chunks == expected

    # Labels get the same chunk sizes, ignoring the axes they do not have
    for label_name in new_omezarr.list_labels():
        label = new_omezarr.get_label(label_name)
        label_expected = tuple(
            {"x": 256, "y": 512}.get(axis, chunk)
            for axis, chunk in zip(
                label.axes, source_omezarr.get_label(label_name).chunks, strict=True
            )
        )
        assert label.chunks == label_expected


def test_crop_image_default_chunk_sizes_are_inherited(zarr_MIP_path):
    """Without new_chunk_sizes the input chunking is kept, capped by the crop."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    source_image = open_ome_zarr_container(zarr_url).get_image()
    source_chunks = source_image.chunks
    axes = source_image.axes

    result = crop_image(
        zarr_url=zarr_url, crops=[AxisCrop(axis="x", start=100, end=1100)]
    )
    new_image = open_ome_zarr_container(
        result["image_list_updates"][0]["zarr_url"]
    ).get_image()

    # zarr caps every chunk at the size of the array, so the 1000-pixel-wide
    # crop shrinks the x chunk while the other axes keep the input chunking
    cropped_sizes = dict(zip(axes, source_image.shape, strict=True)) | {"x": 1000}
    expected = tuple(
        min(chunk, cropped_sizes[axis])
        for axis, chunk in zip(axes, source_chunks, strict=True)
    )
    assert new_image.chunks == expected
    assert new_image.chunks[axes.index("x")] == 1000


def test_crop_image_chunk_size_unknown_axis(zarr_MIP_path):
    """A chunk size for a non-existing axis is rejected."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="does not exist in the image"):
        crop_image(
            zarr_url=zarr_url,
            crops=[AxisCrop(axis="x", end=100)],
            advanced_options=AdvancedOptions(
                new_chunk_sizes=[AxisChunkSize(axis="q", size=128)]
            ),
        )


def test_crop_image_chunk_size_duplicate_axis(zarr_MIP_path):
    """The same axis cannot get two chunk sizes."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="given more than once"):
        crop_image(
            zarr_url=zarr_url,
            crops=[AxisCrop(axis="x", end=100)],
            advanced_options=AdvancedOptions(
                new_chunk_sizes=[
                    AxisChunkSize(axis="x", size=128),
                    AxisChunkSize(axis="x", size=256),
                ]
            ),
        )


@pytest.mark.parametrize("size", [0, -1])
def test_axis_chunk_size_validation(size):
    """Chunk sizes must be positive."""
    with pytest.raises(ValidationError):
        AxisChunkSize(axis="x", size=size)


def test_crop_image_custom_suffix_via_advanced_options(zarr_MIP_path):
    """A suffix set in advanced_options is used to name the new image."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    result = crop_image(
        zarr_url=zarr_url,
        crops=[AxisCrop(axis="x", end=200)],
        advanced_options=AdvancedOptions(new_image_suffix="_roi"),
    )
    assert Path(result["image_list_updates"][0]["zarr_url"]).name == "0_roi"


def test_crop_image_empty_suffix_rejected(zarr_MIP_path):
    """An empty suffix would overwrite the input image, so it is rejected."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    with pytest.raises(ValueError, match="must not be empty"):
        crop_image(
            zarr_url=zarr_url,
            crops=[AxisCrop(axis="x", end=200)],
            advanced_options=AdvancedOptions(new_image_suffix=""),
        )
