"""Tests for extract_images_from_plate_init and _parallel tasks."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ngio import open_ome_zarr_container

from zmb_fractal_tasks.extract_images_from_plate_init import (
    AcquisitionSelectionModel,
    extract_images_from_plate_init,
)
from zmb_fractal_tasks.extract_images_from_plate_parallel import (
    InitArgsExtractImagesFromPlateParallel,
    extract_images_from_plate_parallel,
)


def test_extract_images_from_plate_init_folder_name(zarr_MIP_path, tmp_path):
    """Init task returns correct parallelization list using folder_name mode."""
    zarr_urls = [str(zarr_MIP_path / "B" / "03" / "0")]
    zarr_dir = str(tmp_path / "output")

    result = extract_images_from_plate_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        acquisitions_to_extract=AcquisitionSelectionModel(
            mode="folder_name", identifiers=["0"]
        ),
    )

    assert "parallelization_list" in result
    plist = result["parallelization_list"]
    assert len(plist) == 1

    item = plist[0]
    assert "zarr_url" in item
    assert "init_args" in item

    # Output path should be zarr_dir/{plate_stem}_0.zarr/B/03
    expected_zarr_url = (
        Path(zarr_dir) / f"{zarr_MIP_path.stem}_0.zarr" / "B03"
    ).as_posix()
    assert item["zarr_url"] == expected_zarr_url

    # init_args should point back to the source image
    init_args = item["init_args"]
    assert init_args["zarr_url_source"] == (zarr_MIP_path / "B" / "03" / "0").as_posix()


def test_extract_images_from_plate_init_all_acquisitions(zarr_MIP_path, tmp_path):
    """Init task extracts all acquisitions when identifiers is empty."""
    zarr_urls = [str(zarr_MIP_path / "B" / "03" / "0")]
    zarr_dir = str(tmp_path / "output")

    result = extract_images_from_plate_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        acquisitions_to_extract=AcquisitionSelectionModel(
            mode="folder_name", identifiers=[]
        ),
    )

    plist = result["parallelization_list"]
    # Test plate has one image (B/03/0), so we expect one entry
    assert len(plist) == 1


def test_extract_images_from_plate_parallel_basic(zarr_MIP_path, tmp_path):
    """Parallel task creates output zarr with same structure as source."""
    source_path = zarr_MIP_path / "B" / "03" / "0"
    output_path = tmp_path / "output.zarr"

    init_args = InitArgsExtractImagesFromPlateParallel(
        zarr_url_source=str(source_path),
        extract_label_images=False,
        extract_tables=False,
    )

    result = extract_images_from_plate_parallel(
        zarr_url=str(output_path),
        init_args=init_args,
    )

    # Check return structure
    assert "image_list_updates" in result
    assert result["image_list_updates"][0]["zarr_url"] == str(output_path)
    assert result["image_list_updates"][0]["origin"] == str(source_path)

    # Output zarr should exist with same channel count and levels
    assert output_path.exists()
    src = open_ome_zarr_container(str(source_path))
    dst = open_ome_zarr_container(str(output_path))
    assert dst.num_channels == src.num_channels
    assert dst.level_paths == src.level_paths


def test_extract_images_from_plate_end_to_end(zarr_MIP_path, tmp_path):
    """Init + parallel together produce a valid output zarr."""
    zarr_urls = [str(zarr_MIP_path / "B" / "03" / "0")]
    zarr_dir = str(tmp_path / "output")

    plist = extract_images_from_plate_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        acquisitions_to_extract=AcquisitionSelectionModel(
            mode="folder_name", identifiers=["0"]
        ),
    )["parallelization_list"]

    for item in plist:
        extract_images_from_plate_parallel(
            zarr_url=item["zarr_url"],
            init_args=InitArgsExtractImagesFromPlateParallel(**item["init_args"]),
        )

    output_path = Path(plist[0]["zarr_url"])
    assert output_path.exists()
    dst = open_ome_zarr_container(str(output_path))
    src = open_ome_zarr_container(str(zarr_MIP_path / "B" / "03" / "0"))
    assert dst.num_channels == src.num_channels
    assert dst.level_paths == src.level_paths


def test_extract_images_from_plate_init_create_omexml(zarr_MIP_path, tmp_path):
    """create_omexml=True writes METADATA.ome.xml with channel info."""
    zarr_urls = [str(zarr_MIP_path / "B" / "03" / "0")]
    zarr_dir = str(tmp_path / "output")

    result = extract_images_from_plate_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        acquisitions_to_extract=AcquisitionSelectionModel(
            mode="folder_name", identifiers=["0"]
        ),
        create_omexml=True,
    )

    plist = result["parallelization_list"]
    assert len(plist) == 1

    # Determine the out_folder path from the first item's zarr_url
    out_folder_path = Path(plist[0]["zarr_url"]).parent

    # Root .zattrs should declare bioformats2raw layout
    zattrs = json.loads((out_folder_path / ".zattrs").read_text())
    assert zattrs.get("bioformats2raw.layout") == 3

    # OME subdirectory and METADATA.ome.xml should exist
    ome_dir = out_folder_path / "OME"
    assert ome_dir.is_dir()
    assert (ome_dir / ".zgroup").exists()
    metadata_xml = ome_dir / "METADATA.ome.xml"
    assert metadata_xml.exists()

    # Parse OME-XML and check channel names and colors
    ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    tree = ET.parse(metadata_xml)
    root = tree.getroot()
    images = root.findall(f"{{{ome_ns}}}Image")
    assert len(images) == 1
    assert images[0].get("Name") == "B03"

    pixels = images[0].find(f"{{{ome_ns}}}Pixels")
    assert pixels is not None
    assert pixels.get("Type") == "uint16"
    channels = pixels.findall(f"{{{ome_ns}}}Channel")
    assert len(channels) == 3  # DAPI, nanog, Lamin B1
    channel_names = [ch.get("Name") for ch in channels]
    assert channel_names == ["DAPI", "nanog", "Lamin B1"]
    # Colors should be valid integers
    for ch in channels:
        assert ch.get("Color") is not None
        int(ch.get("Color"))  # should not raise
