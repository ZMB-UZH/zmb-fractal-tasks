import shutil
from pathlib import Path

import numpy as np
from ngio import open_ome_zarr_container
from ngio.tables import GenericTable

from zmb_fractal_tasks.combine_acquisitions_parallel import (
    InitArgsCombineAcquisitionsParallel,
    combine_acquisitions_parallel,
)


def test_combine_acquisitions_parallel_basic(zarr_MIP_path, tmp_path):
    """Test basic combination of two acquisitions."""
    # Create two separate acquisition copies
    acq0_path = tmp_path / "acq0.zarr"
    acq1_path = tmp_path / "acq1.zarr"
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq0_path, dirs_exist_ok=True)
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq1_path, dirs_exist_ok=True)

    # Set up the combined output path
    combined_path = tmp_path / "combined.zarr"

    # Get initial channel count
    omezarr_acq0 = open_ome_zarr_container(str(acq0_path))
    initial_channels = len(omezarr_acq0.get_image().channels_meta.channels)

    # Create init args
    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=[str(acq0_path), str(acq1_path)],
        keep_individual_acquisitions=True,
    )

    # Run the parallel task
    result = combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Check the result structure
    assert "image_list_updates" in result
    assert len(result["image_list_updates"]) == 1
    assert result["image_list_updates"][0]["zarr_url"] == str(combined_path)

    # Verify the combined image was created
    assert combined_path.exists()

    # Check that channels were concatenated
    combined_omezarr = open_ome_zarr_container(str(combined_path))
    combined_image = combined_omezarr.get_image()
    combined_channels = len(combined_image.channels_meta.channels)

    assert combined_channels == initial_channels * 2

    # Verify original acquisitions still exist when keep_individual_acquisitions=True
    assert acq0_path.exists()
    assert acq1_path.exists()

    # Verify image_list_removals is not in result when keeping originals
    assert "image_list_removals" not in result


def test_combine_acquisitions_parallel_delete_originals(zarr_MIP_path, tmp_path):
    """Test combination with deletion of original acquisitions."""
    # Create two separate acquisition copies
    acq0_path = tmp_path / "acq0.zarr"
    acq1_path = tmp_path / "acq1.zarr"
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq0_path, dirs_exist_ok=True)
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq1_path, dirs_exist_ok=True)

    # Set up the combined output path
    combined_path = tmp_path / "combined.zarr"

    # Create init args with keep_individual_acquisitions=False
    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=[str(acq0_path), str(acq1_path)],
        keep_individual_acquisitions=False,
    )

    # Run the parallel task
    result = combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Check the result structure
    assert "image_list_updates" in result
    assert "image_list_removals" in result
    assert len(result["image_list_removals"]) == 2

    # Verify the combined image was created
    assert combined_path.exists()

    # Verify original acquisitions were deleted
    assert not acq0_path.exists()
    assert not acq1_path.exists()


def test_combine_acquisitions_parallel_unique_channel_names(zarr_MIP_path, tmp_path):
    """Test that channel labels are handled correctly when unique."""
    # Create two acquisitions with the same channels
    acq0_path = tmp_path / "acq0.zarr"
    acq1_path = tmp_path / "acq1.zarr"
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq0_path, dirs_exist_ok=True)
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq1_path, dirs_exist_ok=True)

    combined_path = tmp_path / "combined.zarr"

    # Get original channel labels
    omezarr_acq0 = open_ome_zarr_container(str(acq0_path))
    original_channels = omezarr_acq0.get_image().channels_meta.channels
    original_labels = [c.label for c in original_channels]

    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=[str(acq0_path), str(acq1_path)],
        keep_individual_acquisitions=True,
    )

    # Run the parallel task
    combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Check combined channel labels
    combined_omezarr = open_ome_zarr_container(str(combined_path))
    combined_channels = combined_omezarr.get_image().channels_meta.channels
    combined_labels = [c.label for c in combined_channels]

    # Since we're combining the same acquisition twice, channel names are not unique
    # They should have acquisition suffixes
    # The acquisition names are derived from the last part of the path
    assert len(combined_labels) == len(original_labels) * 2
    assert any("_acq" in label for label in combined_labels)


def test_combine_acquisitions_parallel_metadata_preservation(zarr_MIP_path, tmp_path):
    """Test that metadata is correctly preserved in the combined image."""
    # Create two acquisitions
    acq0_path = tmp_path / "acq0.zarr"
    acq1_path = tmp_path / "acq1.zarr"
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq0_path, dirs_exist_ok=True)
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq1_path, dirs_exist_ok=True)

    combined_path = tmp_path / "combined.zarr"

    # Get reference metadata
    ref_omezarr = open_ome_zarr_container(str(acq0_path))
    ref_image = ref_omezarr.get_image()
    ref_pixel_size = ref_image.pixel_size

    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=[str(acq0_path), str(acq1_path)],
        keep_individual_acquisitions=True,
    )

    # Run the parallel task
    combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Check that metadata was preserved
    combined_omezarr = open_ome_zarr_container(str(combined_path))
    combined_image = combined_omezarr.get_image()
    combined_pixel_size = combined_image.pixel_size

    # Verify pixel sizes match
    assert combined_pixel_size.x == ref_pixel_size.x
    assert combined_pixel_size.y == ref_pixel_size.y
    assert combined_pixel_size.z == ref_pixel_size.z
    assert combined_pixel_size.t == ref_pixel_size.t
    assert combined_pixel_size.space_unit == ref_pixel_size.space_unit
    # assert combined_pixel_size.time_unit == ref_pixel_size.time_unit


def test_combine_acquisitions_parallel_data_concatenation(zarr_MIP_path, tmp_path):
    """Test that data is correctly concatenated along the channel axis."""
    # Create two acquisitions
    acq0_path = tmp_path / "acq0.zarr"
    acq1_path = tmp_path / "acq1.zarr"
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq0_path, dirs_exist_ok=True)
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq1_path, dirs_exist_ok=True)

    combined_path = tmp_path / "combined.zarr"

    # Get reference data
    ref_omezarr = open_ome_zarr_container(str(acq0_path))
    ref_data = ref_omezarr.get_image().get_as_dask(axes_order="tczyx")
    ref_shape = ref_data.shape

    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=[str(acq0_path), str(acq1_path)],
        keep_individual_acquisitions=True,
    )

    # Run the parallel task
    combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Check data shape
    combined_omezarr = open_ome_zarr_container(str(combined_path))
    combined_data = combined_omezarr.get_image().get_as_dask(axes_order="tczyx")
    combined_shape = combined_data.shape

    # Verify shape - channel dimension should be doubled, others unchanged
    assert combined_shape[0] == ref_shape[0]  # time
    assert combined_shape[1] == ref_shape[1] * 2  # channels (concatenated)
    assert combined_shape[2] == ref_shape[2]  # z
    assert combined_shape[3] == ref_shape[3]  # y
    assert combined_shape[4] == ref_shape[4]  # x

    # Verify data integrity - first half should match acq0
    np.testing.assert_array_equal(
        combined_data[:, : ref_shape[1], ...].compute(), ref_data.compute()
    )


def test_combine_acquisitions_parallel_three_acquisitions(zarr_MIP_path, tmp_path):
    """Test combining more than two acquisitions."""
    # Create three acquisitions
    acq_paths = []
    for i in range(3):
        acq_path = tmp_path / f"acq{i}.zarr"
        shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq_path, dirs_exist_ok=True)
        acq_paths.append(str(acq_path))

    combined_path = tmp_path / "combined.zarr"

    # Get initial channel count
    omezarr_acq0 = open_ome_zarr_container(acq_paths[0])
    initial_channels = len(omezarr_acq0.get_image().channels_meta.channels)

    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=acq_paths,
        keep_individual_acquisitions=True,
    )

    # Run the parallel task
    result = combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Verify the combined image
    assert combined_path.exists()
    combined_omezarr = open_ome_zarr_container(str(combined_path))
    combined_channels = len(combined_omezarr.get_image().channels_meta.channels)

    # Should have 3x the original channels
    assert combined_channels == initial_channels * 3


def test_combine_acquisitions_parallel_copy_tables(zarr_MIP_path, tmp_path):
    """Test that tables from first acquisition are copied to combined acquisition."""
    import pandas as pd

    # Create two separate acquisition copies
    acq0_path = tmp_path / "acq0.zarr"
    acq1_path = tmp_path / "acq1.zarr"
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq0_path, dirs_exist_ok=True)
    shutil.copytree(zarr_MIP_path / "B" / "03" / "0", acq1_path, dirs_exist_ok=True)

    # Add test tables to both acquisitions
    omezarr_acq0 = open_ome_zarr_container(str(acq0_path))
    omezarr_acq1 = open_ome_zarr_container(str(acq1_path))

    # Create test tables with different data
    test_table_0 = GenericTable(
        pd.DataFrame({"label": [1, 2, 3], "value_acq0": [10, 20, 30]})
    )
    test_table_1 = GenericTable(
        pd.DataFrame({"label": [1, 2, 3], "value_acq1": [100, 200, 300]})
    )

    omezarr_acq0.add_table("test_table", test_table_0)
    omezarr_acq1.add_table("test_table", test_table_1)

    # Also add another table only to acq0
    extra_table = GenericTable(pd.DataFrame({"label": [1], "extra": [999]}))
    omezarr_acq0.add_table("extra_table", extra_table)

    # Set up the combined output path
    combined_path = tmp_path / "combined.zarr"

    # Create init args
    init_args = InitArgsCombineAcquisitionsParallel(
        zarr_urls_to_combine=[str(acq0_path), str(acq1_path)],
        keep_individual_acquisitions=True,
    )

    # Run the parallel task
    result = combine_acquisitions_parallel(
        zarr_url=str(combined_path),
        init_args=init_args,
    )

    # Check that combined acquisition was created
    assert combined_path.exists()

    # Verify tables from first acquisition were copied
    combined_omezarr = open_ome_zarr_container(str(combined_path))
    table_names = combined_omezarr.list_tables()

    assert "test_table" in table_names
    assert "extra_table" in table_names