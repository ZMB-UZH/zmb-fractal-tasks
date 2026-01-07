import shutil
from pathlib import Path

from ngio import ImageInWellPath, create_empty_plate, open_ome_zarr_plate

from zmb_fractal_tasks.combine_acquisitions_init import (
    combine_acquisitions_init,
)


def test_combine_acquisitions_init_basic(zarr_MIP_path, tmp_path):
    """Test combine_acquisitions_init with a plate having two acquisitions."""
    # Create a new plate from scratch with two acquisitions
    test_zarr = tmp_path / "test_plate.zarr"

    list_of_images = [
        ImageInWellPath(
            path="0", row="B", column=3, acquisition_id=0, acquisition_name="acq0"
        ),
        ImageInWellPath(
            path="1", row="B", column=3, acquisition_id=1, acquisition_name="acq1"
        ),
    ]

    plate = create_empty_plate(
        store=str(test_zarr),
        name="test_plate",
        images=list_of_images,
        overwrite=True,
    )

    # Copy actual image data from fixture to the new plate
    source_acq = zarr_MIP_path / "B" / "03" / "0"
    acq0_path = test_zarr / "B" / "03" / "0"
    acq1_path = test_zarr / "B" / "03" / "1"
    shutil.copytree(source_acq, acq0_path, dirs_exist_ok=True)
    shutil.copytree(source_acq, acq1_path, dirs_exist_ok=True)
    zarr_urls = [str(acq0_path), str(acq1_path)]
    result = combine_acquisitions_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        keep_individual_acquisitions=True,
    )

    # Check that parallelization list is returned
    assert isinstance(result, dict)
    assert "parallelization_list" in result
    parallelization_list = result["parallelization_list"]
    assert len(parallelization_list) == 1

    # Check the structure of the parallelization item
    item = parallelization_list[0]
    assert "zarr_url" in item
    assert "init_args" in item

    # Check init_args content
    init_args = item["init_args"]
    assert "zarr_urls_to_combine" in init_args
    assert "keep_individual_acquisitions" in init_args
    assert len(init_args["zarr_urls_to_combine"]) == 2
    assert init_args["keep_individual_acquisitions"] is True

    # Verify plate metadata was updated with new combined acquisition
    plate = open_ome_zarr_plate(test_zarr)
    acquisition_ids = plate.acquisition_ids
    # Should have acquisitions 0, 1, and the new combined one (2)
    assert len(acquisition_ids) >= 3


def test_combine_acquisitions_init_remove_originals(zarr_MIP_path, tmp_path):
    """Test combine_acquisitions_init with keep_individual_acquisitions=False."""
    # Create a new plate from scratch with two acquisitions
    test_zarr = tmp_path / "test_plate_remove.zarr"

    list_of_images = [
        ImageInWellPath(
            path="0", row="B", column=3, acquisition_id=0, acquisition_name="acq0"
        ),
        ImageInWellPath(
            path="1", row="B", column=3, acquisition_id=1, acquisition_name="acq1"
        ),
    ]

    create_empty_plate(
        store=str(test_zarr),
        name="test_plate",
        images=list_of_images,
        overwrite=True,
    )

    # Copy actual image data from fixture
    source_acq = zarr_MIP_path / "B" / "03" / "0"
    acq0_path = test_zarr / "B" / "03" / "0"
    acq1_path = test_zarr / "B" / "03" / "1"
    shutil.copytree(source_acq, acq0_path, dirs_exist_ok=True)
    shutil.copytree(source_acq, acq1_path, dirs_exist_ok=True)

    # Run the init task with keep_individual_acquisitions=False
    zarr_urls = [str(acq0_path), str(acq1_path)]
    result = combine_acquisitions_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        keep_individual_acquisitions=False,
    )

    # Check result structure
    assert isinstance(result, dict)
    assert "parallelization_list" in result
    parallelization_list = result["parallelization_list"]
    assert len(parallelization_list) == 1
    assert parallelization_list[0]["init_args"]["keep_individual_acquisitions"] is False

    # Verify plate metadata - original acquisitions should be removed from metadata
    plate = open_ome_zarr_plate(test_zarr)
    well_images = plate.well_images_paths(row="B", column=3)
    # With keep_individual_acquisitions=False, metadata should only have the new combined one
    assert len(well_images) == 1


def test_combine_acquisitions_init_single_acquisition(zarr_MIP_path, tmp_path):
    """Test that plates with only one acquisition are skipped."""
    # Create a new plate from scratch with only one acquisition
    test_zarr = tmp_path / "test_plate_single.zarr"

    list_of_images = [
        ImageInWellPath(
            path="0", row="B", column=3, acquisition_id=0, acquisition_name="acq0"
        ),
    ]

    create_empty_plate(
        store=str(test_zarr),
        name="test_plate",
        images=list_of_images,
        overwrite=True,
    )

    # Copy actual image data from fixture
    source_acq = zarr_MIP_path / "B" / "03" / "0"
    acq0_path = test_zarr / "B" / "03" / "0"
    shutil.copytree(source_acq, acq0_path, dirs_exist_ok=True)

    # Run the init task
    zarr_urls = [str(acq0_path)]
    result = combine_acquisitions_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        keep_individual_acquisitions=True,
    )

    # Should return empty list since there's only one acquisition
    assert isinstance(result, dict)
    assert "parallelization_list" in result
    parallelization_list = result["parallelization_list"]
    assert len(parallelization_list) == 0


def test_combine_acquisitions_init_multiple_wells(zarr_MIP_path, tmp_path):
    """Test combine_acquisitions_init with multiple wells having the same acquisitions."""
    # Create a new plate from scratch with two wells, each having two acquisitions
    test_zarr = tmp_path / "test_plate_multi_wells.zarr"

    list_of_images = [
        ImageInWellPath(
            path="0", row="B", column=3, acquisition_id=0, acquisition_name="acq0"
        ),
        ImageInWellPath(
            path="1", row="B", column=3, acquisition_id=1, acquisition_name="acq1"
        ),
        ImageInWellPath(
            path="0", row="C", column=3, acquisition_id=0, acquisition_name="acq0"
        ),
        ImageInWellPath(
            path="1", row="C", column=3, acquisition_id=1, acquisition_name="acq1"
        ),
    ]

    create_empty_plate(
        store=str(test_zarr),
        name="test_plate",
        images=list_of_images,
        overwrite=True,
    )

    # Copy actual image data from fixture to both wells
    source_acq = zarr_MIP_path / "B" / "03" / "0"
    for well in ["B", "C"]:
        acq0_path = test_zarr / well / "03" / "0"
        acq1_path = test_zarr / well / "03" / "1"
        shutil.copytree(source_acq, acq0_path, dirs_exist_ok=True)
        shutil.copytree(source_acq, acq1_path, dirs_exist_ok=True)

    # Prepare the zarr_urls for both wells
    zarr_urls = [
        str(test_zarr / "B" / "03" / "0"),
        str(test_zarr / "B" / "03" / "1"),
        str(test_zarr / "C" / "03" / "0"),
        str(test_zarr / "C" / "03" / "1"),
    ]

    # Run the init task
    result = combine_acquisitions_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        keep_individual_acquisitions=True,
    )

    # Should have parallelization items for both wells
    assert isinstance(result, dict)
    assert "parallelization_list" in result
    parallelization_list = result["parallelization_list"]
    assert len(parallelization_list) == 2, f"Expected 2 results (one per well), got {len(parallelization_list)}"

    # Check that both wells are represented
    zarr_urls_new = [item["zarr_url"] for item in parallelization_list]
    # Extract well rows from the URLs (B or C)
    # URL format is .../plate.zarr/B/03/2 or .../plate.zarr/B/3/2
    well_rows_in_results = [url.split("/")[-3] for url in zarr_urls_new]

    # Both wells should be in results
    assert "B" in well_rows_in_results, \
        f"Expected row B in results, got: {well_rows_in_results}"
    assert "C" in well_rows_in_results, \
        f"Expected row C in results, got: {well_rows_in_results}"

    # Verify each result has correct structure
    for item in parallelization_list:
        assert "zarr_url" in item
        assert "init_args" in item
        assert len(item["init_args"]["zarr_urls_to_combine"]) == 2
        assert item["init_args"]["keep_individual_acquisitions"] is True


def test_combine_acquisitions_init_specific_acquisitions(zarr_MIP_path, tmp_path):
    """Test combine_acquisitions_init with acquisitions_to_combine parameter."""
    # Create a new plate from scratch with three acquisitions
    test_zarr = tmp_path / "test_plate_three_acq.zarr"

    list_of_images = [
        ImageInWellPath(
            path="0", row="B", column=3, acquisition_id=0, acquisition_name="acq0"
        ),
        ImageInWellPath(
            path="1", row="B", column=3, acquisition_id=1, acquisition_name="acq1"
        ),
        ImageInWellPath(
            path="2", row="B", column=3, acquisition_id=2, acquisition_name="acq2"
        ),
    ]

    create_empty_plate(
        store=str(test_zarr),
        name="test_plate",
        images=list_of_images,
        overwrite=True,
    )

    # Copy actual image data from fixture
    source_acq = zarr_MIP_path / "B" / "03" / "0"
    acq0_b03 = test_zarr / "B" / "03" / "0"
    acq1_b03 = test_zarr / "B" / "03" / "1"
    acq2_b03 = test_zarr / "B" / "03" / "2"
    shutil.copytree(source_acq, acq0_b03, dirs_exist_ok=True)
    shutil.copytree(source_acq, acq1_b03, dirs_exist_ok=True)
    shutil.copytree(source_acq, acq2_b03, dirs_exist_ok=True)

    # Run the init task, combining only acquisitions 0 and 1 (not 2)
    zarr_urls = [str(acq0_b03), str(acq1_b03), str(acq2_b03)]
    result = combine_acquisitions_init(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        acquisitions_to_combine=[0, 1],
        keep_individual_acquisitions=True,
    )

    # Should have one parallelization item
    assert isinstance(result, dict)
    assert "parallelization_list" in result
    parallelization_list = result["parallelization_list"]
    assert len(parallelization_list) == 1

    # Check that only 2 acquisitions are being combined
    item = parallelization_list[0]
    init_args = item["init_args"]
    assert len(init_args["zarr_urls_to_combine"]) == 2

    # Verify that acquisition 2 is not in the list
    combined_urls = init_args["zarr_urls_to_combine"]
    assert not any("2" == url.split("/")[-1] for url in combined_urls)
    assert init_args["keep_individual_acquisitions"] is True

    # New acquisition should be ID 3 (next after 0, 1, 2)
    assert item["zarr_url"].endswith("/3")
