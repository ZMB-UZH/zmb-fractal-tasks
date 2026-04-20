import numpy as np
import pandas as pd
import pytest
from ngio import open_ome_zarr_container
from ngio.tables import FeatureTable

from zmb_fractal_tasks.measure_shortest_distance import (
    measure_shortest_distance,
    measure_shortest_distance_ROI,
)


def test_measure_shortest_distance(zarr_MIP_path):
    measure_shortest_distance(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        output_table_name="nuclei_features",
        input_label_name="nuclei",
        target_label_names=["wf_2_labels", "wf_3_labels"],
        pyramid_level="0",
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    # TODO: Check outputs


def test_measure_shortest_distance_ROI_empty_labels():
    """Test that measure_shortest_distance_ROI returns all expected columns for empty labels."""
    # Create empty labels (all zeros)
    labels = np.zeros((10, 100, 100), dtype=np.uint16)

    # Create dummy target label images
    target_label_list = [
        np.random.randint(0, 2, (10, 100, 100), dtype=np.uint16),
        np.random.randint(0, 2, (10, 100, 100), dtype=np.uint16),
    ]

    target_prefix_list = ["organoid", "well"]
    pxl_sizes = (1.0, 0.65, 0.65)
    optional_columns = {"plate": "test_plate", "well": "A01", "ROI": "0"}

    df = measure_shortest_distance_ROI(
        labels=labels,
        target_label_list=target_label_list,
        target_prefix_list=target_prefix_list,
        pxl_sizes=pxl_sizes,
        optional_columns=optional_columns,
    )

    # Check that dataframe is empty but has all expected columns
    assert len(df) == 0
    assert df.index.name == "label"

    # Check that all expected columns are present
    expected_columns = [
        "plate",
        "well",
        "ROI",  # optional columns
        "shortest_distance_to_organoid",
        "shortest_distance_to_well",  # distances
    ]
    assert list(df.columns) == expected_columns


def test_measure_shortest_distance_ROI_with_labels():
    """Test that measure_shortest_distance_ROI works correctly with non-empty labels."""
    # Create labels with some objects
    labels = np.zeros((5, 100, 100), dtype=np.uint16)
    labels[1:3, 10:20, 10:20] = 1
    labels[3:5, 50:60, 50:60] = 2

    # Create target labels
    target_label_list = [
        np.zeros((5, 100, 100), dtype=np.uint16),
    ]
    target_label_list[0][2:4, 80:90, 80:90] = 1  # A target object

    target_prefix_list = ["organoid"]
    pxl_sizes = (1.0, 0.65, 0.65)
    optional_columns = {"plate": "test_plate"}

    df = measure_shortest_distance_ROI(
        labels=labels,
        target_label_list=target_label_list,
        target_prefix_list=target_prefix_list,
        pxl_sizes=pxl_sizes,
        optional_columns=optional_columns,
    )

    # Check that dataframe has 2 rows (2 labels)
    assert len(df) == 2
    assert df.index.name == "label"
    assert list(df.index) == [1, 2]

    # Check that all expected columns are present
    expected_columns = [
        "plate",  # optional columns
        "shortest_distance_to_organoid",  # distances
    ]
    assert list(df.columns) == expected_columns

    # Check that distances are positive (not zero, since labels don't overlap with target)
    assert df.loc[1, "shortest_distance_to_organoid"] > 0
    assert df.loc[2, "shortest_distance_to_organoid"] > 0


def test_measure_shortest_distance_pyramid_level_none(zarr_MIP_path):
    """Test measure_shortest_distance with pyramid_level=None (default path)."""
    measure_shortest_distance(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        output_table_name="dist_pyr_none",
        input_label_name="nuclei",
        target_label_names=["wf_2_labels"],
        pyramid_level=None,
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )

    ome_zarr = open_ome_zarr_container(str(zarr_MIP_path / "B" / "03" / "0"))
    assert "dist_pyr_none" in ome_zarr.list_tables()
    df = ome_zarr.get_table("dist_pyr_none").dataframe
    assert "shortest_distance_to_wf_2_labels" in df.columns
    assert len(df) > 0


def test_measure_shortest_distance_3D(zarr_3D_path):
    """Test measure_shortest_distance on 3D data to cover 3D axes_order and pxl_sizes branches."""
    zarr_url = str(zarr_3D_path / "B" / "03" / "0")
    ome_zarr = open_ome_zarr_container(zarr_url)

    # Programmatically create 'nuclei' and target labels
    image = ome_zarr.get_image()
    nuclei_label = ome_zarr.derive_label("nuclei", ref_image=image)
    arr = nuclei_label.get_as_numpy(axes_order="zyx")
    arr[0, 10:20, 10:20] = 1
    arr[0, 50:60, 50:60] = 2
    nuclei_label.set_array(arr, axes_order="zyx")

    target_label = ome_zarr.derive_label("wf_2_labels", ref_image=image)
    arr_t = target_label.get_as_numpy(axes_order="zyx")
    arr_t[0, 80:100, 80:100] = 1
    target_label.set_array(arr_t, axes_order="zyx")

    measure_shortest_distance(
        zarr_url=zarr_url,
        output_table_name="dist_3D",
        input_label_name="nuclei",
        target_label_names=["wf_2_labels"],
        pyramid_level="0",
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )

    ome_zarr = open_ome_zarr_container(zarr_url)
    assert "dist_3D" in ome_zarr.list_tables()
    df = ome_zarr.get_table("dist_3D").dataframe
    assert "shortest_distance_to_wf_2_labels" in df.columns
    assert len(df) > 0


def test_measure_shortest_distance_append(zarr_MIP_path):
    """Test appending distance measurements to an existing table."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")

    # First run: write the table fresh
    measure_shortest_distance(
        zarr_url=zarr_url,
        output_table_name="dist_append",
        input_label_name="nuclei",
        target_label_names=["wf_2_labels"],
        pyramid_level="0",
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )

    # Second run: append a new distance column
    measure_shortest_distance(
        zarr_url=zarr_url,
        output_table_name="dist_append",
        input_label_name="nuclei",
        target_label_names=["wf_3_labels"],
        pyramid_level="0",
        roi_table="FOV_ROI_table",
        append_to_table=True,
    )

    ome_zarr = open_ome_zarr_container(zarr_url)
    df = ome_zarr.get_table("dist_append").dataframe
    assert "shortest_distance_to_wf_2_labels" in df.columns
    assert "shortest_distance_to_wf_3_labels" in df.columns


def test_measure_shortest_distance_raises_on_mismatch(zarr_MIP_path):
    """Test that appending to a table with a mismatched index raises ValueError."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")

    # First run: create table with real nuclei indices
    measure_shortest_distance(
        zarr_url=zarr_url,
        output_table_name="dist_mismatch",
        input_label_name="nuclei",
        target_label_names=["wf_2_labels"],
        pyramid_level="0",
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )

    # Overwrite with a table with shifted (wrong) indices
    ome_zarr = open_ome_zarr_container(zarr_url)
    df_real = ome_zarr.get_table("dist_mismatch").dataframe
    df_wrong = df_real.copy()
    df_wrong.index = df_wrong.index + 99999
    ome_zarr.add_table(
        "dist_mismatch",
        FeatureTable(df_wrong, reference_label="nuclei"),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="Index mismatch"):
        measure_shortest_distance(
            zarr_url=zarr_url,
            output_table_name="dist_mismatch",
            input_label_name="nuclei",
            target_label_names=["wf_2_labels"],
            pyramid_level="0",
            roi_table="FOV_ROI_table",
            append_to_table=True,
        )


def test_measure_shortest_distance_ROI_default_prefix():
    """Test measure_shortest_distance_ROI with no target_prefix_list (uses defaults)."""
    labels = np.zeros((5, 100, 100), dtype=np.uint16)
    labels[1:3, 10:20, 10:20] = 1

    target_label_list = [np.zeros((5, 100, 100), dtype=np.uint16)]
    target_label_list[0][2:4, 80:90, 80:90] = 1

    # No target_prefix_list → defaults to ["dist0"]
    df = measure_shortest_distance_ROI(
        labels=labels,
        target_label_list=target_label_list,
        pxl_sizes=(1.0, 0.65, 0.65),
    )

    assert len(df) == 1
    assert "shortest_distance_to_dist0" in df.columns


def test_measure_shortest_distance_ROI_no_optional_columns():
    """Test measure_shortest_distance_ROI with optional_columns=None (uses default {})."""
    labels = np.zeros((5, 100, 100), dtype=np.uint16)
    labels[1:3, 10:20, 10:20] = 1

    target_label_list = [np.zeros((5, 100, 100), dtype=np.uint16)]
    target_label_list[0][2:4, 80:90, 80:90] = 1

    # No optional_columns → defaults to {}
    df = measure_shortest_distance_ROI(
        labels=labels,
        target_label_list=target_label_list,
        target_prefix_list=["target"],
        pxl_sizes=(1.0, 0.65, 0.65),
        optional_columns=None,
    )

    assert len(df) == 1
    assert "shortest_distance_to_target" in df.columns
    assert "plate" not in df.columns


def test_measure_shortest_distance_ROI_2D():
    """Test measure_shortest_distance_ROI with 2D (single-plane) label arrays."""
    labels = np.zeros((100, 100), dtype=np.uint16)
    labels[10:20, 10:20] = 1
    labels[50:60, 50:60] = 2

    target = np.zeros((100, 100), dtype=np.uint16)
    target[80:90, 80:90] = 1

    df = measure_shortest_distance_ROI(
        labels=labels,
        target_label_list=[target],
        target_prefix_list=["far_target"],
        pxl_sizes=(0.65, 0.65),
    )

    assert len(df) == 2
    assert "shortest_distance_to_far_target" in df.columns
    assert (df["shortest_distance_to_far_target"] > 0).all()
