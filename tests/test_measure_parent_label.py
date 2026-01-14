import numpy as np

from zmb_fractal_tasks.assign_to_parent_label import (
    measure_parent_label,
    measure_parent_ROI,
)


def test_measure_parent_label(zarr_MIP_path):
    measure_parent_label(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        output_table_name="nuclei_features",
        input_label_name="nuclei",
        parent_label_names=["wf_2_labels", "wf_3_labels"],
        pyramid_level="0",
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    # TODO: Check outputs


def test_measure_parent_ROI_empty_labels():
    """Test that measure_parent_ROI returns all expected columns for empty labels."""
    # Create empty labels (all zeros)
    labels = np.zeros((10, 100, 100), dtype=np.uint16)

    # Create dummy parent label image
    parent_labels = np.random.randint(0, 5, (10, 100, 100), dtype=np.uint16)

    parent_prefix = "organoid"
    optional_columns = {"plate": "test_plate", "well": "A01", "ROI": "0"}

    df = measure_parent_ROI(
        labels=labels,
        parent_labels=parent_labels,
        parent_prefix=parent_prefix,
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
        "organoid_ID",  # parent ID
    ]
    assert list(df.columns) == expected_columns


def test_measure_parent_ROI_with_labels():
    """Test that measure_parent_ROI works correctly with non-empty labels."""
    # Create labels with some objects
    labels = np.zeros((5, 100, 100), dtype=np.uint16)
    labels[1:3, 10:20, 10:20] = 1
    labels[3:5, 50:60, 50:60] = 2

    # Create parent labels
    parent_labels = np.ones((5, 100, 100), dtype=np.uint16) * 10

    parent_prefix = "organoid"
    optional_columns = {"plate": "test_plate"}

    df = measure_parent_ROI(
        labels=labels,
        parent_labels=parent_labels,
        parent_prefix=parent_prefix,
        optional_columns=optional_columns,
    )

    # Check that dataframe has 2 rows (2 labels)
    assert len(df) == 2
    assert df.index.name == "label"
    assert list(df.index) == [1, 2]

    # Check that all expected columns are present
    expected_columns = [
        "plate",  # optional columns
        "organoid_ID",  # parent IDs
    ]
    assert list(df.columns) == expected_columns

    # Check that parent IDs are correct
    assert df.loc[1, "organoid_ID"] == 10
    assert df.loc[2, "organoid_ID"] == 10
