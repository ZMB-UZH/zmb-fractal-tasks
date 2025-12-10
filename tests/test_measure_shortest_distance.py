import numpy as np

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
        roi_table_name="FOV_ROI_table",
        append=False,
        overwrite=True,
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
