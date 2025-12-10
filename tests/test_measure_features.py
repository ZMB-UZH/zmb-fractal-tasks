import numpy as np

from zmb_fractal_tasks.from_fractal_tasks_core.channels import ChannelInputModel
from zmb_fractal_tasks.measure_features import measure_features, measure_features_ROI


def test_measure_features(zarr_MIP_path):
    measure_features(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        output_table_name="nuclei_features",
        input_label_name="nuclei",
        channels_to_include=[ChannelInputModel(label="DAPI")],
        channels_to_exclude=None,
        structure_props=["area"],
        intensity_props=["intensity_total"],
        pyramid_level="0",
        roi_table_name="FOV_ROI_table",
        append=False,
        overwrite=True,
    )
    # TODO: Check outputs


def test_measure_features_ROI_empty_labels():
    """Test that measure_features_ROI returns all expected columns for empty labels."""
    # Create empty labels (all zeros)
    labels = np.zeros((10, 100, 100), dtype=np.uint16)

    # Create dummy intensity images
    intensities_list = [
        np.random.randint(0, 255, (10, 100, 100), dtype=np.uint16),
        np.random.randint(0, 255, (10, 100, 100), dtype=np.uint16),
    ]

    int_prefix_list = ["A01_C01", "A01_C02"]
    structure_props = ["area", "num_pixels"]
    intensity_props = ["intensity_mean", "intensity_std", "intensity_total"]
    optional_columns = {"plate": "test_plate", "well": "A01", "ROI": "0"}

    df = measure_features_ROI(
        labels=labels,
        intensities_list=intensities_list,
        int_prefix_list=int_prefix_list,
        structure_props=structure_props,
        intensity_props=intensity_props,
        pxl_sizes=(1.0, 0.65, 0.65),
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
        "area",
        "num_pixels",  # structure props
        "A01_C01_intensity_mean",
        "A01_C01_intensity_std",
        "A01_C01_intensity_total",
        "A01_C02_intensity_mean",
        "A01_C02_intensity_std",
        "A01_C02_intensity_total",
    ]
    assert list(df.columns) == expected_columns


def test_measure_features_ROI_with_labels():
    """Test that measure_features_ROI works correctly with non-empty labels."""
    # Create labels with some objects
    labels = np.zeros((10, 100, 100), dtype=np.uint16)
    labels[2:5, 10:20, 10:20] = 1
    labels[6:9, 50:60, 50:60] = 2

    # Create dummy intensity images
    intensities_list = [
        np.random.randint(0, 255, (10, 100, 100), dtype=np.uint16),
    ]

    int_prefix_list = ["A01_C01"]
    structure_props = ["area"]
    intensity_props = ["intensity_mean"]
    optional_columns = {"plate": "test_plate", "well": "A01"}

    df = measure_features_ROI(
        labels=labels,
        intensities_list=intensities_list,
        int_prefix_list=int_prefix_list,
        structure_props=structure_props,
        intensity_props=intensity_props,
        pxl_sizes=(1.0, 0.65, 0.65),
        optional_columns=optional_columns,
    )

    # Check that dataframe has 2 rows (2 labels)
    assert len(df) == 2
    assert df.index.name == "label"
    assert list(df.index) == [1, 2]

    # Check that all expected columns are present
    expected_columns = [
        "plate",
        "well",  # optional columns
        "area",  # structure props
        "A01_C01_intensity_mean",  # intensity props
    ]
    assert list(df.columns) == expected_columns
