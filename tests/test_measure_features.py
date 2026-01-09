import numpy as np
from ngio import open_ome_zarr_container

from zmb_fractal_tasks.measure_features import (
    LabelInput,
    measure_features,
    measure_features_ROI,
)
from zmb_fractal_tasks.utils.channel_utils import MeasurementChannels


def test_measure_features_single_channel(zarr_MIP_path):
    """Test measure_features with a single channel."""
    measure_features(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        input_labels=[
            LabelInput(input_label_name="nuclei", output_table_name="nuclei_features")
        ],
        channels_to_measure=MeasurementChannels(
            use_all_channels=False,
            mode="label",
            identifiers=["DAPI"],
        ),
        structure_props=["area"],
        intensity_props=["intensity_total"],
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    
    # Verify the table was created
    ome_zarr = open_ome_zarr_container(str(zarr_MIP_path / "B" / "03" / "0"))
    assert "nuclei_features" in ome_zarr.list_tables()
    table = ome_zarr.get_table("nuclei_features")
    df = table.dataframe
    
    # Check that expected columns exist
    assert "area" in df.columns
    assert "DAPI_intensity_total" in df.columns
    assert len(df) > 0  # Should have some measurements


def test_measure_features_all_channels(zarr_MIP_path):
    """Test measure_features with all channels."""
    measure_features(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        input_labels=[
            LabelInput(input_label_name="nuclei", output_table_name="nuclei_all_ch")
        ],
        channels_to_measure=MeasurementChannels(use_all_channels=True),
        structure_props=["area"],
        intensity_props=["intensity_mean"],
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    
    # Verify the table was created
    ome_zarr = open_ome_zarr_container(str(zarr_MIP_path / "B" / "03" / "0"))
    assert "nuclei_all_ch" in ome_zarr.list_tables()
    table = ome_zarr.get_table("nuclei_all_ch")
    df = table.dataframe
    
    # Check that all channels are measured
    assert "DAPI_intensity_mean" in df.columns
    assert "nanog_intensity_mean" in df.columns
    assert "Lamin B1_intensity_mean" in df.columns


def test_measure_features_multiple_channels(zarr_MIP_path):
    """Test measure_features with multiple channels specified."""
    measure_features(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        input_labels=[
            LabelInput(input_label_name="nuclei", output_table_name="nuclei_multi_ch")
        ],
        channels_to_measure=MeasurementChannels(
            use_all_channels=False,
            mode="label",
            identifiers=["DAPI", "nanog"],
        ),
        structure_props=["area"],
        intensity_props=["intensity_mean", "intensity_std"],
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    
    # Verify the table was created
    ome_zarr = open_ome_zarr_container(str(zarr_MIP_path / "B" / "03" / "0"))
    table = ome_zarr.get_table("nuclei_multi_ch")
    df = table.dataframe
    
    # Check that both channels are measured
    assert "DAPI_intensity_mean" in df.columns
    assert "DAPI_intensity_std" in df.columns
    assert "nanog_intensity_mean" in df.columns
    assert "nanog_intensity_std" in df.columns


def test_measure_features_multiple_labels(zarr_MIP_path):
    """Test measure_features with multiple labels."""
    measure_features(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        input_labels=[
            LabelInput(input_label_name="nuclei", output_table_name="nuclei_features"),
            LabelInput(
                input_label_name="wf_2_labels", output_table_name="wf_2_features"
            ),
        ],
        channels_to_measure=MeasurementChannels(
            use_all_channels=False,
            mode="label",
            identifiers=["DAPI"],
        ),
        structure_props=["area"],
        intensity_props=["intensity_mean"],
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    
    # Verify both tables were created
    ome_zarr = open_ome_zarr_container(str(zarr_MIP_path / "B" / "03" / "0"))
    assert "nuclei_features" in ome_zarr.list_tables()
    assert "wf_2_features" in ome_zarr.list_tables()


def test_measure_features_append_to_table(zarr_MIP_path):
    """Test appending measurements to existing table."""
    zarr_url = str(zarr_MIP_path / "B" / "03" / "0")
    
    # First measurement with one channel
    measure_features(
        zarr_url=zarr_url,
        input_labels=[
            LabelInput(input_label_name="nuclei", output_table_name="nuclei_append")
        ],
        channels_to_measure=MeasurementChannels(
            use_all_channels=False,
            mode="label",
            identifiers=["DAPI"],
        ),
        structure_props=["area"],
        intensity_props=["intensity_mean"],
        roi_table="FOV_ROI_table",
        append_to_table=False,
    )
    
    # Second measurement with another channel (append)
    measure_features(
        zarr_url=zarr_url,
        input_labels=[
            LabelInput(input_label_name="nuclei", output_table_name="nuclei_append")
        ],
        channels_to_measure=MeasurementChannels(
            use_all_channels=False,
            mode="label",
            identifiers=["nanog"],
        ),
        structure_props=["area"],
        intensity_props=["intensity_mean"],
        roi_table="FOV_ROI_table",
        append_to_table=True,
    )
    
    # Verify both channels are in the table
    ome_zarr = open_ome_zarr_container(zarr_url)
    table = ome_zarr.get_table("nuclei_append")
    df = table.dataframe
    
    assert "DAPI_intensity_mean" in df.columns
    assert "nanog_intensity_mean" in df.columns
    assert "area" in df.columns


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

    expected_columns = [
        "plate",
        "well",  # optional columns
        "area",  # structure props
        "A01_C01_intensity_mean",  # intensity props
    ]
    assert list(df.columns) == expected_columns


def test_measure_features_ROI_3D_empty_labels():
    """Test that measure_features_ROI works with empty 3D labels."""
    # Create empty 3D labels (all zeros)
    labels = np.zeros((5, 50, 50), dtype=np.uint16)

    # Create dummy intensity images
    intensities_list = [
        np.random.randint(0, 255, (5, 50, 50), dtype=np.uint16),
    ]

    int_prefix_list = ["DAPI"]
    structure_props = ["area"]
    intensity_props = ["intensity_mean"]
    optional_columns = {"plate": "test_plate", "well": "B03", "ROI": "0"}

    df = measure_features_ROI(
        labels=labels,
        intensities_list=intensities_list,
        int_prefix_list=int_prefix_list,
        structure_props=structure_props,
        intensity_props=intensity_props,
        pxl_sizes=(0.1625, 0.1625, 1.0),
        optional_columns=optional_columns,
    )

    # Check that dataframe is empty but has all expected columns
    assert len(df) == 0
    assert df.index.name == "label"
    
    expected_columns = [
        "plate",
        "well",
        "ROI",
        "area",
        "DAPI_intensity_mean",
    ]
    assert list(df.columns) == expected_columns


def test_measure_features_ROI_3D_with_labels():
    """Test that measure_features_ROI works correctly with 3D labels."""
    # Create 3D labels with some objects
    labels = np.zeros((5, 50, 50), dtype=np.uint16)
    labels[1:3, 10:20, 10:20] = 1
    labels[3:5, 30:40, 30:40] = 2

    # Create dummy intensity images
    intensities_list = [
        np.random.randint(0, 255, (5, 50, 50), dtype=np.uint16),
    ]

    int_prefix_list = ["DAPI"]
    structure_props = ["area"]
    intensity_props = ["intensity_mean", "intensity_std"]
    optional_columns = {"plate": "test_plate", "well": "B03"}

    df = measure_features_ROI(
        labels=labels,
        intensities_list=intensities_list,
        int_prefix_list=int_prefix_list,
        structure_props=structure_props,
        intensity_props=intensity_props,
        pxl_sizes=(0.1625, 0.1625, 1.0),
        optional_columns=optional_columns,
    )

    # Check that dataframe has 2 rows (2 labels)
    assert len(df) == 2
    assert df.index.name == "label"
    assert list(df.index) == [1, 2]

    # Check that all expected columns are present
    expected_columns = [
        "plate",
        "well",
        "area",
        "DAPI_intensity_mean",
        "DAPI_intensity_std",
    ]
    assert list(df.columns) == expected_columns
