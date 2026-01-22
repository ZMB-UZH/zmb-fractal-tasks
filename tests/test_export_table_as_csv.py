import shutil
from pathlib import Path

import pandas as pd
import pytest
from ngio import open_ome_zarr_container

from zmb_fractal_tasks.export_table_as_csv import export_table_as_csv
from zmb_fractal_tasks.measure_features import LabelInput, measure_features
from zmb_fractal_tasks.utils.channel_utils import MeasurementChannels


@pytest.fixture
def zarr_with_measurements(zarr_MIP_path, tmp_path):
    """Create a zarr with measurement tables for testing."""
    test_zarr = tmp_path / "test_export_zarr.zarr"
    shutil.copytree(zarr_MIP_path, test_zarr, dirs_exist_ok=True)
    
    # Generate measurement tables (only B/03 exists in test data)
    zarr_url = str(test_zarr / "B" / "03" / "0")
    measure_features(
        zarr_url=zarr_url,
        input_labels=[
            LabelInput(
                input_label_name="nuclei",
                output_table_name="nuclei_measurements"
            )
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
    
    return test_zarr


def test_export_table_basic(zarr_with_measurements, tmp_path):
    """Test basic table export without plate layout."""
    zarr_dir = str(tmp_path / "output")
    Path(zarr_dir).mkdir(exist_ok=True)
    
    zarr_urls = [
        str(zarr_with_measurements / "B" / "03" / "0"),
    ]
    
    export_table_as_csv(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        table_to_export="nuclei_measurements",
    )
    
    # Verify CSV file was created
    csv_path = Path(zarr_dir) / "nuclei_measurements.csv"
    assert csv_path.exists()
    
    # Load and verify content
    df = pd.read_csv(csv_path, index_col=0)
    
    # Check that plate and well columns exist and are at the front
    assert "plate" in df.columns
    assert "well" in df.columns
    assert df.columns[0] == "plate"
    assert df.columns[1] == "well"
    
    # Check that we have data from the well
    assert "B03" in df["well"].values
    
    # Check that measurement columns exist
    assert "area" in df.columns
    assert "DAPI_intensity_mean" in df.columns
    
    # Verify data is present
    assert len(df) > 0


def test_export_table_custom_name(zarr_with_measurements, tmp_path):
    """Test exporting table with custom output name."""
    zarr_dir = str(tmp_path / "output")
    Path(zarr_dir).mkdir(exist_ok=True)
    
    zarr_urls = [
        str(zarr_with_measurements / "B" / "03" / "0"),
    ]
    
    export_table_as_csv(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        table_to_export="nuclei_measurements",
        export_table_name="custom_export_name",
    )
    
    # Verify CSV file was created with custom name
    csv_path = Path(zarr_dir) / "custom_export_name.csv"
    assert csv_path.exists()
    
    df = pd.read_csv(csv_path, index_col=0)
    assert len(df) > 0


def test_export_table_with_plate_layout(zarr_with_measurements, tmp_path):
    """Test table export with plate layout information."""
    zarr_dir = str(tmp_path / "output")
    Path(zarr_dir).mkdir(exist_ok=True)
    
    # Create a simple plate layout CSV with zero-padded column names
    plate_layout_path = tmp_path / "plate_layout.csv"
    plate_layout = pd.DataFrame(
        {
            "01": ["NA", "NA"],
            "02": ["NA", "NA"],
            "03": ["control", "control"],
            "04": ["treatment", "treatment"],
        },
        index=["A", "B"],
    )
    plate_layout.to_csv(plate_layout_path)
    
    zarr_urls = [
        str(zarr_with_measurements / "B" / "03" / "0"),
    ]
    
    export_table_as_csv(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        table_to_export="nuclei_measurements",
        plate_layout_path=str(plate_layout_path),
    )
    
    # Verify CSV file was created
    csv_path = Path(zarr_dir) / "nuclei_measurements.csv"
    assert csv_path.exists()
    
    # Load and verify content
    df = pd.read_csv(csv_path, index_col=0)
    
    # Check that condition column exists and is positioned correctly
    assert "condition" in df.columns
    assert df.columns[0] == "plate"
    assert df.columns[1] == "well"
    assert df.columns[2] == "condition"
    
    # Check condition values - B03 should map to "control"
    assert "control" in df["condition"].values
    
    # Verify correct mapping
    df_B03 = df[df["well"] == "B03"]
    assert (df_B03["condition"] == "control").all()


def test_export_table_single_well(zarr_with_measurements, tmp_path):
    """Test exporting table from a single well."""
    zarr_dir = str(tmp_path / "output")
    Path(zarr_dir).mkdir(exist_ok=True)
    
    zarr_urls = [
        str(zarr_with_measurements / "B" / "03" / "0"),
    ]
    
    export_table_as_csv(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        table_to_export="nuclei_measurements",
    )
    
    csv_path = Path(zarr_dir) / "nuclei_measurements.csv"
    assert csv_path.exists()
    
    df = pd.read_csv(csv_path, index_col=0)
    assert len(df["well"].unique()) == 1
    assert df["well"].iloc[0] == "B03"


def test_export_table_plate_name_parsing(zarr_with_measurements, tmp_path):
    """Test that plate name is correctly extracted from zarr path."""
    zarr_dir = str(tmp_path / "output")
    Path(zarr_dir).mkdir(exist_ok=True)
    
    zarr_urls = [
        str(zarr_with_measurements / "B" / "03" / "0"),
    ]
    
    export_table_as_csv(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        table_to_export="nuclei_measurements",
    )
    
    csv_path = Path(zarr_dir) / "nuclei_measurements.csv"
    df = pd.read_csv(csv_path, index_col=0)
    
    # Check that plate name is extracted correctly
    # Should be the stem of the zarr file
    expected_plate_name = zarr_with_measurements.stem
    assert df["plate"].iloc[0] == expected_plate_name
