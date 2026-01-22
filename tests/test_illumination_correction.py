from pathlib import Path

import numpy as np
import tifffile
from ngio import open_ome_zarr_container

from zmb_fractal_tasks.illumination_correction import illumination_correction


def test_illumination_correction_overwrite(zarr_path, tmpdir):
    """Test illumination correction with overwrite_input_image=True."""
    # Create illumination profiles folder
    illumination_folder = Path(tmpdir) / "illumination_profiles"
    illumination_folder.mkdir(parents=True)

    # Open the zarr to get image dimensions
    omezarr = open_ome_zarr_container(str(zarr_path / "B" / "03" / "0"))
    source_image = omezarr.get_image()
    roi_table = omezarr.get_table("FOV_ROI_table")

    # Get first ROI to determine shape
    first_roi = roi_table.rois()[0]
    patch = source_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    y_size, x_size = patch.shape[2:]

    # Create dummy illumination profiles for each channel
    channels = source_image.wavelength_ids
    illumination_profiles = {}

    for channel in channels:
        # Create a simple gradient flatfield (bright in center, dim at edges)
        y, x = np.ogrid[:y_size, :x_size]
        center_y, center_x = y_size // 2, x_size // 2
        flatfield = 1.0 - 0.3 * np.sqrt(
            ((y - center_y) / y_size) ** 2 + ((x - center_x) / x_size) ** 2
        )
        flatfield = flatfield.astype(np.float32)

        # Save as tif file
        profile_filename = f"{channel}_flatfield.tif"
        tifffile.imwrite(illumination_folder / profile_filename, flatfield)
        illumination_profiles[channel] = profile_filename

    # Get original data to compare
    original_data = source_image.get_roi(
        first_roi, c=0, axes_order=["c", "z", "y", "x"]
    ).copy()

    # Run illumination correction with overwrite
    result = illumination_correction(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        illumination_profiles_folder=str(illumination_folder),
        illumination_profiles=illumination_profiles,
        background=0,
        input_ROI_table="FOV_ROI_table",
        overwrite_input_image=True,
    )

    # Check result structure
    assert "image_list_updates" in result
    assert len(result["image_list_updates"]) == 1
    assert result["image_list_updates"][0]["zarr_url"] == str(
        zarr_path / "B" / "03" / "0"
    )

    # Check that data was modified
    corrected_data = source_image.get_roi(
        first_roi, c=0, axes_order=["c", "z", "y", "x"]
    )
    assert not np.array_equal(original_data, corrected_data), (
        "Data should be modified after correction"
    )

    # Check that correction was applied (corrected values should differ from original)
    assert corrected_data.shape == original_data.shape


def test_illumination_correction_no_overwrite(zarr_path, tmpdir):
    """Test illumination correction with overwrite_input_image=False."""
    # Create illumination profiles folder
    illumination_folder = Path(tmpdir) / "illumination_profiles"
    illumination_folder.mkdir(parents=True)

    # Open the zarr to get image dimensions
    omezarr = open_ome_zarr_container(str(zarr_path / "B" / "03" / "0"))
    source_image = omezarr.get_image()
    roi_table = omezarr.get_table("FOV_ROI_table")

    # Get first ROI to determine shape
    first_roi = roi_table.rois()[0]
    patch = source_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    y_size, x_size = patch.shape[2:]

    # Create dummy illumination profiles
    channels = source_image.wavelength_ids
    illumination_profiles = {}

    for channel in channels:
        # Create uniform flatfield
        flatfield = np.ones((y_size, x_size), dtype=np.float32)

        profile_filename = f"{channel}_flatfield.tif"
        tifffile.imwrite(illumination_folder / profile_filename, flatfield)
        illumination_profiles[channel] = profile_filename

    # Get original data
    original_data = source_image.get_roi(
        first_roi, c=0, axes_order=["c", "z", "y", "x"]
    ).copy()

    # Run illumination correction without overwrite
    suffix = "corrected"
    result = illumination_correction(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        illumination_profiles_folder=str(illumination_folder),
        illumination_profiles=illumination_profiles,
        background=0,
        input_ROI_table="FOV_ROI_table",
        overwrite_input_image=False,
        new_well_subgroup_suffix=suffix,
    )

    # Check result structure
    assert "image_list_updates" in result
    assert len(result["image_list_updates"]) == 1
    new_zarr_url = result["image_list_updates"][0]["zarr_url"]
    assert suffix in str(new_zarr_url)
    assert result["image_list_updates"][0]["origin"] == str(
        zarr_path / "B" / "03" / "0"
    )

    # Check that original data was not modified
    current_data = source_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    assert np.array_equal(original_data, current_data), (
        "Original data should not be modified"
    )

    # Check that new zarr was created
    assert Path(new_zarr_url).exists()

    # Check new zarr has corrected data
    new_omezarr = open_ome_zarr_container(new_zarr_url)
    new_image = new_omezarr.get_image()
    corrected_data = new_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    assert corrected_data.shape == original_data.shape

    # Check tables were copied
    assert "FOV_ROI_table" in new_omezarr.list_tables()


def test_illumination_correction_with_background(zarr_path, tmpdir):
    """Test illumination correction with background subtraction."""
    # Create illumination profiles folder
    illumination_folder = Path(tmpdir) / "illumination_profiles"
    illumination_folder.mkdir(parents=True)

    # Open the zarr
    omezarr = open_ome_zarr_container(str(zarr_path / "B" / "03" / "0"))
    source_image = omezarr.get_image()
    roi_table = omezarr.get_table("FOV_ROI_table")

    # Get first ROI to determine shape
    first_roi = roi_table.rois()[0]
    patch = source_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    y_size, x_size = patch.shape[2:]

    # Create dummy illumination profiles
    channels = source_image.wavelength_ids
    illumination_profiles = {}

    for channel in channels:
        flatfield = np.ones((y_size, x_size), dtype=np.float32)

        profile_filename = f"{channel}_flatfield.tif"
        tifffile.imwrite(illumination_folder / profile_filename, flatfield)
        illumination_profiles[channel] = profile_filename

    # Get original data
    original_data = source_image.get_roi(
        first_roi, c=0, axes_order=["c", "z", "y", "x"]
    ).copy()

    # Run with background subtraction
    background_value = 100
    result = illumination_correction(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        illumination_profiles_folder=str(illumination_folder),
        illumination_profiles=illumination_profiles,
        background=background_value,
        input_ROI_table="FOV_ROI_table",
        overwrite_input_image=True,
    )

    # Check that correction was applied
    corrected_data = source_image.get_roi(
        first_roi, c=0, axes_order=["c", "z", "y", "x"]
    )

    # With uniform flatfield and background subtraction,
    # corrected values should be roughly original - background (where original > background)
    # This is a simple check that background subtraction was applied
    assert corrected_data.max() < original_data.max()


def test_illumination_correction_multiple_channels(zarr_path, tmpdir):
    """Test illumination correction with multiple channels."""
    # Create illumination profiles folder
    illumination_folder = Path(tmpdir) / "illumination_profiles"
    illumination_folder.mkdir(parents=True)

    # Open the zarr
    omezarr = open_ome_zarr_container(str(zarr_path / "B" / "03" / "0"))
    source_image = omezarr.get_image()
    roi_table = omezarr.get_table("FOV_ROI_table")

    # Get first ROI to determine shape
    first_roi = roi_table.rois()[0]
    patch = source_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    y_size, x_size = patch.shape[2:]

    # Get all channels
    channels = source_image.wavelength_ids
    illumination_profiles = {}

    # Create different profiles for each channel
    for idx, channel in enumerate(channels):
        # Create flatfield with noticeable variation (0.5 to 1.5 range)
        # This ensures the correction produces measurable changes
        y, x = np.ogrid[:y_size, :x_size]
        center_y, center_x = y_size // 2, x_size // 2
        flatfield = 0.7 + 0.3 * (
            1 - np.sqrt(((y - center_y) / y_size) ** 2 + ((x - center_x) / x_size) ** 2)
        )
        flatfield = flatfield.astype(np.float32)

        profile_filename = f"{channel}_flatfield.tif"
        tifffile.imwrite(illumination_folder / profile_filename, flatfield)
        illumination_profiles[channel] = profile_filename

    # Store original data for all channels
    original_data_per_channel = []
    for c_idx in range(len(channels)):
        original_data_per_channel.append(
            source_image.get_roi(
                first_roi, c=c_idx, axes_order=["c", "z", "y", "x"]
            ).copy()
        )

    # Run illumination correction
    result = illumination_correction(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        illumination_profiles_folder=str(illumination_folder),
        illumination_profiles=illumination_profiles,
        background=0,
        input_ROI_table="FOV_ROI_table",
        overwrite_input_image=True,
    )

    # Check all channels were corrected
    for c_idx in range(len(channels)):
        corrected_data = source_image.get_roi(
            first_roi, c=c_idx, axes_order=["c", "z", "y", "x"]
        )
        original_data = original_data_per_channel[c_idx]

        # Each channel should be modified
        assert not np.array_equal(original_data, corrected_data)


def test_illumination_correction_multiple_rois(zarr_path, tmpdir):
    """Test that illumination correction is applied to all ROIs."""
    # Create illumination profiles folder
    illumination_folder = Path(tmpdir) / "illumination_profiles"
    illumination_folder.mkdir(parents=True)

    # Open the zarr
    omezarr = open_ome_zarr_container(str(zarr_path / "B" / "03" / "0"))
    source_image = omezarr.get_image()
    roi_table = omezarr.get_table("FOV_ROI_table")

    # Get first ROI to determine shape
    first_roi = roi_table.rois()[0]
    patch = source_image.get_roi(first_roi, c=0, axes_order=["c", "z", "y", "x"])
    y_size, x_size = patch.shape[2:]

    # Create dummy illumination profiles
    channels = source_image.wavelength_ids
    illumination_profiles = {}

    for channel in channels:
        flatfield = np.ones((y_size, x_size), dtype=np.float32) * 0.8

        profile_filename = f"{channel}_flatfield.tif"
        tifffile.imwrite(illumination_folder / profile_filename, flatfield)
        illumination_profiles[channel] = profile_filename

    # Store original data for all ROIs
    original_data_per_roi = []
    roi_list = roi_table.rois()
    for roi in roi_list:
        original_data_per_roi.append(
            source_image.get_roi(roi, c=0, axes_order=["c", "z", "y", "x"]).copy()
        )

    # Run illumination correction
    result = illumination_correction(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        illumination_profiles_folder=str(illumination_folder),
        illumination_profiles=illumination_profiles,
        background=0,
        input_ROI_table="FOV_ROI_table",
        overwrite_input_image=True,
    )

    # Check all ROIs were corrected
    for idx, roi in enumerate(roi_list):
        corrected_data = source_image.get_roi(roi, c=0, axes_order=["c", "z", "y", "x"])
        original_data = original_data_per_roi[idx]

        # Each ROI should be modified
        assert not np.array_equal(original_data, corrected_data), (
            f"ROI {idx} should be modified"
        )
