import shutil
from pathlib import Path

import zarr

from zmb_fractal_tasks.delete_labels import delete_labels


def test_delete_labels(zarr_MIP_path, tmp_path):
    """Test deleting labels from an OME-Zarr image."""
    # Create a copy of the zarr to avoid modifying the fixture
    test_zarr = tmp_path / "test_zarr.zarr"
    shutil.copytree(zarr_MIP_path, test_zarr, dirs_exist_ok=True)
    
    zarr_url = str(test_zarr / "B" / "03" / "0")
    
    # Check initial state
    image_group = zarr.group(zarr_url)
    labels_group = image_group["labels"]
    initial_labels = labels_group.attrs.asdict().get("labels", [])
    
    # Ensure we have at least one label to delete
    assert len(initial_labels) > 0, "Test requires at least one label in the fixture"
    
    # Delete first label
    label_to_delete = initial_labels[0]
    delete_labels(
        zarr_url=zarr_url,
        labels_to_delete=[label_to_delete],
    )
    
    # Verify label was removed from metadata
    labels_group = zarr.group(zarr_url)["labels"]
    remaining_labels = labels_group.attrs.asdict().get("labels", [])
    assert label_to_delete not in remaining_labels
    assert len(remaining_labels) == len(initial_labels) - 1
    
    # Verify label directory was deleted
    label_path = Path(zarr_url) / "labels" / label_to_delete
    assert not label_path.exists()


def test_delete_multiple_labels(zarr_MIP_path, tmp_path):
    """Test deleting multiple labels at once."""
    # Create a copy of the zarr
    test_zarr = tmp_path / "test_zarr_multi.zarr"
    shutil.copytree(zarr_MIP_path, test_zarr, dirs_exist_ok=True)
    
    zarr_url = str(test_zarr / "B" / "03" / "0")
    
    # Check initial state
    image_group = zarr.group(zarr_url)
    labels_group = image_group["labels"]
    initial_labels = labels_group.attrs.asdict().get("labels", [])
    
    # Delete all labels if we have multiple, otherwise just the one
    labels_to_delete = initial_labels[:min(2, len(initial_labels))]
    
    delete_labels(
        zarr_url=zarr_url,
        labels_to_delete=labels_to_delete,
    )
    
    # Verify labels were removed
    labels_group = zarr.group(zarr_url)["labels"]
    remaining_labels = labels_group.attrs.asdict().get("labels", [])
    
    for label in labels_to_delete:
        assert label not in remaining_labels
        label_path = Path(zarr_url) / "labels" / label
        assert not label_path.exists()
    
    assert len(remaining_labels) == len(initial_labels) - len(labels_to_delete)


def test_delete_nonexistent_label(zarr_MIP_path, tmp_path):
    """Test that deleting a non-existent label raises an error."""
    # Create a copy of the zarr
    test_zarr = tmp_path / "test_zarr_error.zarr"
    shutil.copytree(zarr_MIP_path, test_zarr, dirs_exist_ok=True)
    
    zarr_url = str(test_zarr / "B" / "03" / "0")
    
    # Try to delete a label that doesn't exist
    import pytest
    with pytest.raises(ValueError, match="Label nonexistent_label not found"):
        delete_labels(
            zarr_url=zarr_url,
            labels_to_delete=["nonexistent_label"],
        )
