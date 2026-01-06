"""Fractal task to delete labels from an OME-Zarr image."""

import shutil
from pathlib import Path
from typing import Optional

import zarr
from pydantic import validate_call


# TODO: handle using ngio when support is added
@validate_call
def delete_labels(
    *,
    zarr_url: str,
    labels_to_delete: Optional[list[str]],
):
    """Delete labels from an OME-Zarr image.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
        labels_to_delete: Add names of labels to delete. If left empty, all
            labels will be deleted.
    """
    # Load image group
    image_group = zarr.group(zarr_url)

    # Check if labels group exists
    if "labels" not in set(image_group.group_keys()):
        raise ValueError(f"No labels group found in {zarr_url}.")

    labels_group = image_group["labels"]
    label_names = labels_group.attrs.asdict().get("labels", [])

    # If labels_to_delete is empty, delete all labels
    if not labels_to_delete:
        labels_to_delete = label_names.copy()
    else:
        # Check if all labels to delete exist
        for label_name in labels_to_delete:
            if label_name not in label_names:
                raise ValueError(f"Label {label_name} not found in {zarr_url}.")

    # Remove labels from metadata
    updated_label_names = [
        label for label in label_names if label not in labels_to_delete
    ]
    labels_group.attrs["labels"] = updated_label_names

    # Delete label directories
    for label_name in labels_to_delete:
        label_path = Path(zarr_url) / "labels" / label_name
        if label_path.exists():
            shutil.rmtree(str(label_path))


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=delete_labels)
