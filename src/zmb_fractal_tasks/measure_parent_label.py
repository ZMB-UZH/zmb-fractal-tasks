"""Fractal task to measure features of labels."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.tables import FeatureTable
from pydantic import validate_call

from zmb_fractal_tasks.utils.regionprops_table_plus import regionprops_table_plus


@validate_call
def measure_parent_label(
    *,
    zarr_url: str,
    output_table_name: str,
    label_name: str,
    parent_label_names: Sequence[str],
    level: str = "0",
    roi_table_name: str = "FOV_ROI_table",
    append: bool = True,
    overwrite: bool = False,
) -> None:
    """Assign label to parent label.

    Takes a label image and a parent label image and assigns each label to a
    parent label based on maximum overlap. Writes results to a feature table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_table_name: Name of the output table.
        label_name: Name of the label that contains the seeds.
            Needs to exist in OME-Zarr file.
        parent_label_names: Names of the parent labels to assign to.
        level: Resolution of the label image to use for calculations.
        roi_table_name: Name of the ROI table to iterate over.
        append: If True, append new measurements to existing table.
        overwrite: Only used if append is False. If True, overwrite existing
            table. If False, raise error if table already exists.
    """
    omezarr = open_ome_zarr_container(zarr_url)
    label_image = omezarr.get_label(label_name, path=level)
    parent_label_images = {
        name: omezarr.get_label(name, path=level) for name in parent_label_names
    }

    # find plate and well names
    plate_name = Path(Path(zarr_url).as_posix().split(".zarr/")[0]).stem
    try:
        component = Path(zarr_url).as_posix().split(".zarr/")[1]
        well_name = component.split("/")[0] + component.split("/")[1]
    except Exception:
        well_name = "None"

    logging.info(f"Calculating {output_table_name} for well {well_name}")

    roi_table = omezarr.get_table(roi_table_name, check_type="roi_table")

    measurements = []
    for roi in roi_table.rois():
        # load label image
        label_patch = label_image.get_roi(roi, mode="numpy", axes_order="zyx")
        # load parent label images
        parent_label_patches = {
            name: image.get_roi(roi, mode="numpy", axes_order="zyx")
            for name, image in parent_label_images.items()
        }

        measurement = measure_parents_ROI(
            labels=label_patch,
            parent_label_list=parent_label_patches.values(),
            parent_prefix_list=parent_label_patches.keys(),
            optional_columns={
                "plate": plate_name,
                "well": well_name,
                "ROI": roi.name,
            },
        )
        measurements.append(measurement)

    df_measurements = pd.concat(measurements, axis=0)

    if append and (output_table_name in omezarr.list_tables()):
        feat_table_org = omezarr.get_table(
            output_table_name, check_type="feature_table"
        )
        df_org = feat_table_org.dataframe
        # Ensure same index (labels) to avoid misalignment
        if not df_org.index.equals(df_measurements.index):
            raise ValueError(
                "Index mismatch between existing feature table and new measurements."
            )
        # Merge horizontally
        df_measurements = pd.concat([df_org, df_measurements], axis=1)
        # Remove duplicate columns, keeping the values from new df (rightmost)
        df_measurements = df_measurements.loc[
            :, ~df_measurements.columns.duplicated(keep="last")
        ]

    if append:
        overwrite = True

    feat_table = FeatureTable(df_measurements, reference_label=label_name)
    omezarr.add_table(output_table_name, feat_table, overwrite=overwrite)


def measure_parents_ROI(
    labels,
    parent_label_list,
    parent_prefix_list=None,
    optional_columns: dict[str, Any] | None = None,
):
    """Returns dataframe with index of the parent-label of each label.

    Args:
        labels: Label image to be measured
        parent_label_list: list of parent label images to measure
        parent_prefix_list: prefix to use for annotations
            (default: parent0, parent1, parent2,...)
        optional_columns: list of any additional columns and their entries
            (e.g. {'well':'C01'})

    Returns:
        Pandas dataframe
    """
    if optional_columns is None:
        optional_columns = {}

    # initiate dataframe
    df = pd.DataFrame(index=np.unique(labels)[np.unique(labels) != 0])
    df.index.name = "label"

    # assign labels to parent-labels
    if parent_prefix_list is None:
        parent_prefix_list = [f"parent{i}" for i in range(len(parent_label_list))]
    df_parent_list = []
    for parent_labels, parent_prefix in zip(
        parent_label_list, parent_prefix_list, strict=True
    ):
        df_parent = pd.DataFrame(
            regionprops_table_plus(
                labels,
                parent_labels,
                properties=(
                    [
                        "label",
                        "most_frequent_value",
                    ]
                ),
            )
        )
        df_parent = df_parent.rename(
            columns={
                "most_frequent_value": f"{parent_prefix}_ID",
            }
        )
        df_parent.set_index("label", inplace=True)
        df_parent_list.append(df_parent)

    # combine all
    df = pd.concat(
        [df, *df_parent_list],
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_parent_label)
