"""Fractal task to measure features of labels."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.tables import FeatureTable
from ngio.transforms import ZoomTransform
from pydantic import validate_call

from zmb_fractal_tasks.utils.regionprops_table_plus import regionprops_table_plus


@validate_call
def measure_parent_label(
    *,
    zarr_url: str,
    input_label_name: str,
    output_table_name: str,
    parent_label_names: Sequence[str],
    pyramid_level: str = "0",
    roi_table: str = "FOV_ROI_table",
    append_to_table: bool = True,
) -> None:
    """Assign label to parent label.

    Takes a label image and a parent label image and assigns each label to a
    parent label based on maximum overlap. Writes results to a feature table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_label_name: Name of the label that contains the seeds to be
            assigned to a parent.
        output_table_name: Name of the output table. (Usually the feature table
            of the input label).
        parent_label_names: Names of the parent labels to assign to.
        pyramid_level: Resolution level of the label image to use for
            calculations. Choose `0` for full resolution.
            Only tested for 2D images at level 0.
        roi_table: ROI table name to iterate over (e.g 'FOV_ROI_table').
            If left empty, measure over whole image.
        append_to_table: If True, append new measurements to existing table.
    """
    ome_zarr = open_ome_zarr_container(zarr_url)
    label_image = ome_zarr.get_label(input_label_name, path=pyramid_level)
    parent_label_images = {
        name: ome_zarr.get_label(name, path=pyramid_level)
        for name in parent_label_names
    }

    if ome_zarr.is_time_series:
        raise NotImplementedError("Time series are not yet supported.")

    # find plate and well names
    plate_name = Path(Path(zarr_url).as_posix().split(".zarr/")[0]).stem
    try:
        component = Path(zarr_url).as_posix().split(".zarr/")[1]
        well_name = component.split("/")[0] + component.split("/")[1]
    except Exception:
        well_name = "None"

    logging.info(f"Calculating {output_table_name} for well {well_name}")

    df_measurements_list = []
    for parent_label_name, parent_label_image in parent_label_images.items():
        # transform to resample label, in case of different resolutions
        zoom_transform = ZoomTransform(
            input_image=label_image,
            target_image=parent_label_image,
            order="nearest",  # Nearest neighbor interpolation for labels
        )

        if ome_zarr.is_3d:
            axes_order = ["y", "x", "z"]
        else:
            axes_order = ["y", "x"]

        iterator = FeatureExtractorIterator(
            input_image=parent_label_image,
            input_label=label_image,
            label_transforms=[zoom_transform],
            axes_order=axes_order,
        )

        if roi_table != "":
            # If a ROI table is provided, we load it and use it to further restrict
            # the iteration to the ROIs defined in the table
            table = ome_zarr.get_generic_roi_table(name=roi_table)
            logging.info(f"ROI table retrieved: {table=}")
            iterator = iterator.product(table)
            logging.info(f"Iterator updated with ROI table: {iterator=}")

        measurements = []
        for parent_label_data, label_data, roi in iterator.iter_as_numpy():
            logging.info(f"Processing ROI: {roi}")

            # Squeeze singleton dimensions from label_data
            label_data = np.squeeze(label_data)
            parent_label_data = np.squeeze(parent_label_data)

            roi_measurements = measure_parent_ROI(
                labels=label_data,
                parent_labels=parent_label_data,
                parent_prefix=parent_label_name,
                optional_columns={
                    "plate": plate_name,
                    "well": well_name,
                    "ROI": roi.name,
                },
            )
            measurements.append(roi_measurements)

        df_measurements_list.append(pd.concat(measurements, axis=0))

    # merge all parent measurements
    df_measurements = pd.concat(df_measurements_list, axis=1)
    # Remove duplicate columns
    df_measurements = df_measurements.loc[
        :, ~df_measurements.columns.duplicated(keep="first")
    ]

    if append_to_table and (output_table_name in ome_zarr.list_tables()):
        feat_table_org = ome_zarr.get_table(output_table_name)
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

    feat_table = FeatureTable(df_measurements, reference_label=input_label_name)
    ome_zarr.add_table(output_table_name, feat_table, overwrite=True)


def measure_parent_ROI(
    labels,
    parent_labels,
    parent_prefix,
    optional_columns: dict[str, Any] | None = None,
):
    """Returns dataframe with index of the parent-label of each label.

    Args:
        labels: Label image to be measured
        parent_labels: lparent label image to measure
        parent_prefix: prefix to use for annotation
        optional_columns: additional columns to add to dataframe

    Returns:
        Pandas dataframe
    """
    # Check if labels are empty (all zeros)
    unique_labels = np.unique(labels)[np.unique(labels) != 0]
    is_empty = len(unique_labels) == 0

    if is_empty:
        # Create & return empty dataframe with all expected columns
        columns = list(optional_columns.keys()) if optional_columns else []
        columns.append(f"{parent_prefix}_ID")
        df = pd.DataFrame(columns=columns)
        df.index.name = "label"
        return df

    # initiate dataframe
    df = pd.DataFrame(index=unique_labels)
    df.index.name = "label"

    # assign labels to parent-labels
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

    # combine all
    df = pd.concat(
        [df, df_parent],
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_parent_label)
