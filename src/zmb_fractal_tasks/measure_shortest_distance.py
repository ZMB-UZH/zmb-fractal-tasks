"""Fractal task to measure features of labels."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ngio.transforms import ZoomTransform
import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.tables import FeatureTable
from pydantic import validate_call
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table


@validate_call
def measure_shortest_distance(
    *,
    zarr_url: str,
    output_table_name: str,
    input_label_name: str,
    target_label_names: Sequence[str],
    pyramid_level: str | None = None,
    roi_table: str = "FOV_ROI_table",
    append_to_table: bool = True,
) -> None:
    """Measure shortest distance of labels to target labels.

    Takes a label image and a target label image and calculates the shortest
    distance from each label to the nearest target label. Writes results to a
    feature table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_table_name: Name of the output table.
        input_label_name: Name of the label that contains the seeds.
            Needs to exist in OME-Zarr file.
        target_label_names: Names of the target labels to measure distance to.
        pyramid_level: Optional path to the pyramid level of the input label.
            The target label image will be resampled to match it. If None, the
            highest resolution of the input label will be used.
        roi_table: ROI table name to iterate over (e.g 'FOV_ROI_table').
            If left empty, measure over whole image.
        append_to_table: If True, append new measurements to existing table.
            If False, overwrite existing table.
    """
    ome_zarr = open_ome_zarr_container(zarr_url)
    if pyramid_level is None:
        label = ome_zarr.get_label(input_label_name)
    else:
        label = ome_zarr.get_label(input_label_name, path=pyramid_level)
    target_label_images = {
        name: ome_zarr.get_label(name, pixel_size=label.pixel_size)
        for name in target_label_names
    }

    if ome_zarr.is_time_series:
        raise NotImplementedError("Time series are not yet supported.")

    # find plate and well names
    plate_name = Path(Path(zarr_url).as_posix().split(".zarr/")[0]).stem
    try:
        component = Path(zarr_url).as_posix().split(".zarr/")[1]
        well_name = component.split("/")[0] + f"{int(component.split('/')[1]):02d}"
    except Exception:
        well_name = "None"

    logging.info(f"Calculating {output_table_name} for well {well_name}")

    df_measurements_list = []
    for target_label_name, target_label in target_label_images.items():
        logging.info(f"Processing target label: {target_label_name}")

        # transform to resample target label, in case of different resolutions
        zoom_transform = ZoomTransform(
            input_image=label,
            target_image=target_label,
            order="nearest",  # Nearest neighbor interpolation for labels
        )

        if ome_zarr.is_3d:
            axes_order = ["y", "x", "z"]
        else:
            axes_order = ["y", "x"]

        iterator = FeatureExtractorIterator(
            input_image=target_label,
            input_label=label,
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
        for label_data, target_label_data, roi in iterator.iter_as_numpy():
            logging.info(f"Processing ROI: {roi}")

            # Squeeze singleton dimensions from label_data
            label_data = np.squeeze(label_data)
            target_label_data = np.squeeze(target_label_data)

            # Determine pixel sizes based on actual dimensionality
            if label_data.ndim == 2:
                pxl_sizes = (label.pixel_size.y, label.pixel_size.x)
            else:  # 3D
                pxl_sizes = (
                    label.pixel_size.y,
                    label.pixel_size.x,
                    label.pixel_size.z,
                )

            roi_measurements = measure_shortest_distance_ROI(
                labels=label_data,
                target_label_list=[target_label_data],
                target_prefix_list=[target_label_name],
                pxl_sizes=pxl_sizes,
                optional_columns={
                    "plate": plate_name,
                    "well": well_name,
                    "ROI": roi.name,
                },
            )
            # Only append if there are measurements
            if len(roi_measurements) > 0:
                measurements.append(roi_measurements)

        if not measurements:
            logging.warning(
                f"No labels found for '{input_label_name}' in any ROI. "
                f"Skipping feature table '{output_table_name}'."
            )
            return

        df_measurements_list.append(pd.concat(measurements, axis=0))

    # Merge all per-target distance columns into a single dataframe
    df_measurements = pd.concat(df_measurements_list, axis=1)
    # Remove duplicate columns (e.g. plate, well, ROI repeated per target)
    df_measurements = df_measurements.loc[
        :, ~df_measurements.columns.duplicated(keep="first")
    ]

    if append_to_table and (output_table_name in ome_zarr.list_tables()):
        logging.info(f"Appending to existing table: {output_table_name}")
        feat_table_org = ome_zarr.get_table(output_table_name)
        df_org = feat_table_org.dataframe
        # Ensure same index (labels) to avoid misalignment
        if not df_org.index.equals(df_measurements.index):
            raise ValueError(
                "Index mismatch between existing and new feature table. Cannot append."
            )
        # Merge horizontally
        df_measurements = pd.concat([df_org, df_measurements], axis=1)
        # Remove duplicate columns, keeping the values from new df (rightmost)
        df_measurements = df_measurements.loc[
            :, ~df_measurements.columns.duplicated(keep="last")
        ]

    logging.info(f"Writing feature table: {output_table_name}")
    feat_table = FeatureTable(df_measurements, reference_label=input_label_name)
    ome_zarr.add_table(output_table_name, feat_table, overwrite=True)


def measure_shortest_distance_ROI(
    labels,
    target_label_list,
    target_prefix_list=None,
    pxl_sizes=None,
    optional_columns: dict[str, Any] | None = None,
):
    """Returns dataframe with shortest distance of each label to target labels.

    Args:
        labels: Label image to be measured
        target_label_list: list of target label images to measure
        target_prefix_list: prefix to use for annotations
            (default: dist0, dist1, dist2,...)
        pxl_sizes: list of pixel sizes, must have same length as passed image
            dimensions
        optional_columns: list of any additional columns and their entries
            (e.g. {'well':'C01'})

    Returns:
        Pandas dataframe
    """
    if optional_columns is None:
        optional_columns = {}

    # Set default prefix list
    if target_prefix_list is None:
        target_prefix_list = [f"dist{i}" for i in range(len(target_label_list))]

    # Check if labels are empty (all zeros)
    unique_labels = np.unique(labels)[np.unique(labels) != 0]
    is_empty = len(unique_labels) == 0

    if is_empty:
        # Create empty dataframe with all expected columns
        columns = list(optional_columns.keys()) + [
            f"shortest_distance_to_{target_prefix}"
            for target_prefix in target_prefix_list
        ]
        df = pd.DataFrame(columns=columns)
        df.index.name = "label"
        return df

    # initiate dataframe
    df = pd.DataFrame(index=unique_labels)
    df.index.name = "label"

    # calculated shortest distances
    df_dist_list = []
    for target_label, target_prefix in zip(
        target_label_list, target_prefix_list, strict=True
    ):
        dist_transform = distance_transform_edt(
            np.logical_not(target_label), sampling=pxl_sizes
        )
        df_dist = pd.DataFrame(
            regionprops_table(
                labels,
                dist_transform,
                properties=(
                    [
                        "label",
                        "intensity_min",
                    ]
                ),
                spacing=pxl_sizes,
            )
        )
        df_dist = df_dist.rename(
            columns={
                "intensity_min": f"shortest_distance_to_{target_prefix}",
            }
        )
        df_dist.set_index("label", inplace=True)
        df_dist_list.append(df_dist)

    # combine all
    df = pd.concat(
        [df, *df_dist_list],
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_shortest_distance)
