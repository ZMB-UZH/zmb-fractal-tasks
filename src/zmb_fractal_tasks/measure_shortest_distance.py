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
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table


@validate_call
def measure_shortest_distance(
    *,
    zarr_url: str,
    output_table_name: str,
    label_name: str,
    target_label_names: Sequence[str],
    level: str = "0",
    roi_table_name: str = "FOV_ROI_table",
    append: bool = True,
    overwrite: bool = False,
) -> None:
    """Measure shortest distance of labels to target labels.

    Takes a label image and a target label image and calculates the shortest
    distance from each label to the nearest target label. Writes results to a
    feature table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_table_name: Name of the output table.
        label_name: Name of the label that contains the seeds.
            Needs to exist in OME-Zarr file.
        target_label_names: Names of the target labels to measure distance to.
        level: Resolution of the label image to use for calculations.
        roi_table_name: Name of the ROI table to iterate over.
        append: If True, append new measurements to existing table.
        overwrite: Only used if append is False. If True, overwrite existing
            table. If False, raise error if table already exists.
    """
    omezarr = open_ome_zarr_container(zarr_url)
    label_image = omezarr.get_label(label_name, path=level)
    target_label_images = {
        name: omezarr.get_label(name, path=level) for name in target_label_names
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
        # load target label images
        target_label_patches = {
            name: image.get_roi(roi, mode="numpy", axes_order="zyx")
            for name, image in target_label_images.items()
        }

        measurement = measure_shortest_distance_ROI(
            labels=label_patch,
            target_label_list=target_label_patches.values(),
            target_prefix_list=target_label_patches.keys(),
            pxl_sizes=(
                label_image.pixel_size.z,
                label_image.pixel_size.y,
                label_image.pixel_size.x,
            ),
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

    # initiate dataframe
    df = pd.DataFrame(index=np.unique(labels)[np.unique(labels) != 0])
    df.index.name = "label"

    # calculated shortest distances
    if target_prefix_list is None:
        target_prefix_list = [f"dist{i}" for i in range(len(target_label_list))]
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
