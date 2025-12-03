"""Fractal task to measure features of labels."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.tables import FeatureTable
from pydantic import validate_call

from zmb_fractal_tasks.from_fractal_tasks_core.channels import (
    ChannelInputModel,
    get_omero_channel_list,
)
from zmb_fractal_tasks.utils.regionprops_table_plus import regionprops_table_plus


@validate_call
def measure_features(
    *,
    zarr_url: str,
    output_table_name: str,
    label_name: str,
    channels_to_include: Optional[Sequence[ChannelInputModel]] = None,
    channels_to_exclude: Optional[Sequence[ChannelInputModel]] = None,
    structure_props: Optional[Sequence[str]] = None,
    intensity_props: Optional[Sequence[str]] = None,
    level: str = "0",
    roi_table_name: str = "FOV_ROI_table",
    append: bool = True,
    overwrite: bool = False,
) -> None:
    """Calculate features based on label image and intensity image (optional).

    Takes a label image and an optional intensity image and calculates
    morphology, intensity and texture features in 2D. Writes results to a
    feature table in the OME-Zarr file. If a feature table with the same name
    already exists, the features will be added to the existing table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_table_name: Name of the output table.
        label_name: Name of the label that contains the seeds.
            Needs to exist in OME-Zarr file.
        channels_to_include: List of channels to include for intensity
            and texture measurements. Use the channel label to indicate
            single channels. If None, all channels are included.
        channels_to_exclude: List of channels to exclude for intensity
            and texture measurements. Use the channel label to indicate
            single channels. If None, no channels are excluded.
        structure_props: List of regionprops structure properties to measure.
        intensity_props: List of regionprops intensity properties to measure.
                ROI_table_name: Name of the ROI table to process.
        level: Resolution of the label image to calculate features.
            Only tested for level 0.
        roi_table_name: Name of the ROI table to iterate over.
        append: If True, append new measurements to existing table.
        overwrite: Only used if append is False. If True, overwrite existing
            table. If False, raise error if table already exists.
    """
    omezarr = open_ome_zarr_container(zarr_url)
    label_image = omezarr.get_label(label_name, path=level)
    intensity_image = omezarr.get_image(path=level)

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

        # load intensity images
        # first, get all channels in the acquisition and find the ones of interest
        # TODO: handle with ngio
        omero_channels = get_omero_channel_list(image_zarr_path=zarr_url)
        if channels_to_include:
            channel_labels_to_include = [c.label for c in channels_to_include]
            channel_wavelength_ids_to_include = [
                c.wavelength_id for c in channels_to_include
            ]
            omero_channels = [
                c
                for c in omero_channels
                if (c.label in channel_labels_to_include)
                or (c.wavelength_id in channel_wavelength_ids_to_include)
            ]
        if channels_to_exclude:
            channel_labels_to_exclude = [c.label for c in channels_to_exclude]
            channel_wavelength_ids_to_exclude = [
                c.wavelength_id for c in channels_to_exclude
            ]
            omero_channels = [
                c
                for c in omero_channels
                if (c.label not in channel_labels_to_exclude)
                and (c.wavelength_id not in channel_wavelength_ids_to_exclude)
            ]
        intensity_patches = {}
        for omero_channel in omero_channels:
            channel_idx = intensity_image.channel_labels.index(omero_channel.label)
            intensity_patches[omero_channel.wavelength_id] = intensity_image.get_roi(
                roi, c=channel_idx, mode="numpy", axes_order="czyx"
            )

        measurement = measure_features_ROI(
            labels=label_patch,
            intensities_list=[data[0] for data in intensity_patches.values()],
            int_prefix_list=intensity_patches.keys(),
            structure_props=structure_props,
            intensity_props=intensity_props,
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


def measure_features_ROI(
    labels,
    intensities_list,
    int_prefix_list=None,
    structure_props=None,
    intensity_props=None,
    pxl_sizes=None,
    optional_columns: dict[str, Any] | None = None,
):
    """Returns measurements of labels.

    Args:
        labels: Label image to be measured
        intensities_list: list of intensity images to measure
        int_prefix_list: prefix to use for intensity measurements
            (default: c0, c1, c2, ...)
        structure_props: list of structure properties to measure
        intensity_props: list of intensity properties to measure
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

    # do structure measurements
    if structure_props is None:
        structure_props = ["num_pixels", "area"]
    df_struct = pd.DataFrame(
        regionprops_table_plus(
            labels,
            None,
            properties=(["label", *structure_props]),
            spacing=pxl_sizes,
        )
    )
    df_struct.set_index("label", inplace=True)

    # do intensity measurements
    if int_prefix_list is None:
        int_prefix_list = [f"c{i}" for i in range(len(intensities_list))]
    if intensity_props is None:
        intensity_props = ["intensity_mean", "intensity_std", "intensity_total"]
    df_int_list = []
    for intensities, int_prefix in zip(intensities_list, int_prefix_list, strict=True):
        df_int = pd.DataFrame(
            regionprops_table_plus(
                labels,
                intensities,
                properties=(["label", *intensity_props]),
                spacing=pxl_sizes,
            )
        )
        df_int = df_int.rename(
            columns={prop: f"{int_prefix}_{prop}" for prop in intensity_props}
        )
        df_int.set_index("label", inplace=True)
        df_int_list.append(df_int)

    # combine all
    df = pd.concat(
        [df, df_struct, *df_int_list],
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_features)
