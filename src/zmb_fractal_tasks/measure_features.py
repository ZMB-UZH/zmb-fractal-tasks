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
from pydantic import BaseModel, validate_call

from zmb_fractal_tasks.utils.channel_utils import MeasurementChannels
from zmb_fractal_tasks.utils.regionprops_table_plus import regionprops_table_plus


class LabelInput(BaseModel):
    """Input label configuration.

    Args:
        input_label_name (str): Name of the label to be used for measurement.
        output_table_name (str): Name of corresponding output feature table.
    """

    input_label_name: str
    output_table_name: str


@validate_call
def measure_features(
    *,
    zarr_url: str,
    input_labels: Sequence[LabelInput],
    channels_to_measure: MeasurementChannels,
    structure_props: Sequence[str] = ["area"],
    intensity_props: Sequence[str] = [
        "intensity_mean",
        "intensity_std",
        "intensity_total",
    ],
    roi_table: str = "FOV_ROI_table",
    pyramid_level: str | None = None,
    append_to_table: bool = True,
) -> None:
    """Measure shape and intensity features of labels and write to feature table.

    This task takes one or more label images and measures specified shape and
    intensity features for each label, using the skimage.measure.regionprops
    function.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_labels: Label(s) to be measured.
        channels_to_measure: Channels for intensity measurements.
        structure_props: List of regionprops structure properties to measure.
            E.g. 'area', 'perimeter', 'solidity'.
            See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            for full list of possible properties.
        intensity_props: List of regionprops intensity properties to measure.
            E.g. 'intensity_mean', 'intensity_std', 'intensity_total'.
            See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            for full list of possible properties.
        roi_table: ROI table name to iterate over (e.g 'FOV_ROI_table').
            If left empty, measure over whole image.
        pyramid_level: Optional path to the pyramid level of the INTENSITY
            image used for measurement. The label image will be resampled to
            match it. If None, the highest resolution of the intensity image
            will be used.
        append_to_table: If True, append new measurements to existing table.
            If False, overwrite existing table.
    """
    ome_zarr = open_ome_zarr_container(zarr_url)
    if pyramid_level is None:
        image = ome_zarr.get_image()
    else:
        image = ome_zarr.get_image(path=pyramid_level)

    if ome_zarr.is_time_series:
        raise NotImplementedError("Time series are not yet supported.")

    # find plate and well names
    plate_name = Path(Path(zarr_url).as_posix().split(".zarr/")[0]).stem
    try:
        component = Path(zarr_url).as_posix().split(".zarr/")[1]
        well_name = component.split("/")[0] + f"{int(component.split('/')[1]):02d}"
    except Exception:
        well_name = "None"

    if channels_to_measure.use_all_channels:
        channels = image.channel_labels
    else:
        channel_selection_models = channels_to_measure.to_list()
        # convert to channel labels
        # TODO: figure out how to better get labels from ChannelSelectionModel
        channels = []
        for channel_selection_model in channel_selection_models:
            if channel_selection_model.mode == "index":
                channel_idx = int(channel_selection_model.identifier)
                channels.append(image.channel_labels[channel_idx])
            elif channel_selection_model.mode == "wavelength_id":
                channel_idx = image.get_channel_idx(channel_selection_model.identifier)
                channels.append(image.channel_labels[channel_idx])
            else:
                channels.append(channel_selection_model.identifier)

    for input_label_model in input_labels:
        logging.info(f"Processing label: {input_label_model.input_label_name}")
        input_label_name = input_label_model.input_label_name
        output_table_name = input_label_model.output_table_name
        label = ome_zarr.get_label(input_label_name, pixel_size=image.pixel_size)

        # transform to resample label, in case of different resolutions
        zoom_transform = ZoomTransform(
            input_image=label,
            target_image=image,
            order="nearest",  # Nearest neighbor interpolation for labels
        )

        if ome_zarr.is_3d:
            axes_order = ["y", "x", "z", "c"]
        else:
            axes_order = ["y", "x", "c"]

        iterator = FeatureExtractorIterator(
            input_image=image,
            input_label=label,
            channel_selection=channels,
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
        for image_data, label_data, roi in iterator.iter_as_numpy():
            logging.info(f"Processing ROI: {roi}")

            # Squeeze singleton dimensions from label_data
            label_data = np.squeeze(label_data)

            # Convert image_data to list of per-channel arrays
            # image_data shape is (y, x, num_channels) or (y, x, z, num_channels)
            intensities_list = [image_data[..., i] for i in range(image_data.shape[-1])]

            # Determine pixel sizes based on actual dimensionality
            if label_data.ndim == 2:
                pxl_sizes = (image.pixel_size.y, image.pixel_size.x)
            else:  # 3D
                pxl_sizes = (image.pixel_size.y, image.pixel_size.x, image.pixel_size.z)

            roi_measurements = measure_features_ROI(
                labels=label_data,
                intensities_list=intensities_list,
                int_prefix_list=channels,
                structure_props=structure_props,
                intensity_props=intensity_props,
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
            continue

        df_measurements = pd.concat(measurements, axis=0)

        if append_to_table and (output_table_name in ome_zarr.list_tables()):
            logging.info(f"Appending to existing table: {output_table_name}")
            feat_table_org = ome_zarr.get_table(output_table_name)
            df_org = feat_table_org.dataframe
            # Ensure same index (labels) to avoid misalignment
            if not df_org.index.equals(df_measurements.index):
                raise ValueError(
                    "Index mismatch between existing and new feature table."
                    " Cannot append."
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


def measure_features_ROI(
    labels: np.ndarray,
    intensities_list: list[np.ndarray],
    int_prefix_list: list[str] | None = None,
    structure_props: list[str] | None = None,
    intensity_props: list[str] | None = None,
    pxl_sizes: tuple[float, ...] | None = None,
    optional_columns: dict[str, Any] | None = None,
):
    """Returns measurements of labels.

    Args:
        labels: Label image to be measured. (ndarray x,y,[z])
        intensities_list: list of intensity images to measure (ndarray x,y,[z]).
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
    # Set default properties
    if structure_props is None:
        structure_props = ["num_pixels", "area"]
    if int_prefix_list is None:
        int_prefix_list = [f"c{i}" for i in range(len(intensities_list))]
    if intensity_props is None:
        intensity_props = ["intensity_mean", "intensity_std", "intensity_total"]
    if optional_columns is None:
        optional_columns = {}

    # Check if labels are empty (all zeros)
    unique_labels = np.unique(labels)[np.unique(labels) != 0]
    is_empty = len(unique_labels) == 0

    if is_empty:
        # Create & return empty dataframe with all expected columns
        columns = list(optional_columns.keys()) + structure_props
        for int_prefix in int_prefix_list:
            columns.extend([f"{int_prefix}_{prop}" for prop in intensity_props])
        df = pd.DataFrame(columns=columns)
        df.index.name = "label"
        return df

    # initiate dataframe
    df = pd.DataFrame(index=unique_labels)
    df.index.name = "label"

    # do structure measurements
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
