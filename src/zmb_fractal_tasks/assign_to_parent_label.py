"""Fractal task to measure features of labels."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import FeatureExtractorIterator
from ngio.tables import FeatureTable
from ngio.transforms import ZoomTransform
from pydantic import BaseModel, validate_call

from zmb_fractal_tasks.utils.regionprops_table_plus import regionprops_table_plus


class ParentLabelInput(BaseModel):
    """Parent label configuration.

    Args:
        parent_label_name (str): Name of the parent label.
        output_parent_table_name (str): Name of corresponding output parent
            feature table. (Only needed if aggregate_features is True).
    """

    parent_label_name: str
    output_parent_table_name: Optional[str] = None


class AggregationOptions(BaseModel):
    """Options for feature aggregation.

    Args:
        aggregate_features (bool): Whether to aggregate features in seed-table
            to parent-table.
        features_to_aggregate (Sequence[str] | None): List of feature names
            (columns in seed-table) to aggregate to parent-table.
            If left empty, all features from seed table are aggregated.
        aggregation_methods (Sequence[str] | None): List of aggregation methods
            to use for each feature. Typical methods are 'sum', 'mean', 'std',
            'sem', 'min', 'max' (any built-in pandas function). A count of
            seeds per parent label is always added automatically.
        append_to_parent_table (bool): If True, aggregated features to existing
            table(s). If False, overwrite existing table or create new one.
    """

    aggregate_features: bool = True
    features_to_aggregate: Optional[Sequence[str]] = None
    aggregation_methods: Sequence[str] = ["mean", "std"]
    append_to_parent_table: bool = True


class AdditionalOptions(BaseModel):
    """Additional options.

    Args:
        pyramid_level (str): Resolution level of the label image to use for
            calculations. Choose `0` for full resolution.
        roi_table (str): ROI table name to iterate over (e.g 'FOV_ROI_table').
            If left empty, measure over whole image.
        append_to_seed_table (bool): If True, append new measurements to
            existing seed table. If False, overwrite existing table or create
            new one.
    """

    pyramid_level: str = "0"
    roi_table: str = "FOV_ROI_table"
    append_to_seed_table: bool = True


@validate_call
def assign_to_parent_label(
    *,
    zarr_url: str,
    seed_label_name: str,
    seed_table_name: str,
    parent_labels: Sequence[ParentLabelInput],
    aggregation_options: AggregationOptions,
    additional_options: AdditionalOptions,
) -> None:
    """Assign label to parent label and optionally aggregate features.

    Takes a seed label image and a parent label image and assigns each seed
    label to a parent label based on maximum overlap. The assigned parent
    label IDs are then stored in the seed-table. More than one parent label can
    be provided to assign to multiple parents.

    Optionally, features from the seed-table (if it already exists) can be
    aggregated to the parent-table by using the specified aggregation methods.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        seed_label_name: Name of the label that contains the seeds to be
            assigned to a parent.
        seed_table_name: Name of the output seed table. (Usually this is the
            feature table of the seed label).
        parent_labels: Parent label(s) to assign seeds to.
        aggregation_options: Options for aggregating features from seed-table
            to parent-table(s).
        additional_options: Additional options for the task.
    """
    ome_zarr = open_ome_zarr_container(zarr_url)

    if ome_zarr.is_time_series:
        raise NotImplementedError("Time series are not yet supported.")

    seed_label_image = ome_zarr.get_label(
        seed_label_name, path=additional_options.pyramid_level
    )
    parent_label_images = {
        parent_label.parent_label_name: ome_zarr.get_label(
            parent_label.parent_label_name, path=additional_options.pyramid_level
        )
        for parent_label in parent_labels
    }

    # find plate and well names
    plate_name = Path(Path(zarr_url).as_posix().split(".zarr/")[0]).stem
    try:
        component = Path(zarr_url).as_posix().split(".zarr/")[1]
        well_name = component.split("/")[0] + f"{int(component.split('/')[1]):02d}"
    except Exception:
        well_name = "None"

    df_measurements_list = []
    for parent_label_name, parent_label_image in parent_label_images.items():
        logging.info(f"Processing parent label: {parent_label_name}")

        # transform to resample label, in case of different resolutions
        zoom_transform = ZoomTransform(
            input_image=seed_label_image,
            target_image=parent_label_image,
            order="nearest",  # Nearest neighbor interpolation for labels
        )

        if ome_zarr.is_3d:
            axes_order = ["y", "x", "z"]
        else:
            axes_order = ["y", "x"]

        iterator = FeatureExtractorIterator(
            input_image=parent_label_image,
            input_label=seed_label_image,
            label_transforms=[zoom_transform],
            axes_order=axes_order,
        )

        if additional_options.roi_table != "":
            # If a ROI table is provided, we load it and use it to further restrict
            # the iteration to the ROIs defined in the table
            table = ome_zarr.get_generic_roi_table(name=additional_options.roi_table)
            logging.info(f"ROI table retrieved: {table=}")
            iterator = iterator.product(table)
            logging.info(f"Iterator updated with ROI table: {iterator=}")

        measurements = []
        for parent_label_data, seed_label_data, roi in iterator.iter_as_numpy():
            logging.info(f"Processing ROI: {roi}")

            # Squeeze singleton dimensions from label_data
            label_data = np.squeeze(seed_label_data)
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

        # Filter out empty dataframes before concatenation to avoid FutureWarning
        non_empty_measurements = [m for m in measurements if not m.empty]
        if non_empty_measurements:
            df_measurements_list.append(pd.concat(non_empty_measurements, axis=0))
        else:
            # If all measurements are empty, create an empty dataframe with expected columns
            # Use the first measurement as template for columns
            df_measurements_list.append(measurements[0])

    # merge all parent measurements
    df_measurements = pd.concat(df_measurements_list, axis=1)
    # Remove duplicate columns
    df_measurements = df_measurements.loc[
        :, ~df_measurements.columns.duplicated(keep="first")
    ]

    if additional_options.append_to_seed_table and (
        seed_table_name in ome_zarr.list_tables()
    ):
        feat_table_org = ome_zarr.get_table(seed_table_name)
        df_org = feat_table_org.dataframe
        # Ensure same index (labels) to avoid misalignment
        if not df_org.index.equals(df_measurements.index):
            raise ValueError(
                f"Index mismatch between existing feature table {seed_table_name} and "
                "new measurements. Cannot append."
            )
        # Merge horizontally
        df_measurements = pd.concat([df_org, df_measurements], axis=1)
        # Remove duplicate columns, keeping the values from new df (rightmost)
        df_measurements = df_measurements.loc[
            :, ~df_measurements.columns.duplicated(keep="last")
        ]

    logging.info(f"Writing measurements to feature table: {seed_table_name}")
    feat_table = FeatureTable(df_measurements, reference_label=seed_label_name)
    ome_zarr.add_table(seed_table_name, feat_table, overwrite=True)

    if aggregation_options.aggregate_features:
        logging.info("Starting feature aggregation to parent tables.")
        for parent_label in parent_labels:
            parent_label_name = parent_label.parent_label_name
            output_parent_table_name = parent_label.output_parent_table_name
            if output_parent_table_name is None:
                raise ValueError(
                    "Output parent table name must be provided for aggregation."
                )

            if aggregation_options.append_to_parent_table and (
                output_parent_table_name in ome_zarr.list_tables()
            ):
                df_parent_org = ome_zarr.get_table(output_parent_table_name).dataframe
            else:
                df_parent_org = None

            df_aggregated = aggregate_features(
                seed_df=df_measurements,
                seed_label_name=seed_label_name,
                parent_df=df_parent_org,
                parent_label_name=parent_label_name,
                features_to_aggregate=aggregation_options.features_to_aggregate,
                aggregation_methods=aggregation_options.aggregation_methods,
            )

            logging.info(
                f"Writing aggregated measurements to feature table: "
                f"{output_parent_table_name}"
            )
            feat_table_parent = FeatureTable(
                df_aggregated, reference_label=parent_label_name
            )
            ome_zarr.add_table(
                output_parent_table_name, feat_table_parent, overwrite=True
            )


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
        # Set proper dtype for parent ID column to avoid object dtype
        df[f"{parent_prefix}_ID"] = df[f"{parent_prefix}_ID"].astype("Int64")
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


def aggregate_features(
    seed_df: pd.DataFrame,
    seed_label_name: str = "seed",
    parent_df: Optional[pd.DataFrame] = None,
    parent_label_name: str = "parent",
    features_to_aggregate: Optional[Sequence[str]] = None,
    aggregation_methods: Sequence[str] = ["mean", "std"],
):
    """Function to aggregate features from seed-table.

    Args:
        seed_df: Dataframe of the seed feature table.
        seed_label_name: Name of the seed label (used to name columns).
        parent_df: Optional dataframe of the parent feature table to aggregate
            to. If None, only aggregated features are returned without
            existing parent features.
        parent_label_name: Name of the parent label (used to find parent IDs in
            seed table).
        features_to_aggregate: List of feature names (columns in seed_df) to
            aggregate to parent-table. If None, all features are aggregated.
        aggregation_methods: List of aggregation methods to use for each feature.
            Typical methods are 'sum', 'mean', 'std', 'sem', 'min', 'max'
            (any built-in pandas function). A count of seeds per parent label
            is always added automatically.
    """
    # If no features specified, aggregate all numeric columns
    if features_to_aggregate is None:
        features_to_aggregate = seed_df.select_dtypes(include=[np.number]).columns

    # perform aggregation
    group = seed_df.groupby(by=parent_label_name + "_ID")
    agg_dict = {f"{seed_label_name}_count": (features_to_aggregate[0], "count")}
    for feature in features_to_aggregate:
        for agg in aggregation_methods:
            agg_dict[f"{seed_label_name}_{feature}_{agg}"] = (feature, agg)
    seed_df_agg = group.agg(**agg_dict)
    seed_df_agg.index.name = "label"

    # combine with parent df if provided
    if parent_df is not None:
        seed_df_agg = seed_df_agg.reindex(parent_df.index, fill_value=0)
        df_out = pd.concat([parent_df, seed_df_agg], axis=1)
    else:
        df_out = seed_df_agg

    return df_out


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=assign_to_parent_label)
