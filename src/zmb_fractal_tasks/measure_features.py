"""Segment spot-like particles."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# import fractal_tasks_core
from ngio import open_omezarr_container
from ngio.tables import FeatureTable
from pydantic import validate_call
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, regionprops_table
from skimage.measure._regionprops import COL_DTYPES

from zmb_fractal_tasks.from_fractal_tasks_core.channels import (
    ChannelInputModel,
    get_omero_channel_list,
)

# __OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


@validate_call
def measure_features(
    *,
    zarr_url: str,
    output_table_name: str,
    label_name: str,
    annotation_label_names: Optional[Sequence[str]] = None,
    shortest_distance_label_names: Optional[Sequence[str]] = None,
    channels_to_include: Optional[Sequence[ChannelInputModel]] = None,
    channels_to_exclude: Optional[Sequence[ChannelInputModel]] = None,
    structure_props: Optional[Sequence[str]] = None,
    intensity_props: Optional[Sequence[str]] = None,
    level: str = "0",
    overwrite: bool = True,
) -> None:
    """Calculate features based on label image and intensity image (optional).

    Takes a label image and an optional intensity image and calculates
    morphology, intensity and texture features in 2D.

    TODO: Currently only works for well-plates, since it also writes well_name.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_table_name: Name of the output table.
        label_name: Name of the label that contains the seeds.
            Needs to exist in OME-Zarr file.
        annotation_label_names: List of label names that should be used
            to annotate the labels. Need to exist in OME-Zarr file.
        shortest_distance_label_names: List of label names that should be used
            to calculate the shortest distance to the labels.
            Need to exist in OME-Zarr file.
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
        overwrite: If True, overwrite existing feature table.
    """
    omezarr = open_omezarr_container(zarr_url)
    label_image = omezarr.get_label(label_name, path=level)
    if annotation_label_names:
        annotation_label_images = {
            name: omezarr.get_label(name, path=level) for name in annotation_label_names
        }
    else:
        annotation_label_images = {}
    if shortest_distance_label_names:
        shortest_distance_label_images = {
            name: omezarr.get_label(name, path=level)
            for name in shortest_distance_label_names
        }
    else:
        shortest_distance_label_images = {}
    intensity_image = omezarr.get_image(path=level)

    # find plate and well names
    plate_name = Path(zarr_url.split(".zarr/")[0]).name
    component = zarr_url.split(".zarr/")[1]
    well_name = component.split("/")[0] + component.split("/")[1]

    logging.info(f"Calculating {output_table_name} for well {well_name}")

    # TODO: expose which roi-table to use?
    roi_table = omezarr.get_table("FOV_ROI_table", check_type="roi_table")

    measurements = []
    for roi in roi_table.rois():
        # load label image
        label_patch = label_image.get_roi(roi, mode="numpy")
        # load annotation label images
        annotation_label_patches = {
            name: image.get_roi(roi, mode="numpy")
            for name, image in annotation_label_images.items()
        }
        # load shortest distance label images
        shortest_distance_label_patches = {
            name: image.get_roi(roi, mode="numpy")
            for name, image in shortest_distance_label_images.items()
        }

        # load intensity images
        # first, get all channels in the acquisition and find the ones of interest
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
                roi, c=channel_idx, mode="numpy"
            )

        measurement = measure_features_ROI(
            labels=label_patch,
            annotations_list=list(annotation_label_patches.values()),
            intensities_list=[data[0] for data in intensity_patches.values()],
            shortest_distance_list=list(shortest_distance_label_patches.values()),
            ann_prefix_list=annotation_label_patches.keys(),
            int_prefix_list=intensity_patches.keys(),
            dist_prefix_list=shortest_distance_label_patches.keys(),
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

    df_measurements = pd.concat(measurements, axis=0, ignore_index=True)

    feat_table = FeatureTable(df_measurements, reference_label=label_name)
    omezarr.add_table(output_table_name, feat_table, overwrite=True)


def most_frequent_value(mask, img):  # noqa: D103
    masked_img = img[mask].astype(int)
    return np.bincount(masked_img).argmax()


def intensity_std(mask, img):  # noqa: D103
    masked_img = img[mask]
    return np.std(masked_img)


def intensity_total(mask, img):  # noqa: D103
    masked_img = img[mask]
    return np.sum(masked_img)


FUNS_PLUS = {
    "most_frequent_value": most_frequent_value,
    "intensity_std": intensity_std,
    "intensity_total": intensity_total,
}


def regionprops_plus(
    label_image,
    intensity_image=None,
    cache=True,
    *,
    extra_properties=None,
    spacing=None,
    offset=None,
):
    """Wrapper around regionprops, to integrate extra properties.

    The additional properties are:
    - most_frequent_value
        returns most frequent value of img inside mask
        (mainly used for annotating labels, where img are annotation labels)
    - intensity_std: float
        standard deviation of pixel values
    - intensity_total:
        sum of pixel values
    """
    properties_plus = [most_frequent_value, intensity_std, intensity_total]
    if extra_properties is not None:
        extra_properties = extra_properties + properties_plus
    else:
        extra_properties = properties_plus
    return regionprops(
        label_image=label_image,
        intensity_image=intensity_image,
        cache=cache,
        extra_properties=extra_properties,
        spacing=spacing,
        offset=offset,
    )


def regionprops_table_plus(
    label_image,
    intensity_image=None,
    properties=("label", "bbox"),
    *,
    cache=True,
    separator="-",
    extra_properties=None,
    spacing=None,
):
    """Like skimage.measure.regionprops_table(), but with some additional properties:

    - most_frequent_value
        returns most frequent value of img inside mask
        (mainly used for annotating labels, where img are annotation labels)
    - intensity_std: float
        standard deviation of pixel values
    - intensity_total:
        sum of pixel values
    """
    properties_org = []
    properties_plus = []
    for prop in properties:
        if prop in COL_DTYPES.keys():
            properties_org.append(prop)
        elif prop in FUNS_PLUS.keys():
            properties_plus.append(FUNS_PLUS[prop])
    rpt = regionprops_table(
        label_image,
        intensity_image,
        properties=properties_org,
        cache=cache,
        separator=separator,
        extra_properties=properties_plus,
        spacing=spacing,
    )
    return {
        prop: rpt[prop] for prop in properties
    }  # sort table according to input properties


def measure_features_ROI(
    labels,
    annotations_list,
    intensities_list,
    shortest_distance_list,
    ann_prefix_list=None,
    int_prefix_list=None,
    dist_prefix_list=None,
    structure_props=None,
    intensity_props=None,
    pxl_sizes=None,
    optional_columns: Optional[dict[str, Any]] = None,
):
    """Returns measurements of labels.

    Args:
        labels: Label image to be measured
        annotations_list: list of label images, which are used to annotate
            labels
        intensities_list: list of intensity images to measure
        shortest_distance_list: list of label images, for which the shortest
            distance is calculated
        ann_prefix_list: prefix to use for annotations (default: ann0, ann1, ann2,...)
        int_prefix_list: prefix to use for intensity measurements
            (default: c0, c1, c2, ...)
        dist_prefix_list: prefix to use for shortest_distance measurements
            (default: dist0, dist1, dist2, ...)
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

    # assign labels to annotations
    if ann_prefix_list is None:
        ann_prefix_list = [f"ann{i}" for i in range(len(annotations_list))]
    df_ann_list = []
    for annotations, ann_prefix in zip(annotations_list, ann_prefix_list):
        df_ann = pd.DataFrame(
            regionprops_table_plus(
                labels,
                annotations,
                properties=(
                    [
                        "label",
                        "most_frequent_value",
                    ]
                ),
            )
        )
        df_ann = df_ann.rename(
            columns={
                "most_frequent_value": f"{ann_prefix}_ID",
            }
        )
        df_ann.set_index("label", inplace=True)
        df_ann_list.append(df_ann)

    # do structure measurements
    if structure_props is None:
        structure_props = ["num_pixels"]
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
    for intensities, int_prefix in zip(intensities_list, int_prefix_list):
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

    # calculated shortest distances
    if dist_prefix_list is None:
        dist_prefix_list = [f"dist{i}" for i in range(len(shortest_distance_list))]
    df_dist_list = []
    for dist_label, dist_prefix in zip(shortest_distance_list, dist_prefix_list):
        dist_transform = distance_transform_edt(
            np.logical_not(dist_label), sampling=pxl_sizes
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
                "intensity_min": f"shortest_distance_to_{dist_prefix}",
            }
        )
        df_dist.set_index("label", inplace=True)
        df_dist_list.append(df_dist)
    # combine all
    df = pd.concat(
        [df, *df_ann_list, df_struct, *df_int_list, *df_dist_list],
        axis=1,
    )
    # add additional columns:
    for i, (col_name, col_val) in enumerate(optional_columns.items()):
        df.insert(i, col_name, col_val)
    return df.reset_index()


# if __name__ == "__main__":
#     from fractal_tasks_core.tasks._utils import run_fractal_task

#     run_fractal_task(task_function=segment_particles)
