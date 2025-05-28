# Adapted from https://github.com/fractal-analytics-platform/fractal-tasks-core/...
# ...blob/main/fractal_tasks_core/tasks/cellpose_utils.py

"""Helper functions for image normalization."""

import logging
from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from ngio import open_ome_zarr_container
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from zmb_fractal_tasks.from_fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
)
from zmb_fractal_tasks.utils.histogram import Histogram, anndata_to_histograms

logger = logging.getLogger(__name__)

# TODO: Add functionality to choose which histogram to use

class CustomNormalizer(BaseModel):
    """Validator to handle different image normalization scenarios.

    If `mode="default"`, then default normalization of the function is
    used and no other parameters can be specified.
    If `mode="no_normalization"`, then no normalization is used and no
    other parameters can be specified.
    If `mode="custom"`, then either percentiles or explicit integer
    bounds can be applied.
    If `mode="omero"`, then the "start" and "end" values from the omero
    channels in the zarr file are used.
    If `mode="histogram"`, then a precomputed histogram is used to calculate
    the percentiles for normalization.

    Attributes:
        mode: One of `default` (default normalization), `custom`
            (using the other custom parameters), `omero` (using the
            values in the omero channel), `histogram` (using a precalculated
            histogram) or `no_normalization`.
        lower_percentile: Specify a custom lower-bound percentile for
            rescaling as a float value between 0 and 100. You can only specify
            percentiles or bounds, not both.
        upper_percentile: Specify a custom upper-bound percentile for
            rescaling as a float value between 0 and 100. You can only specify
            percentiles or bounds, not both.
        lower_bound: Explicit lower bound value to rescale the image at.
            Needs to be an integer, e.g. 100.
            You can only specify percentiles or bounds, not both.
        upper_bound: Explicit upper bound value to rescale the image at.
            Needs to be an integer, e.g. 2000.
            You can only specify percentiles or bounds, not both.
        histogram_name: Name of the histogram to use for rescaling in
            `histogram` mode.
            Needs to be a string, e.g. "channel_histograms".
    """

    mode: Literal["default", "custom", "omero", "histogram", "no_normalization"] = (
        "default"
    )
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    histogram_name: Optional[str] = "channel_histograms"

    # TODO: use this pydantic model to check that histograms actually exist

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        """Validate the custom normalization parameters."""
        # Extract values
        mode = self.mode
        lower_percentile = self.lower_percentile
        upper_percentile = self.upper_percentile
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        # Verify that percentiles are only provided with "custom" or "histogram" mode
        if mode not in ["custom", "histogram"]:
            if lower_percentile is not None:
                raise ValueError(
                    f"Mode='{mode}' but {lower_percentile=}.\n"
                    "Hint: set mode='custom' or mode='histogram'."
                )
            if upper_percentile is not None:
                raise ValueError(
                    f"Mode='{mode}' but {upper_percentile=}.\n"
                    "Hint: set mode='custom' or mode='histogram'."
                )
        # Verify that bounds are only provided with "custom" mode
        if mode != "custom":
            if lower_bound is not None:
                raise ValueError(
                    f"Mode='{mode}' but {lower_bound=}. Hint: set mode='custom'."
                )
            if upper_bound is not None:
                raise ValueError(
                    f"Mode='{mode}' but {upper_bound=}. Hint: set mode='custom'."
                )

        # The only valid options are:
        # 1. Both percentiles are set and both bounds are unset
        # 2. Both bounds are set and both percentiles are unset
        are_percentiles_set = (
            lower_percentile is not None,
            upper_percentile is not None,
        )
        are_bounds_set = (
            lower_bound is not None,
            upper_bound is not None,
        )
        if len(set(are_percentiles_set)) != 1:
            raise ValueError(
                "Both lower_percentile and upper_percentile must be set together."
            )
        if len(set(are_bounds_set)) != 1:
            raise ValueError("Both lower_bound and upper_bound must be set together")
        if lower_percentile is not None and lower_bound is not None:
            raise ValueError(
                "You cannot set both explicit bounds and percentile bounds "
                "at the same time. Hint: use only one of the two options."
            )

        return self

    @property
    def use_default_normalization(self) -> bool:
        """Determine whether function should apply its internal normalization.

        If mode is set to other than `default`, don't apply internal
        normalization
        """
        return self.mode == "default"


class NormalizedChannelInputModel(ChannelInputModel):
    """Channel input with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
        normalize: Validator to handle different normalization scenarios.
    """

    normalize: CustomNormalizer = Field(default_factory=CustomNormalizer)

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        """Get omero channel from zarr file"""
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {e!s}"
            )
            return None

    def get_histogram(self, zarr_url, histogram_name) -> Histogram:
        """Get histogram from zarr file"""
        try:
            omezarr = open_ome_zarr_container(zarr_url)
            channel_histograms = omezarr.get_table(histogram_name)
            adata = channel_histograms.anndata
            histogram_dict = anndata_to_histograms(adata)
            histogram = histogram_dict[self.label]
            return histogram
        except Exception as e:
            logger.error(f"An error occurred while getting the histogram: {e}")
            return None

    def update_normalization_from_omero(self, zarr_url) -> None:
        """Load omero channel and update the normalization parameters."""
        if self.normalize.mode == "omero":
            omero_channel = self.get_omero_channel(zarr_url)
            if omero_channel is None:
                raise ValueError("Mode='omero' but omero_channel is not found.")
            self.normalize = CustomNormalizer(
                mode="custom",
                lower_bound=omero_channel.window.start,
                upper_bound=omero_channel.window.end,
            )

    def update_normalization_from_histogram(self, zarr_url) -> None:
        """Load histogram and update the normalization parameters."""
        if self.normalize.mode == "histogram":
            histogram = self.get_histogram(zarr_url, self.normalize.histogram_name)
            if histogram is None:
                raise ValueError("Mode='histogram' but histogram is not found.")
            percentile_values = histogram.get_quantiles(
                [
                    self.normalize.lower_percentile / 100,
                    self.normalize.upper_percentile / 100,
                ]
            )
            self.normalize = CustomNormalizer(
                mode="custom",
                lower_bound=percentile_values[0],
                upper_bound=percentile_values[1],
            )


def normalize_channels(
    x: np.ndarray,
    normalizers: Sequence[CustomNormalizer],
    channel_axis: int = 0,
) -> np.ndarray:
    """Normalize an input array.

    Args:
        x: numpy array, where the first axis is c.
        normalizers: CustomNormalizer objects for each channel.
        omero_channels: OmeroChannel object for each channel.
        channel_axis: Axis of the channels in the input array.
    """
    for i, normalize in enumerate(normalizers):
        x[i] = normalize_channel(
            x=x[i],
            normalize=normalize,
        )
    return x


def normalize_channel(
    x: np.ndarray,
    normalize: CustomNormalizer,
) -> np.ndarray:
    """Normalize a single channel of an image."""
    if normalize.mode == "omero":
        raise ValueError(
            "Normalization mode 'omero' not supported for this function.\n"
            "Hint: First run:\n"
            "NormalizedChannelInputModel.update_normalization_from_omero(zarr_url)"
        )
    elif normalize.mode == "histogram":
        raise ValueError(
            "Normalization mode 'histogram' not supported for this function.\n"
            "Hint: First run:\n"
            "NormalizedChannelInputModel.update_normalization_from_histogram(zarr_url)"
        )
    elif normalize.mode == "custom":
        x = normalized_image(
            x,
            lower_p=normalize.lower_percentile,
            upper_p=normalize.upper_percentile,
            lower_bound=normalize.lower_bound,
            upper_bound=normalize.upper_bound,
        )
    return x


def normalized_image(
    img: np.ndarray,
    invert: bool = False,
    lower_p: float = 1.0,
    upper_p: float = 99.0,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
):
    """Normalize a single channel image.

    Based on
    https://github.com/MouseLand/cellpose/blob/...
    ...4f5661983c3787efa443bbbd3f60256f4fd8bf53/cellpose/transforms.py#L375
    """
    img = img.astype(np.float32)
    if lower_p is not None:
        # ptp can still give nan's with weird images
        i99 = np.percentile(img, upper_p)
        i1 = np.percentile(img, lower_p)
        if i99 - i1 > +1e-3:  # np.ptp(img[k]) > 1e-3:
            img = normalize_percentile(img, lower=lower_p, upper=upper_p)
            if invert:
                img = -1 * img + 1
        else:
            img = 0
    elif lower_bound is not None:
        if upper_bound - lower_bound > +1e-3:
            img = normalize_bounds(img, lower=lower_bound, upper=upper_bound)
            if invert:
                img = -1 * img + 1
        else:
            img = 0
    else:
        raise ValueError("No normalization mode specified")
    return img


def normalize_percentile(Y: np.ndarray, lower: float = 1, upper: float = 99):
    """Normalize image so 0.0 is lower percentile and 1.0 is upper percentile.

    Percentiles are passed as floats (must be between 0 and 100)

    Args:
        Y: The image to be normalized
        lower: Lower percentile
        upper: Upper percentile

    """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X


def normalize_bounds(Y: np.ndarray, lower: int = 0, upper: int = 65535):
    """Normalize image so 0.0 is lower value and 1.0 is upper value.

    Args:
        Y: The image to be normalized
        lower: Lower normalization value
        upper: Upper normalization value

    """
    X = Y.copy()
    X = (X - lower) / (upper - lower)
    return X
