import pytest

from zmb_fractal_tasks.calculate_histograms import calculate_histograms
from zmb_fractal_tasks.segment_particles import segment_particles
from zmb_fractal_tasks.utils.normalization import (
    CustomNormalizer,
    NormalizedChannelInputModel,
)


@pytest.mark.parametrize(
    "zarr_name",
    [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_segment_particles(temp_dir, zarr_name):
    calculate_histograms(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        level="2",
        omero_percentiles=[1, 99],
    )
    segment_particles(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        # level="2",
        channel=NormalizedChannelInputModel(
            label="DAPI",
            normalize=CustomNormalizer(
                mode="histogram",
                lower_percentile=1,
                upper_percentile=99,
            ),
        ),
    )
    # TODO: Check outputs
