import pytest

from zmb_fractal_tasks.normalization_utils import (
    CustomNormalizer,
    NormalizedChannelInputModel,
)
from zmb_fractal_tasks.segment_particles import segment_particles


@pytest.mark.parametrize(
    "zarr_name",
    [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_segment_particles(temp_dir, zarr_name):
    segment_particles(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        # level="2",
        channel=NormalizedChannelInputModel(
            label="DAPI",
            normalize=CustomNormalizer(mode="default"),
        ),
    )
    # TODO: Check outputs
