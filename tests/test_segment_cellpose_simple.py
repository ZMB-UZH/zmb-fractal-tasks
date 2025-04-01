import pytest

from zmb_fractal_tasks.normalization_utils import (
    CustomNormalizer,
    NormalizedChannelInputModel,
)
from zmb_fractal_tasks.segment_cellpose_simple import segment_cellpose_simple


@pytest.mark.parametrize(
    "zarr_name",
    [
        #"20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_segment_cellpose_simple(temp_dir, zarr_name):
    segment_cellpose_simple(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        level="2",
        channel=NormalizedChannelInputModel(
            label="DAPI",
            normalize=CustomNormalizer(mode="omero"),
        ),
        diameter = 60.0,
    )
    # TODO: Check outputs
