import pytest

from zmb_fractal_tasks.aggregate_plate_histograms import (
    aggregate_plate_histograms,
)
from zmb_fractal_tasks.calculate_histograms import (
    calculate_histograms,
)


@pytest.mark.parametrize(
    "zarr_name",
    [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_basic_calculate_illumination_profile_plate(temp_dir, zarr_name):
    calculate_histograms(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        level="2",
        omero_percentiles=[1, 99],
    )
    aggregate_plate_histograms(
        zarr_urls=[str(temp_dir / zarr_name / "B" / "03" / "0")],
        zarr_dir=str(temp_dir),
        omero_percentiles=[1, 99],
    )
    # TODO: Check outputs
