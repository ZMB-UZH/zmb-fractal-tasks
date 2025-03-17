import pytest

from zmb_fractal_tasks.smo_background_estimation import (
    smo_background_estimation,
)


@pytest.mark.parametrize(
    "zarr_name",
    [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_smo_background_estimation(temp_dir, zarr_name):
    smo_background_estimation(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        sigma=0,
        size=7,
        subtract_background=False,
        new_well_sub_group=None,
    )
    # TODO: Check outputs
