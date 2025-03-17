import pytest

from zmb_fractal_tasks.basic_apply_illumination_profile import (
    basic_apply_illumination_profile,
)
from zmb_fractal_tasks.basic_calculate_illumination_profile_plate import (
    basic_calculate_illumination_profile_plate,
)


@pytest.mark.parametrize(
    "zarr_name",
    [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_basic_apply_illumination_profile(temp_dir, zarr_name):
    basic_calculate_illumination_profile_plate(
        zarr_urls=[str(temp_dir / zarr_name / "B" / "03" / "0")],
        zarr_dir=str(temp_dir),
        illumination_profiles_folder=str(temp_dir / "illumination_profiles"),
        n_images=200,
        overwrite=True,
        random_seed=11,
        basic_smoothness=1.0,
        get_darkfield=True,
    )
    basic_apply_illumination_profile(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        illumination_profiles_folder=str(temp_dir / "illumination_profiles"),
        subtract_median_baseline=False,
        new_well_sub_group=None,
    )
    # TODO: Check outputs
