from zmb_fractal_tasks.basic_apply_illumination_profile import (
    basic_apply_illumination_profile,
)
from zmb_fractal_tasks.basic_calculate_illumination_profile_plate import (
    basic_calculate_illumination_profile_plate,
)


def test_basic_apply_illumination_profile(tmpdir, zarr_path):
    basic_calculate_illumination_profile_plate(
        zarr_urls=[str(zarr_path / "B" / "03" / "0")],
        zarr_dir=str(tmpdir),
        illumination_profiles_folder=str(tmpdir / "illumination_profiles"),
        n_images=200,
        overwrite_illumination_profiles=True,
        random_seed=11,
        basic_smoothness=1.0,
        calculate_darkfield=True,
    )
    basic_apply_illumination_profile(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        illumination_profiles_folder=str(tmpdir / "illumination_profiles"),
        subtract_median_baseline=False,
        overwrite_input_image=False,
        new_well_subgroup_suffix="illumination_corrected",
    )
    # TODO: Check outputs
