from zmb_fractal_tasks.basic_calculate_illumination_profile_plate import (
    basic_calculate_illumination_profile_plate,
)


def test_basic_calculate_illumination_profile_plate(tmpdir, zarr_path):
    basic_calculate_illumination_profile_plate(
        zarr_urls=[str(zarr_path / "B" / "03" / "0")],
        zarr_dir=str(tmpdir),
        illumination_profiles_folder=str(tmpdir / "illumination_profiles"),
        n_images=200,
        overwrite=True,
        random_seed=11,
        basic_smoothness=1.0,
        get_darkfield=True,
    )
    # TODO: Check outputs
