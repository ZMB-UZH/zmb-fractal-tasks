from zmb_fractal_tasks.basic_apply_illumination_profile import (
    InitArgsBaSiCApply,
    basic_apply_illumination_profile,
)
from zmb_fractal_tasks.basic_correct_illumination_plate_init import (
    AdvancedBaSiCParameters,
    CoreBaSiCParameters,
    OutputOptions,
    basic_correct_illumination_plate_init,
)


def test_basic_apply_illumination_profile(tmpdir, zarr_path):
    # First calculate illumination profiles
    core_parameters = CoreBaSiCParameters(
        n_images_sampled=200,
        get_darkfield=True,
        smoothness_flatfield=1.0,
        smoothness_darkfield=1.0,
        random_seed=11,
    )
    advanced_basic_parameters = AdvancedBaSiCParameters()
    output_options = OutputOptions()

    result = basic_correct_illumination_plate_init(
        zarr_urls=[str(zarr_path / "B" / "03" / "0")],
        zarr_dir=str(tmpdir),
        illumination_profiles_folder_name="illumination_profiles",
        core_basic_parameters=core_parameters,
        advanced_basic_parameters=advanced_basic_parameters,
        output_options=output_options,
    )

    # Apply illumination correction
    init_args = InitArgsBaSiCApply(
        illumination_profiles_folder=str(tmpdir / "illumination_profiles"),
        subtract_median_baseline=False,
        overwrite_input_image=False,
        new_well_subgroup_suffix="illumination_corrected",
    )

    basic_apply_illumination_profile(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        init_args=init_args,
    )
    # TODO: Check outputs
