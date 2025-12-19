from zmb_fractal_tasks.basic_correct_illumination_plate_init import (
    AdvancedBaSiCParameters,
    AdvancedCorrectionParameters,
    CoreBaSiCParameters,
    basic_correct_illumination_plate_init,
)


def test_basic_calculate_illumination_profile_plate(tmpdir, zarr_path):
    core_parameters = CoreBaSiCParameters(
        n_images_sampled=200,
        get_darkfield=True,
        smoothness_flatfield=1.0,
        smoothness_darkfield=1.0,
        random_seed=11,
    )
    advanced_basic_parameters = AdvancedBaSiCParameters()
    advanced_correction_parameters = AdvancedCorrectionParameters(
        overwrite_input_image=True,
        subtract_median_baseline=False,
    )

    result = basic_correct_illumination_plate_init(
        zarr_urls=[str(zarr_path / "B" / "03" / "0")],
        zarr_dir=str(tmpdir),
        illumination_profiles_folder_name="illumination_profiles",
        overwrite_illumination_profiles=True,
        core_basic_parameters=core_parameters,
        advanced_basic_parameters=advanced_basic_parameters,
        advanced_correction_parameters=advanced_correction_parameters,
    )

    # Check that parallelization list is returned
    assert "parallelization_list" in result
    assert len(result["parallelization_list"]) == 1
    # TODO: Check outputs
