from zmb_fractal_tasks.basic_correct_illumination_plate_init import (
    AdvancedBaSiCParameters,
    CoreBaSiCParameters,
    OutputOptions,
    basic_correct_illumination_plate_init,
)


def test_basic_correct_illumination_plate_init(tmpdir, zarr_path):
    core_parameters = CoreBaSiCParameters(
        n_images_sampled=200,
        get_darkfield=True,
        smoothness_flatfield=1.0,
        smoothness_darkfield=1.0,
        random_seed=11,
    )
    advanced_basic_parameters = AdvancedBaSiCParameters()
    output_options = OutputOptions(
        overwrite_input_image=True,
        subtract_median_baseline=False,
    )

    result = basic_correct_illumination_plate_init(
        zarr_urls=[str(zarr_path / "B" / "03" / "0")],
        zarr_dir=str(tmpdir),
        illumination_profiles_folder_name="illumination_profiles",
        core_basic_parameters=core_parameters,
        advanced_basic_parameters=advanced_basic_parameters,
        output_options=output_options,
    )

    # Check that parallelization list is returned
    assert "parallelization_list" in result
    assert len(result["parallelization_list"]) == 1
    # TODO: Check outputs
