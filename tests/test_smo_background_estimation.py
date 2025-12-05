from zmb_fractal_tasks.smo_background_estimation import (
    smo_background_estimation,
)


def test_smo_background_estimation(zarr_path):
    smo_background_estimation(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        sigma=0,
        size=7,
        subtract_background=False,
        overwrite_input_image=False,
        new_well_subgroup_suffix="BG_subtracted",
    )
    # TODO: Check outputs
