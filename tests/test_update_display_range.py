from zmb_fractal_tasks.update_display_range import update_display_range


def test_update_display_range(zarr_path):
    update_display_range(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        pyramid_level="0",
        percentiles=(0, 99),
    )
    # TODO: Check outputs
