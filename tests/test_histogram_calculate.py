from zmb_fractal_tasks.histogram_calculate import histogram_calculate


def test_histogram_calculate(zarr_path):
    histogram_calculate(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        pyramid_level="2",
        update_display_range=True,
        display_range_percentiles=[1, 99],
    )
    # TODO: Check outputs
