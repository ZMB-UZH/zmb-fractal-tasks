from zmb_fractal_tasks.calculate_histograms import calculate_histograms


def test_calculate_histograms(zarr_path):
    calculate_histograms(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        level="2",
        omero_percentiles=[1, 99],
    )
    # TODO: Check outputs
