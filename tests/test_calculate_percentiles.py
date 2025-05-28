from zmb_fractal_tasks.calculate_percentiles import calculate_percentiles


def test_calculate_percentiles(zarr_path):
    calculate_percentiles(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        level="0",
        percentiles=(0, 99),
    )
    # TODO: Check outputs
