import pytest

from zmb_fractal_tasks.calculate_percentiles import calculate_percentiles


@pytest.mark.parametrize(
    "zarr_name",
    [
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_calculate_percentiles(temp_dir, zarr_name):
    calculate_percentiles(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        level="0",
        percentiles=(0, 99),
    )
    # TODO: Check outputs
