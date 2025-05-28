from zmb_fractal_tasks.aggregate_plate_histograms import (
    aggregate_plate_histograms,
)
from zmb_fractal_tasks.calculate_histograms import (
    calculate_histograms,
)


def test_aggregate_plate_histograms(tmpdir, zarr_path):
    calculate_histograms(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        level="2",
        omero_percentiles=[1, 99],
    )
    aggregate_plate_histograms(
        zarr_urls=[str(zarr_path / "B" / "03" / "0")],
        zarr_dir=str(tmpdir),
        omero_percentiles=[1, 99],
    )
    # TODO: Check outputs
