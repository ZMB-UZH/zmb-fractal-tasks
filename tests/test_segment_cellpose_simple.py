from zmb_fractal_tasks.aggregate_plate_histograms import aggregate_plate_histograms
from zmb_fractal_tasks.calculate_histograms import calculate_histograms
from zmb_fractal_tasks.segment_cellpose_simple import segment_cellpose_simple
from zmb_fractal_tasks.utils.normalization import (
    CustomNormalizer,
    NormalizedChannelInputModel,
)


def test_segment_cellpose_simple(zarr_MIP_path):
    calculate_histograms(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        level="2",
        omero_percentiles=[1, 99],
    )
    aggregate_plate_histograms(
        zarr_urls=[str(zarr_MIP_path / "B" / "03" / "0")],
        zarr_dir=str(zarr_MIP_path),
        omero_percentiles=[1, 99],
    )
    segment_cellpose_simple(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        level="2",
        channel=NormalizedChannelInputModel(
            label="DAPI",
            normalize=CustomNormalizer(
                mode="histogram",
                lower_percentile=1,
                upper_percentile=99,
                histogram_name="channel_histograms_plate",
            ),
        ),
        diameter=60.0,
    )
    # TODO: Check outputs
