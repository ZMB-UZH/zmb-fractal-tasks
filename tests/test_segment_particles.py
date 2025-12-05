from zmb_fractal_tasks.calculate_histograms import calculate_histograms
from zmb_fractal_tasks.segment_particles import segment_particles
from zmb_fractal_tasks.utils.normalization import (
    CustomNormalizer,
    NormalizedChannelInputModel,
)


def test_segment_particles(zarr_path):
    calculate_histograms(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        pyramid_level="2",
        update_display_range=True,
        display_range_percentiles=[1, 99],
    )
    segment_particles(
        zarr_url=str(zarr_path / "B" / "03" / "0"),
        # level="2",
        channel=NormalizedChannelInputModel(
            label="DAPI",
            normalize=CustomNormalizer(
                mode="histogram",
                lower_percentile=1,
                upper_percentile=99,
            ),
        ),
    )
    # TODO: Check outputs
