import pytest

from zmb_fractal_tasks.expand_segmentation import expand_segmentation


@pytest.mark.parametrize(
    "zarr_name",
    [
        #"20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_expand_segmentation(temp_dir, zarr_name):
    expand_segmentation(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        input_label_name="nuclei",
        expansion_distance=10,
        save_union=True,
        output_label_name_union="cells",
        save_diff=True,
        output_label_name_diff="cytoplasms",
    )
    # TODO: Check outputs
