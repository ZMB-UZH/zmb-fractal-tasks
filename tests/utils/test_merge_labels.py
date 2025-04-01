import pytest

from zmb_fractal_tasks.utils.merge_labels import merge_labels


def test_merge_labels(temp_dir):
    merge_labels(
        zarr_url_origin=str(
            temp_dir
            / "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
            / "B"
            / "03"
            / "0"
        ),
        zarr_url_target=str(
            temp_dir
            / "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr"
            / "B"
            / "03"
            / "0"
        ),
        label_names_to_copy=["wf_2_labels", "wf_3_labels"],
    )
    merge_labels(
        zarr_url_origin=str(
            temp_dir
            / "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr"
            / "B"
            / "03"
            / "0"
        ),
        zarr_url_target=str(
            temp_dir
            / "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr"
            / "B"
            / "03"
            / "0"
        ),
        label_names_to_copy=["wf_2_labels"],
    )
    # TODO: Check outputs
