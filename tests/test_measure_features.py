import pytest

from zmb_fractal_tasks.from_fractal_tasks_core.channels import ChannelInputModel
from zmb_fractal_tasks.measure_features import measure_features


@pytest.mark.parametrize(
    "zarr_name",
    [
        # "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
    ],
)
def test_measure_features(temp_dir, zarr_name):
    measure_features(
        zarr_url=str(temp_dir / zarr_name / "B" / "03" / "0"),
        output_table_name="nuclei_features",
        label_name="nuclei",
        annotation_label_names=["wf_2_labels", "wf_3_labels"],
        shortest_distance_label_names=["wf_4_labels"],
        channels_to_include=[ChannelInputModel(label="DAPI")],
        channels_to_exclude=None,
        structure_props=["area"],
        intensity_props=["intensity_total"],
        level="0",
        overwrite=True,
    )
    # TODO: Check outputs
