from zmb_fractal_tasks.from_fractal_tasks_core.channels import ChannelInputModel
from zmb_fractal_tasks.measure_features import measure_features


def test_measure_features(zarr_MIP_path):
    measure_features(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        output_table_name="nuclei_features",
        label_name="nuclei",
        channels_to_include=[ChannelInputModel(label="DAPI")],
        channels_to_exclude=None,
        structure_props=["area"],
        intensity_props=["intensity_total"],
        level="0",
        roi_table_name="FOV_ROI_table",
        append=False,
        overwrite=True,
    )
    # TODO: Check outputs
