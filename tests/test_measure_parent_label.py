from zmb_fractal_tasks.measure_parent_label import measure_parent_label


def test_measure_parent_label(zarr_MIP_path):
    measure_parent_label(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        output_table_name="nuclei_features",
        label_name="nuclei",
        parent_label_names=["wf_2_labels", "wf_3_labels"],
        level="0",
        roi_table_name="FOV_ROI_table",
        append=False,
        overwrite=True,
    )
    # TODO: Check outputs
