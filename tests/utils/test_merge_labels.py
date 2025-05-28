from zmb_fractal_tasks.utils.merge_labels import merge_labels


def test_merge_labels(zarr_3D_path, zarr_MIP_path):
    merge_labels(
        zarr_url_origin=str(zarr_MIP_path / "B" / "03" / "0"),
        zarr_url_target=str(zarr_3D_path / "B" / "03" / "0"),
        label_names_to_copy=["wf_2_labels", "wf_3_labels"],
    )
    merge_labels(
        zarr_url_origin=str(zarr_MIP_path / "B" / "03" / "0"),
        zarr_url_target=str(zarr_3D_path / "B" / "03" / "0"),
        label_names_to_copy=["wf_2_labels"],
    )
    # TODO: Check outputs
