from zmb_fractal_tasks.expand_segmentation import expand_segmentation


def test_expand_segmentation(zarr_MIP_path):
    expand_segmentation(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        input_label_name="nuclei",
        expansion_distance=10,
        save_union=True,
        output_label_name_union="cells",
        save_diff=True,
        output_label_name_diff="cytoplasms",
    )
    # TODO: Check outputs
