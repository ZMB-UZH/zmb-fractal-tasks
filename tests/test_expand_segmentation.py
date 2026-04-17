from zmb_fractal_tasks.expand_segmentation import expand_segmentation


def test_expand_segmentation(zarr_MIP_path):
    expand_segmentation(
        zarr_url=str(zarr_MIP_path / "B" / "03" / "0"),
        input_label_name="nuclei",
        expansion_distance=10,
        save_union=True,
        union_output_label_name="cells",
        save_difference=True,
        difference_output_label_name="cytoplasms",
    )
    # TODO: Check outputs
