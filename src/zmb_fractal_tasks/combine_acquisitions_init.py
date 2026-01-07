"""Fractal init task to combine multiple acquisitions of a plate."""

import logging
from pathlib import Path
from typing import Optional

from ngio import open_ome_zarr_plate
from pydantic import validate_call


@validate_call
def combine_acquisitions_init(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    acquisitions_to_combine: Optional[list[int]] = None,
    keep_individual_acquisitions: bool = False,
):
    """Combine multiple acquisitions of a plate into a single acquisition.

    If there are multiple acquisitions for a given well, this task will
    combine them into a single acquisition by concatenating the channels.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr images to
            be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: Profiles will be saved in
            {zarr_dir}/{illumination_profiles_folder_name}
            (Standard argument for Fractal tasks, managed by Fractal server).
        acquisitions_to_combine: Optional: List of acquisition IDs to combine.
            If left empty, all acquisitions found in the plate will be
            combined.
        keep_individual_acquisitions: If True, keep the individual acquisitions
            and add combined acquisition. If False, delete them and only keep
            the combined acquisition.
    """
    zarr_paths = [Path(url) for url in zarr_urls]
    # extract all plate roots
    plate_roots = {p.parent.parent.parent for p in zarr_paths}
    parallelization_list = []
    for plate_root in plate_roots:
        ome_zarr_plate = open_ome_zarr_plate(plate_root)
        acquisition_ids = ome_zarr_plate.acquisition_ids
        new_acquisition_id = max(acquisition_ids) + 1
        if acquisitions_to_combine is not None:
            acquisition_ids = [
                aid for aid in acquisition_ids if aid in acquisitions_to_combine
            ]
        if len(acquisition_ids) < 2:
            logging.info(
                f"Plate at {plate_root} has less than two acquisitions to combine. "
                "Skipping."
            )
            continue
        ome_zarr_plate.add_acquisition(new_acquisition_id, "combined")
        for well_path in ome_zarr_plate.wells_paths():
            row = well_path.split("/")[0]
            column = int(well_path.split("/")[1])
            acquisition_paths = []
            for acquisition_id in acquisition_ids:
                acquisition_paths.extend(
                    ome_zarr_plate.well_images_paths(
                        row=row, column=column, acquisition=acquisition_id
                    )
                )
            if len(acquisition_paths) > 1:
                ome_zarr_plate.add_image(
                    row=row,
                    column=column,
                    image_path=str(new_acquisition_id),
                    acquisition_id=new_acquisition_id,
                )
                zarr_url_new = (
                    plate_root / well_path / str(new_acquisition_id)
                ).as_posix()
                init_args = {
                    "zarr_urls_to_combine": [
                        (plate_root / p).as_posix() for p in acquisition_paths
                    ],
                    "keep_individual_acquisitions": keep_individual_acquisitions,
                }
                parallelization_list.append(
                    {
                        "zarr_url": zarr_url_new,
                        "init_args": init_args,
                    }
                )
            if not keep_individual_acquisitions:
                # remove individual acquisitions from plate metadata
                for acquisition_path in acquisition_paths:
                    ome_zarr_plate.remove_image(
                        row=row,
                        column=column,
                        image_path=str(acquisition_path.split("/")[-1]),
                    )

    logging.info("Returning parallelization list for combine_acquisitions_parallel.")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=combine_acquisitions_init)
