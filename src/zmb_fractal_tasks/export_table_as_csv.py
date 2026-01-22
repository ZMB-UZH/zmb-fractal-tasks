"""Fractal task to export a table to .csv format."""

import logging
from typing import Optional
from pathlib import Path

import pandas as pd
from pydantic import validate_call
from ngio import open_ome_zarr_container


@validate_call
def export_table_as_csv(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    tables_to_export: list[str],
    plate_layout_path: Optional[str] = None,
) -> None:
    r"""Combine and export tables from multiple OME-Zarr files to CSV.

    Args:
        zarr_urls (list[str]): List of paths or urls to the individual OME-Zarr
            images to be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir (str): Table will be exported to
            {zarr_dir}/{export_table_name}.csv
            (Standard argument for Fractal tasks, managed by Fractal server).
        tables_to_export (list[str]): Names of the tables to be exported.
        plate_layout_path (Optional[str]): Path to a CSV file containing plate
            layout information. Column names should be non-zero-padded numbers
            (e.g., 1, 2, 3, not 01, 02, 03). It should have the following
            format:
            , 1, 2, 3, ...
            A, conditionA1, conditionA2, conditionA3, ...
            B, conditionB1, conditionB2, conditionB3, ...
            ...
    """
    if plate_layout_path:
        logging.info(f"loading plate layout from {plate_layout_path}")
        plate_layout = pd.read_csv(
            plate_layout_path,
            header=0,
            index_col=0,
        )

    for table_to_export in tables_to_export:
        logging.info(f"Collecting table {table_to_export}")
        df_list = []
        for zarr_url in zarr_urls:
            ome_zarr_container = open_ome_zarr_container(zarr_url)
            table_df = ome_zarr_container.get_table(table_to_export).dataframe
            table_df = table_df.reset_index()
            # find plate and well names
            plate_name = Path(Path(zarr_url).as_posix().split(".zarr/")[0]).stem
            component = Path(zarr_url).as_posix().split(".zarr/")[1]
            well_row = component.split("/")[0]
            well_col = int(component.split("/")[1])
            well_name = well_row + f"{well_col:02d}"
            table_df["plate"] = plate_name
            table_df["well"] = well_name
            # insert plate and well columns at the front
            table_df.insert(0, "plate", table_df.pop("plate"))
            table_df.insert(1, "well", table_df.pop("well"))
            # add condition from plate layout if provided
            if plate_layout_path:
                condition = plate_layout.loc[well_row, str(well_col)]
                table_df["condition"] = condition
                table_df.insert(2, "condition", table_df.pop("condition"))

            df_list.append(table_df)

        # concatenate all dataframes
        df = pd.concat(df_list, axis=0).reset_index(drop=True)

        export_table_name = table_to_export
        logging.info(
            f"Exporting table {table_to_export} to {zarr_dir}/{export_table_name}.csv")
        output_path = Path(zarr_dir) / f"{export_table_name}.csv"
        df.to_csv(output_path)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=export_table_as_csv)
