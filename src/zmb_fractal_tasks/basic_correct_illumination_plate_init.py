"""Fractal task to perform illumination correction for a plate using BaSiC."""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

import dask.array as da
import numpy as np
from basicpy import BaSiC
from ngio import open_ome_zarr_container
from pydantic import BaseModel, Field, validate_call


class OutputOptions(BaseModel):
    """Options for output."""

    overwrite_illumination_profiles: bool = True
    """If True, overwrite existing illumination profiles of the same name.
    If False, an error is raised if illumination profiles already exist."""
    overwrite_input_image: bool = True
    """If True, overwrite the input image. If False, create a new well
    sub-group to store the corrected image."""
    new_well_subgroup_suffix: str = "illumination_corrected"
    """Suffix to add to original well sub-group name. Only used if
    overwrite_input_image is False."""
    subtract_median_baseline: bool = False
    """If True, do a background subtraction by subtracting the median of
    all baseline values from the corrected image."""


class CoreBaSiCParameters(BaseModel):
    """Core Parameters for BaSiC calculation."""

    n_images_sampled: int = 256
    """Number of images to sample for illumination correction. If there
    are less images available than n_images, all available images will be
    used."""
    get_darkfield: bool = False
    """If True, calculate darkfield correction in addition to flatfield
    correction."""
    smoothness_flatfield: float = 1.0
    """Smoothing parameter for flatfield.
    (Weight of the flatfield term in the Lagrangian.)"""
    smoothness_darkfield: float = 1.0
    """Smoothing parameter for darkfield.
    (Weight of the darkfield term in the Lagrangian.)"""
    random_seed: Optional[int] = None
    """Integer random seed to initialize random number generator. If left
    empty, it will result in non-reproducible outputs."""


class AdvancedBaSiCParameters(BaseModel):
    """Advanced Parameters for BaSiC."""

    epsilon: float = 0.1
    """Weight regularization term."""
    fitting_mode: Literal["approximate", "ladmap"] = "approximate"
    """Fitting mode for optimization.
    Must be either "approximate" or "ladmap"."""
    max_iterations: int = 500
    """Maximum number of iterations for single optimization."""
    max_mu_coef: float = 10000000.0
    """Maximum allowed value of mu, divided by the initial value."""
    max_reweight_iterations: int = 10
    """Maximum number of reweighting iterations."""
    max_reweight_iterations_baseline: int = 5
    """Maximum number of reweighting iterations for baseline."""
    mu_coef: float = 12.5
    """Coefficient for initial mu value."""
    optimization_tol: float = 0.001
    """Optimization tolerance."""
    optimization_tol_diff: float = 0.01
    """Optimization tolerance for update diff."""
    reweighting_tol: float = 0.01
    """Reweighting tolerance in mean absolute difference of images."""
    resize_params: dict[str, Any] = Field(default_factory=dict)
    """Parameters for the resize function when downsampling images."""
    rho: float = 1.5
    """Parameter rho for mu update."""
    sort_intensity: bool = False
    """Whether or not to sort the intensities of the image."""
    sparse_cost_darkfield: float = 0.01
    """Weight of the darkfield sparse term in the Lagrangian."""
    working_size: int = 128
    """Size for running computations. None means no rescaling."""


@validate_call
def basic_correct_illumination_plate_init(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    illumination_profiles_folder_name: str = "BaSiC_illumination_profiles",
    output_options: OutputOptions = OutputOptions(),  # noqa: B008
    core_basic_parameters: CoreBaSiCParameters = CoreBaSiCParameters(),  # noqa: B008
    advanced_basic_parameters: AdvancedBaSiCParameters = AdvancedBaSiCParameters(),  # noqa: B008
) -> dict[str, Any]:
    """Calculate illumination profiles and correct channels using BaSiC.

    See https://basicpy.readthedocs.io for more information on BaSiC.
    This task calculates illumination correction profiles based on a random
    sample of FOVs of the entire plate for each channel. It stores the
    calculated illumination profiles in a specified folder and corrects each
    image in the plate using these profiles.
    NOTE: All FOVs within a single image must have the same (y, x)
    dimensions. If the plate contains multiple acquisitions with differing
    tile dimensions (e.g. different cameras/binnings), images are
    automatically grouped by their (y, x) FOV dimensions and a separate set
    of illumination profiles is calculated and applied for each group.

    Args:
        zarr_urls (list[str]): List of paths or urls to the individual OME-Zarr
            images to be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir (str): Profiles will be saved in
            {zarr_dir}/{illumination_profiles_folder_name}
            (Standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder_name (str): Name of folder to save
            illumination profiles in. The folder will be created inside
            dataset folder (zarr_dir).
        output_options (OutputOptions): Options for output.
        core_basic_parameters (CoreBaSiCParameters): Core parameters for BaSiC
            illumination correction.
        advanced_basic_parameters (AdvancedBaSiCParameters): Advanced
            parameters for BaSiC illumination correction.
            See https://basicpy.readthedocs.io/en/latest/api.html
    """
    # Set illumination profiles folder
    illumination_profiles_folder = str(
        Path(zarr_dir) / illumination_profiles_folder_name
    )

    random.seed(core_basic_parameters.random_seed)

    logging.info(f"Processing {len(zarr_urls)} images")

    omezarrs = [open_ome_zarr_container(zarr_url) for zarr_url in zarr_urls]
    ngio_images = [omezarr.get_image() for omezarr in omezarrs]

    # Group images by their FOV (y, x) dimensions in pixel space. This
    # supports plates that combine multiple acquisitions with differing
    # tile dimensions: illumination profiles are calculated separately for
    # each group of images that share the same FOV dimensions. FOVs within
    # a single image must all have the same dimensions.
    # Dimensions are compared in pixel space (rather than world/physical
    # space) so that floating-point noise in pixel size metadata doesn't
    # cause otherwise-identical tile dimensions to be treated as distinct.
    image_dims = []
    for omezarr, ngio_image in zip(omezarrs, ngio_images, strict=True):
        roi_table = omezarr.get_table("FOV_ROI_table")
        dims = {
            (
                round(roi.to_pixel(ngio_image.pixel_size)["y"].length),
                round(roi.to_pixel(ngio_image.pixel_size)["x"].length),
            )
            for roi in roi_table.rois()
        }
        if len(dims) > 1:
            raise ValueError(
                f"FOVs within a single image have differing dimensions: {dims}"
            )
        image_dims.append(next(iter(dims)))

    unique_dims = sorted(set(image_dims))
    multiple_dim_groups = len(unique_dims) > 1
    if multiple_dim_groups:
        logging.info(
            f"Found {len(unique_dims)} different FOV dimensions across the "
            f"provided images: {unique_dims}. Illumination profiles will be "
            "calculated separately for each group of matching dimensions."
        )

    def dim_subfolder(dims: tuple[int, int]) -> str:
        return f"tile_{dims[0]}y_{dims[1]}x"

    # process each group of images with matching FOV dimensions
    basic_dict = {}
    for dims in unique_dims:
        group_omezarrs = []
        group_ngio_images = []
        for omezarr, ngio_image, image_dim in zip(
            omezarrs, ngio_images, image_dims, strict=True
        ):
            if image_dim == dims:
                group_omezarrs.append(omezarr)
                group_ngio_images.append(ngio_image)

        group_profiles_folder = illumination_profiles_folder
        if multiple_dim_groups:
            group_profiles_folder = str(
                Path(illumination_profiles_folder) / dim_subfolder(dims)
            )

        # get list of all channels present in this group of images
        group_wavelength_ids = [
            ngio_image.wavelength_ids for ngio_image in group_ngio_images
        ]
        group_wavelength_ids = {
            wlid for sublist in group_wavelength_ids for wlid in sublist
        }
        logging.info(
            f"Processing {len(group_wavelength_ids)} channels for FOV "
            f"dimensions {dims}: {group_wavelength_ids}"
        )

        for i, wl_id in enumerate(group_wavelength_ids):
            logging.info(
                f"Processing channel {i}/{len(group_wavelength_ids)}: {wl_id}"
            )
            fov_data_all = []
            for omezarr, ngio_image in zip(
                group_omezarrs, group_ngio_images, strict=True
            ):
                if wl_id in ngio_image.wavelength_ids:
                    channel_indices = [
                        i
                        for i, wl in enumerate(ngio_image.wavelength_ids)
                        if wl == wl_id
                    ]
                    roi_table = omezarr.get_table("FOV_ROI_table")
                    for roi in roi_table.rois():
                        for channel_idx in channel_indices:
                            roi_data = ngio_image.get_roi(
                                roi,
                                axes_order=["c", "z", "y", "x"],
                                c=channel_idx,
                                mode="dask",
                            )
                            fov_data_all.append(roi_data)
            if len(fov_data_all) >= core_basic_parameters.n_images_sampled:
                logging.info(
                    f"Using {core_basic_parameters.n_images_sampled} random images"
                    + f" out of {len(fov_data_all)}."
                )
                fov_data_sample = random.sample(
                    fov_data_all, core_basic_parameters.n_images_sampled
                )
            else:
                logging.warning(
                    f"{core_basic_parameters.n_images_sampled} images requested,"
                    + f" but only {len(fov_data_all)} available. "
                    + f"Using all {len(fov_data_all)} images."
                )
                fov_data_sample = fov_data_all
            if fov_data_sample[0].shape[1] > 1:
                # take random slice along z-axis
                logging.info("Image is z-stack, taking random slices along z-axis.")
                fov_data_sample = [
                    img[0, random.randint(0, img.shape[1] - 1), ...]
                    for img in fov_data_sample
                ]
            else:
                fov_data_sample = [img[0, 0, ...] for img in fov_data_sample]
            logging.info("Loading data...")
            basic_data = da.stack(fov_data_sample).compute()

            # calculate illumination correction profile
            logging.info("Calculating illumination correction profile...")
            basic = BaSiC(
                get_darkfield=core_basic_parameters.get_darkfield,
                smoothness_flatfield=core_basic_parameters.smoothness_flatfield,
                smoothness_darkfield=core_basic_parameters.smoothness_darkfield,
                epsilon=advanced_basic_parameters.epsilon,
                fitting_mode=advanced_basic_parameters.fitting_mode,
                max_iterations=advanced_basic_parameters.max_iterations,
                max_mu_coef=advanced_basic_parameters.max_mu_coef,
                max_reweight_iterations=advanced_basic_parameters.max_reweight_iterations,
                max_reweight_iterations_baseline=advanced_basic_parameters.max_reweight_iterations_baseline,
                mu_coef=advanced_basic_parameters.mu_coef,
                optimization_tol=advanced_basic_parameters.optimization_tol,
                optimization_tol_diff=advanced_basic_parameters.optimization_tol_diff,
                reweighting_tol=advanced_basic_parameters.reweighting_tol,
                resize_params=advanced_basic_parameters.resize_params,
                rho=advanced_basic_parameters.rho,
                sort_intensity=advanced_basic_parameters.sort_intensity,
                sparse_cost_darkfield=advanced_basic_parameters.sparse_cost_darkfield,
                working_size=advanced_basic_parameters.working_size,
            )
            basic.fit(basic_data)

            # save illumination correction profile
            logging.info("Saving illumination correction profile...")
            folder_path = Path(group_profiles_folder) / f"{wl_id}"
            if output_options.overwrite_illumination_profiles:
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=False)
            # basic.save_model(model_dir=filename, overwrite=overwrite)
            np.save(folder_path / "flatfield.npy", basic.flatfield)
            np.save(folder_path / "darkfield.npy", basic.darkfield)
            np.save(folder_path / "baseline.npy", basic.baseline)
            basic_dict[(dims, wl_id)] = basic

    logging.info("Finished processing all channels.")

    # create parallelization list for applying illumination correction
    parallelization_list = []
    for zarr_url, dims in zip(zarr_urls, image_dims, strict=True):
        zarr_profiles_folder = illumination_profiles_folder
        if multiple_dim_groups:
            zarr_profiles_folder = str(
                Path(illumination_profiles_folder) / dim_subfolder(dims)
            )
        init_args = {
            "illumination_profiles_folder": zarr_profiles_folder,
            "subtract_median_baseline": output_options.subtract_median_baseline,
            "overwrite_input_image": output_options.overwrite_input_image,
            "new_well_subgroup_suffix": output_options.new_well_subgroup_suffix,
        }
        parallelization_list.append(
            {
                "zarr_url": zarr_url,
                "init_args": init_args,
            }
        )
    logging.info("Returning parallelization list for applying illumination correction:")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=basic_correct_illumination_plate_init)
