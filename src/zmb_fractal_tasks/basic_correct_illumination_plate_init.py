"""Fractal task to perform illumination correction for a plate using BaSiC."""

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import dask.array as da
import numpy as np
from basicpy import BaSiC
from ngio import open_ome_zarr_container
from pydantic import BaseModel, validate_call


class OutputOptions(BaseModel):
    """Options for output

    Args:
        overwrite_illumination_profiles: If True, overwrite existing
            illumination profiles of the same name. If False, an error is
            raised if illumination profiles already exist.
        overwrite_input_image: If True, overwrite the input image. If False,
            create a new well sub-group to store the corrected image.
        new_well_subgroup_suffix: Suffix to add to original well sub-group
            name. Only used if overwrite_input_image is False.
        subtract_median_baseline: If True, do a background subtraction by
            subtracting the median of all baseline values from the corrected
            image.
    """

    overwrite_illumination_profiles: bool = (True,)
    overwrite_input_image: bool = True
    new_well_subgroup_suffix: str = "illumination_corrected"
    subtract_median_baseline: bool = False


class CoreBaSiCParameters(BaseModel):
    """Core Parameters for BaSiC calculation

    Args:
        n_images_sampled: Number of images to sample for illumination
            correction. If there are less images available than n_images, all
            available images will be used.
        get_darkfield: If True, calculate darkfield correction.
        smoothness_flatfield: Smoothing parameter for flatfield.
            (Weight of the flatfield term in the Lagrangian.)
        smoothness_darkfield: Smoothing parameter for darkfield.
            (Weight of the darkfield term in the Lagrangian.)
        random_seed: Integer random seed to initialize random number generator.
            None will result in non-reproducibel outputs.
    """

    n_images_sampled: int = 256
    get_darkfield: bool = False
    smoothness_flatfield: float = 1.0
    smoothness_darkfield: float = 1.0
    random_seed: Optional[int] = None


class AdvancedBaSiCParameters(BaseModel):
    """Advanced Parameters for BaSiC

    Args:
        autosegment: When not False, automatically segment the image before
            fitting. When True, threshold_otsu from scikit-image is used and
            the brighter pixels are taken.When a callable is given, it is used
            as the segmentation function.
        autosegment_margin: Margin of the segmentation mask to the thresholded
            region.
        epsilon: Weight regularization term.
        max_iterations: Maximum number of iterations for single optimization.
        max_mu_coef: Maximum allowed value of mu, divided by the initial value.
        max_reweight_iterations: Maximum number of reweighting iterations.
        max_reweight_iterations_baseline: Maximum number of reweighting
            iterations for baseline.
        max_workers: Maximum number of threads used for processing.
        mu_coef: Coefficient for initial mu value.
        optimization_tol: Optimization tolerance.
        optimization_tol_diff: Optimization tolerance for update diff.
        reweighting_tol: Reweighting tolerance in mean absolute difference of
            images.
        rho: Parameter rho for mu update.
        sort_intensity: Whether or not to sort the intensities of the image.
        sparse_cost_darkfield: Weight of the darkfield sparse term in the
            Lagrangian.
        working_size: Size for running computations. None means no rescaling.
    """

    autosegment: bool = False
    autosegment_margin: int = 10
    epsilon: float = 0.1
    max_iterations: int = 500
    max_mu_coef: float = 10000000.0
    max_reweight_iterations: int = 10
    max_reweight_iterations_baseline: int = 5
    max_workers: int = 8
    mu_coef: float = 12.5
    optimization_tol: float = 0.001
    optimization_tol_diff: float = 0.01
    reweighting_tol: float = 0.01
    rho: float = 1.5
    sort_intensity: bool = False
    sparse_cost_darkfield: float = 0.01
    working_size: Optional[int] = 128


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
    NOTE: This assumes that all FOVs in the plate have the same dimensions.

    Args:
        zarr_urls (list[str]): List of paths or urls to the individual OME-Zarr
            images to be processed.
            (Standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir (str): Profiles will be saved in
            {zarr_dir}/{illumination_profiles_folder_name}
            (Standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder_name (str): Name of folder to save
            illumination profiles in. The folder will be created inside
            zarr_dir.
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

    # check if all FOVs have the same dimensions
    roi_dims = []
    for omezarr in omezarrs:
        roi_table = omezarr.get_table("FOV_ROI_table")
        for roi in roi_table.rois():
            roi_dims.append((roi.z_length, roi.y_length, roi.x_length))

    if not all(dim == roi_dims[0] for dim in roi_dims):
        raise ValueError("FOVs have differing dimensions")

    # get list of all channels
    wavelength_ids = [ngio_image.wavelength_ids for ngio_image in ngio_images]
    wavelength_ids = {wlid for sublist in wavelength_ids for wlid in sublist}
    logging.info(f"Processing {len(wavelength_ids)} channels: {wavelength_ids}")

    # process each channel
    basic_dict = {}
    for i, channel in enumerate(wavelength_ids):
        logging.info(f"Processing channel {i}/{len(wavelength_ids)}: {channel}")
        fov_data_all = []
        for omezarr in omezarrs:
            ngio_image = omezarr.get_image()
            if channel in ngio_image.wavelength_ids:
                channel_idx = ngio_image.wavelength_ids.index(channel)
                roi_table = omezarr.get_table("FOV_ROI_table")
                for roi in roi_table.rois():
                    roi_data = ngio_image.get_roi(
                        roi, axes_order=["c", "z", "y", "x"], c=channel_idx, mode="dask"
                    )
                    fov_data_all.append(roi_data)
        if len(fov_data_all) >= core_basic_parameters.n_images_sampled:
            logging.info(
                f"Using {core_basic_parameters.n_images_sampled} random images out of"
                + f" {len(fov_data_all)}."
            )
            fov_data_sample = random.sample(
                fov_data_all, core_basic_parameters.n_images_sampled
            )
        else:
            logging.warning(
                f"{core_basic_parameters.n_images_sampled} images requested, but only"
                + f" {len(fov_data_all)} available. "
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
            autosegment=advanced_basic_parameters.autosegment,
            autosegment_margin=advanced_basic_parameters.autosegment_margin,
            epsilon=advanced_basic_parameters.epsilon,
            max_iterations=advanced_basic_parameters.max_iterations,
            max_mu_coef=advanced_basic_parameters.max_mu_coef,
            max_reweight_iterations=advanced_basic_parameters.max_reweight_iterations,
            max_reweight_iterations_baseline=advanced_basic_parameters.max_reweight_iterations_baseline,
            max_workers=advanced_basic_parameters.max_workers,
            mu_coef=advanced_basic_parameters.mu_coef,
            optimization_tol=advanced_basic_parameters.optimization_tol,
            optimization_tol_diff=advanced_basic_parameters.optimization_tol_diff,
            reweighting_tol=advanced_basic_parameters.reweighting_tol,
            rho=advanced_basic_parameters.rho,
            sort_intensity=advanced_basic_parameters.sort_intensity,
            sparse_cost_darkfield=advanced_basic_parameters.sparse_cost_darkfield,
            working_size=advanced_basic_parameters.working_size,
        )
        basic.fit(basic_data)

        # save illumination correction profile
        logging.info("Saving illumination correction profile...")
        folder_path = Path(illumination_profiles_folder) / f"{channel}"
        if output_options.overwrite_illumination_profiles:
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=False)
        # basic.save_model(model_dir=filename, overwrite=overwrite)
        np.save(folder_path / "flatfield.npy", basic.flatfield)
        np.save(folder_path / "darkfield.npy", basic.darkfield)
        np.save(folder_path / "baseline.npy", basic.baseline)
        basic_dict[channel] = basic

    logging.info("Finished processing all channels.")

    # create parallelization list for applying illumination correction
    parallelization_list = []
    init_args = {
        "illumination_profiles_folder": illumination_profiles_folder,
        "subtract_median_baseline": output_options.subtract_median_baseline,
        "overwrite_input_image": output_options.overwrite_input_image,
        "new_well_subgroup_suffix": output_options.new_well_subgroup_suffix,
    }
    for zarr_url in zarr_urls:
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
