"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    NonParallelTask,
    ParallelTask,
)

AUTHORS = "Flurin Sturzenegger"
DOCS_LINK = None
INPUT_MODELS = []

TASK_LIST = [
    ParallelTask(
        name="Add type 'marked' to image list",
        executable="do_nothing.py",
        output_types={"marked": True},
        meta={"cpus_per_task": 1, "mem": 100},
        category="Utility",
        tags=["Utility", "Type", "Types", "Image List", "Mark"],
    ),
    NonParallelTask(
        name="Aggregate all channel histograms for plate",
        executable="aggregate_plate_histograms.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Percentiles", "Histogram", "Normalization"],
    ),
    ParallelTask(
        name="BaSiC: Apply illumination profile",
        input_types={"illumination_corrected": False},
        executable="basic_apply_illumination_profile.py",
        output_types={"illumination_corrected": True},
        meta={"cpus_per_task": 8, "mem": 32000},
        category="Image Processing",
        tags=["Illumination correction", "BaSiC"],
        # docs_info="file:docs_info/thresholding_task.md",
    ),
    NonParallelTask(
        name="BaSiC: Calculate illumination profile for plate",
        executable="basic_calculate_illumination_profile_plate.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        tags=["Illumination correction", "BaSiC"],
    ),
    ParallelTask(
        name="Calculate channel-histograms for each image",
        executable="calculate_histograms.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Percentiles", "Histogram", "Normalization"],
    ),
    ParallelTask(
        name="Delete labels from OME-Zarr image",
        executable="delete_labels.py",
        meta={"cpus_per_task": 1, "mem": 500},
        category="Utility",
        tags=["Labels", "Delete"],
    ),
    ParallelTask(
        name="Update display range",
        executable="update_display_range.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Percentiles", "Histogram", "Normalization"],
    ),
    ParallelTask(
        name="Expand segmentation",
        executable="expand_segmentation.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Expand"],
    ),
    ParallelTask(
        name="Measure features",
        executable="measure_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Measure"],
    ),
    ParallelTask(
        name="Measure parent label",
        executable="measure_parent_label.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Measure"],
    ),
    ParallelTask(
        name="Measure shortest distance to label",
        executable="measure_shortest_distance.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Measure"],
    ),
    ParallelTask(
        name="Cellpose segmentation, simple",
        executable="segment_cellpose_simple.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Cellpose", "Segmentation"],
    ),
    ParallelTask(
        name="Segment particles",
        executable="segment_particles.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Spot", "Segmentation", "Allen", "Allen Cell & Structure Segmenter"],
    ),
    ParallelTask(
        name="SMO background estimation",
        executable="smo_background_estimation.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Image Processing",
        tags=["Background", "SMO", "BG", "BG Subtraction", "Background Subtraction"],
    ),
]
