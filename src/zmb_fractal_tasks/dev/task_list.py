"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    CompoundTask,
    NonParallelTask,
    ParallelTask,
)

AUTHORS = "Flurin Sturzenegger"
DOCS_LINK = None
INPUT_MODELS = []

TASK_LIST = [
    NonParallelTask(
        name="Aggregate all channel histograms for plate",
        executable="aggregate_plate_histograms.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Percentiles", "Histogram", "Normalization"],
    ),
    CompoundTask(
        name="BaSiC: Calculate and apply illumination correction for plate",
        input_types={"illumination_corrected": False},
        executable_init="basic_correct_illumination_plate_init.py",
        executable="basic_apply_illumination_profile.py",
        output_types={"illumination_corrected": True},
        meta_init={"cpus_per_task": 8, "mem": 32000},
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
    CompoundTask(
        name="Merge acquisitions along channel axis",
        executable_init="combine_acquisitions_init.py",
        executable="combine_acquisitions_parallel.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Utility",
        tags=["Merge", "Acquisitions", "Combine"],
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
