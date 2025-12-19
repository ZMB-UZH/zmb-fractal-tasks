"""Fractal task to do absolutely nothing..."""

from pydantic import validate_call


@validate_call
def do_nothing(
    *,
    zarr_url: str,
):
    """Do noting...

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
            (standard argument for Fractal tasks, managed by Fractal server).
    """
    pass


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=do_nothing)
