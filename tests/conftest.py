import os
import shutil
from pathlib import Path

import pytest
from ngio.utils import download_ome_zarr_dataset

zenodo_download_dir = Path(__file__).parent.parent / "data"
os.makedirs(zenodo_download_dir, exist_ok=True)
cardiomyocyte_tiny_source_path = download_ome_zarr_dataset(
    "CardiomyocyteTiny", download_dir=zenodo_download_dir
)

cardiomyocyte_small_mip_source_path = download_ome_zarr_dataset(
    "CardiomyocyteSmallMip", download_dir=zenodo_download_dir
)


@pytest.fixture
def zarr_3D_path(tmp_path: Path) -> Path:
    dest_path = tmp_path / (cardiomyocyte_tiny_source_path.stem + ".zarr")
    shutil.copytree(cardiomyocyte_tiny_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture
def zarr_MIP_path(tmp_path: Path) -> Path:
    dest_path = tmp_path / (cardiomyocyte_small_mip_source_path.stem + ".zarr")
    shutil.copytree(cardiomyocyte_small_mip_source_path, dest_path, dirs_exist_ok=True)
    return dest_path


@pytest.fixture(params=["zarr_3D_path", "zarr_MIP_path"])
def zarr_path(request):
    return request.getfixturevalue(request.param)
