import pytest
from ngio.utils import download_ome_zarr_dataset


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    # Create a temporary directory that lasts for the session
    base_temp = tmp_path_factory.mktemp("data")

    # Download files from Zenodo into the temporary directory
    download_ome_zarr_dataset("CardiomyocyteTiny", download_dir=base_temp)
    download_ome_zarr_dataset("CardiomyocyteSmallMip", download_dir=base_temp)

    return base_temp
