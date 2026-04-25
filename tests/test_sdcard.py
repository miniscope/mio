import pytest

from mio.devices.sdcard.headers import SectorConfig
import numpy as np


@pytest.fixture
def random_sectorconfig():
    return SectorConfig(
        header=np.random.randint(0, 2048),
        config=np.random.randint(0, 2048),
        data=np.random.randint(0, 2048),
        size=np.random.randint(0, 2048),
    )


def test_get_sector_position(random_sectorconfig):
    """
    Sectorconfig should get the correct values and be able to compute positions from the size
    """
    sectors = random_sectorconfig
    assert sectors.header_pos == sectors.header * sectors.size
    assert sectors.config_pos == sectors.config * sectors.size
    assert sectors.data_pos == sectors.data * sectors.size

    # We should raise an attribute error if we try and get a nonexistent one
    with pytest.raises(AttributeError):
        print(sectors.mybig_undefined_pos)
