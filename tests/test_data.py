import pytest
import pandas as pd
from mio.devices.sdcard.headers import SDBufferHeader


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
def test_header_df(wirefree_frames):
    header_df = wirefree_frames.to_df(what="headers")
    assert isinstance(header_df, pd.DataFrame)

    # check columns present
    for col in header_df.columns:
        assert col in SDBufferHeader.model_fields

    assert len(header_df) == 1937
