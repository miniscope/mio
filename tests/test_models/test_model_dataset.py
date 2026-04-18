import pandas as pd

import pytest

from mio.models.dataset import Recording
from numpydantic.interface.video import VideoProxy

from ..conftest import DATA_DIR


def test_recording_trim_metadata():
    vid_path = DATA_DIR / "stitch" / "video1.avi"
    vid = VideoProxy(vid_path)
    n_frames = vid.shape[0]
    metadata = pd.DataFrame({"reconstructed_frame_index": list(range(n_frames + 1))})

    with pytest.warns(UserWarning):
        recording = Recording(name="video1", type="raw", video=vid_path, metadata=metadata)

    assert recording.metadata["reconstructed_frame_index"].max() == n_frames - 1
