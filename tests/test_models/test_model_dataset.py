import pandas as pd
import pytest
from numpydantic.interface.video import VideoProxy

from mio.models.dataset import Recording

from ..conftest import DATA_DIR


def test_recording_trim_metadata():
    vid_path = DATA_DIR / "stitch" / "video1.avi"
    metadata = pd.read_csv(vid_path.with_suffix(".csv"))
    metadata_list = metadata.values.tolist()
    metadata_list.append(metadata_list[-1])
    metadata = pd.DataFrame(metadata_list, columns=metadata.columns)
    metadata.iloc[-1]["reconstructed_frame_index"] += 1
    vid = VideoProxy(vid_path)
    n_frames = vid.shape[0]

    with pytest.warns(UserWarning):
        recording = Recording(name="video1", type="raw", video=vid_path, metadata=metadata)

    assert recording.metadata["reconstructed_frame_index"].max() == n_frames - 1
