from pathlib import Path

import numpy as np
import pytest

from mio.io.video import VideoWriter


@pytest.mark.parametrize("force", [False, True])
def test_write_video_force(force: bool, tmp_path: Path):
    """
    When force is true, overwrite an existing file. Otherwise, raise an error.
    """
    out_file = tmp_path / "test.avi"
    out_file.write_bytes(bytes(100))

    if force:
        writer = VideoWriter(out_file, 30, force=force)
        writer.write_frame(np.ones((100, 100, 1)))
        writer.close()
        data = out_file.read_bytes()
        assert len(data) != 100
    else:
        with pytest.raises(FileExistsError):
            VideoWriter(out_file, 30, force=force)
