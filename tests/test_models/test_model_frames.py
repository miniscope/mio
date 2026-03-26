from unittest.mock import MagicMock
from pathlib import Path

import pytest
import numpy as np

import mio.io.video
import mio.models.frames
from mio.models.frames import NamedFrame, NamedVideo


def test_export_image_frame(monkeypatch):
    mock_imwrite = MagicMock()
    monkeypatch.setattr(mio.models.frames.cv2, "imwrite", mock_imwrite)

    image_frame = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # Create instance of NamedFrame with a single image
    named_frame = NamedFrame(name="test", frame=image_frame)
    # Call the export method
    named_frame.export("output_path", True)
    # Check that cv2.imwrite was called correctly
    mock_imwrite.assert_called_once_with("output_path_test.png", image_frame)


def test_export_video_frame(monkeypatch):
    mock_VideoWriter = MagicMock()
    mock_instance = MagicMock()
    mock_VideoWriter.return_value = mock_instance
    monkeypatch.setattr(mio.io, "VideoWriter", mock_VideoWriter)

    frames = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(10)]

    # Create instance of NamedFrame with a video
    named_frame = NamedVideo(name="test", video=frames)
    # Call the export method
    named_frame.export(output_path="output_path", fps=20, suffix=True)

    # Verify init_video was called with correct parameters
    mock_VideoWriter.assert_called_once_with(path=Path("output_path_test.avi"), fps=20)

    assert mock_instance.write_frame.call_count == len(frames)


def test_invalid_frame_type_raises_exception(monkeypatch):
    mock_VideoWriter = MagicMock()
    monkeypatch.setattr(mio.io.video, "VideoWriter", mock_VideoWriter)

    # Test with an invalid type
    with pytest.raises(ValueError):
        named_frame = NamedFrame(name="test", frame=12345)
        named_frame.export("output_path", True)

    # Test with a list containing non-ndarray elements
    with pytest.raises(ValueError):
        named_frame = NamedFrame(name="test", frame=[123, 456])
        named_frame.export("output_path", True)

    # Ensure that no write methods are called
    mock_VideoWriter.write_frame.assert_not_called()
    mock_VideoWriter.assert_not_called()
