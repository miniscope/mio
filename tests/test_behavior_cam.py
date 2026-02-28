import csv
from pathlib import Path

import numpy as np
import pytest

from mio.behavior_cam import BehaviorCam
from mio.io import VideoReader
from mio.models.usbcam import USBCameraRecordingConfig

NUM_TEST_FRAMES = 10
TEST_WIDTH = 1280
TEST_HEIGHT = 720
TEST_FPS = 20


def _make_npz(path: Path, num_frames: int = NUM_TEST_FRAMES) -> Path:
    """Generate a synthetic .npz matching elp-camera config."""
    frames = np.random.default_rng().integers(
        0, 255, size=(num_frames, TEST_HEIGHT, TEST_WIDTH, 3), dtype=np.uint8
    )
    timestamps = np.arange(num_frames, dtype=np.float64) / TEST_FPS
    np.savez(path, frames=frames, timestamps=timestamps)
    return path


def test_capture_with_mock(set_usbcam_input, tmp_path):
    """Test that BehaviorCam.capture() produces video and CSV output using mock data."""
    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path)
    config.ntp_server = None

    npz_path = _make_npz(tmp_path / "test_input.npz")
    set_usbcam_input(npz_path)

    behavior_cam = BehaviorCam(recording_config=config, camera_index=0)
    behavior_cam.capture(show_video=False)

    video_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
    csv_files = list(tmp_path.glob("*.csv"))

    assert len(video_files) == 1, f"Expected 1 video file, found {len(video_files)}"
    assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"

    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert "frame_index" in reader.fieldnames
    assert "unix_time" in reader.fieldnames
    assert len(rows) == NUM_TEST_FRAMES, f"Expected {NUM_TEST_FRAMES} rows, got {len(rows)}"


def test_capture_binary_export(set_usbcam_input, tmp_path):
    """Test that capture_binary saves raw frames to .npz."""
    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path)
    config.ntp_server = None

    npz_path = _make_npz(tmp_path / "test_input.npz")
    set_usbcam_input(npz_path)

    binary_output = tmp_path / "export.npz"
    behavior_cam = BehaviorCam(recording_config=config, camera_index=0)
    behavior_cam.capture(show_video=False, capture_binary=binary_output)

    assert binary_output.exists()

    data = np.load(binary_output)
    assert "frames" in data
    assert "timestamps" in data
    assert data["frames"].shape[0] == NUM_TEST_FRAMES
    assert data["timestamps"].shape[0] == NUM_TEST_FRAMES


STRESS_NUM_FRAMES = 1000


@pytest.fixture(params=["cv2", "skvideo"])
def backend_config(request, tmp_path) -> USBCameraRecordingConfig:
    """ELP camera config with parameterized backend for comparison testing."""
    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path)
    config.ntp_server = None
    config.backend = request.param
    if request.param == "skvideo":
        config.codec = "libx264"
        config.pix_fmt = "yuv420p"
    return config


@pytest.mark.xfail(reason="skvideo backend drops frames — CSV count != video frame count", strict=False)
def test_frame_count_matches_csv(backend_config, set_usbcam_input, tmp_path):
    """Verify video frame count matches CSV row count for each backend (1000 frames).

    Known bug: skvideo backend may produce fewer video frames than CSV rows.
    """
    npz_path = _make_npz(tmp_path / "stress_input.npz", num_frames=STRESS_NUM_FRAMES)
    set_usbcam_input(npz_path)

    behavior_cam = BehaviorCam(recording_config=backend_config, camera_index=0)
    behavior_cam.capture(show_video=False)

    video_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
    csv_files = list(tmp_path.glob("*.csv"))

    assert len(video_files) == 1, f"Expected 1 video file, found {len(video_files)}"
    assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"

    with open(csv_files[0]) as f:
        csv_row_count = sum(1 for _ in csv.DictReader(f))

    reader = VideoReader(str(video_files[0]))
    video_frame_count = sum(1 for _ in reader.read_frames())
    reader.release()

    assert video_frame_count == csv_row_count, (
        f"Frame count mismatch ({backend_config.backend} backend): "
        f"video has {video_frame_count} frames but CSV has {csv_row_count} rows"
    )


REALTIME_NUM_FRAMES = 1000


@pytest.mark.xfail(reason="skvideo backend drops frames — CSV count != video frame count", strict=False)
def test_frame_count_realtime(backend_config, set_usbcam_input, tmp_path):
    """Verify frame count with realtime replay to simulate real camera timing.

    Known bug: skvideo backend may produce fewer video frames than CSV rows
    under real-time write pressure.
    """
    npz_path = _make_npz(tmp_path / "realtime_input.npz", num_frames=REALTIME_NUM_FRAMES)
    set_usbcam_input(npz_path, realtime=True)

    behavior_cam = BehaviorCam(recording_config=backend_config, camera_index=0)
    behavior_cam.capture(show_video=False)

    video_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
    csv_files = list(tmp_path.glob("*.csv"))

    assert len(video_files) == 1, f"Expected 1 video file, found {len(video_files)}"
    assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"

    with open(csv_files[0]) as f:
        csv_row_count = sum(1 for _ in csv.DictReader(f))

    reader = VideoReader(str(video_files[0]))
    video_frame_count = sum(1 for _ in reader.read_frames())
    reader.release()

    assert video_frame_count == csv_row_count, (
        f"Frame count mismatch ({backend_config.backend} backend, realtime): "
        f"video has {video_frame_count} frames but CSV has {csv_row_count} rows"
    )
