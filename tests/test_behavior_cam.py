import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from mio.behavior_cam import BehaviorCam
from mio.devices.usbcam import convert_frame_for_codec, format_camera_info
from mio.io import VideoReader, VideoWriter
from mio.models.usbcam import USBCameraRecordingConfig

NUM_TEST_FRAMES = 10
TEST_WIDTH = 1280
TEST_HEIGHT = 720
TEST_FPS = 20
STRESS_NUM_FRAMES = 100



def _make_npz(path: Path, num_frames: int = NUM_TEST_FRAMES, fps: float = TEST_FPS) -> Path:
    """Generate a synthetic .npz matching elp-camera config."""
    frames = np.random.default_rng().integers(
        0, 255, size=(num_frames, TEST_HEIGHT, TEST_WIDTH, 3), dtype=np.uint8
    )
    timestamps = np.arange(num_frames, dtype=np.float64) / fps
    np.savez(path, frames=frames, timestamps=timestamps)
    return path


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

def test_videowriter_close_writes_moov_atom(tmp_path):
    """Verify VideoWriter.close() produces a valid mp4 with moov atom.

    Without proper close (stdin.close + wait), FFmpeg may exit before
    writing the moov atom, making the file unreadable.
    """
    video_path = tmp_path / "test.mp4"
    writer = VideoWriter(
        path=video_path,
        fps=20,
        output_dict={"-vcodec": "libx264", "-f": "mp4", "-pix_fmt": "yuv420p", "-vsync": "0"},
        backend="skvideo",
    )

    frames = np.random.default_rng().integers(
        0, 255, size=(NUM_TEST_FRAMES, TEST_HEIGHT, TEST_WIDTH, 3), dtype=np.uint8
    )
    for frame in frames:
        writer.write_frame(frame)

    writer.close()

    # If moov atom is missing, VideoCapture will fail to open or report 0 frames
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Failed to open video — moov atom likely missing"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == NUM_TEST_FRAMES, (
        f"Expected {NUM_TEST_FRAMES} frames, got {frame_count}"
    )

def test_format_camera_info():
    info = {"name": "Camera 0", "resolution": "1280x720", "fps": 30}
    result = format_camera_info(0, info)
    assert "[0]" in result
    assert "1280x720" in result
    assert "30 fps" in result

def test_usbcam_mock_read_all_frames(set_usbcam_input, tmp_path):
    """USBCamMock should yield all frames then raise EndOfRecordingException."""
    from mio.devices.mocks import USBCamMock
    from mio.exceptions import EndOfRecordingException

    npz_path = _make_npz(tmp_path / "mock.npz", num_frames=5)
    set_usbcam_input(npz_path)

    mock = USBCamMock()
    assert mock.isOpened()

    frames_read = 0
    while True:
        try:
            ret, frame = mock.read()
            assert ret is True
            assert frame.shape == (TEST_HEIGHT, TEST_WIDTH, 3)
            frames_read += 1
        except EndOfRecordingException:
            break

    assert frames_read == 5

def test_usbcam_mock_set_get(tmp_path):
    """USBCamMock.set()/get() should store and retrieve properties."""
    from mio.devices.mocks import USBCamMock

    # Need DATA_FILE set; use a minimal npz
    npz_path = tmp_path / "mock_props.npz"
    np.savez(npz_path, frames=np.zeros((1, 2, 2, 3), dtype=np.uint8), timestamps=np.array([0.0]))
    USBCamMock.DATA_FILE = npz_path
    mock = USBCamMock()

    assert mock.get(cv2.CAP_PROP_FPS) == 0.0  # default
    mock.set(cv2.CAP_PROP_FPS, 30.0)
    assert mock.get(cv2.CAP_PROP_FPS) == 30.0

def test_usbcam_mock_release(set_usbcam_input, tmp_path):
    """USBCamMock.release() should mark camera as closed."""
    from mio.devices.mocks import USBCamMock

    npz_path = _make_npz(tmp_path / "mock.npz", num_frames=1)
    set_usbcam_input(npz_path)

    mock = USBCamMock()
    assert mock.isOpened()
    mock.release()
    assert not mock.isOpened()

def test_csv_structure(set_usbcam_input, tmp_path):
    """Verify CSV has correct columns and monotonically increasing frame indices."""
    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path)
    config.ntp_server = None

    npz_path = _make_npz(tmp_path / "csv_test.npz", num_frames=20)
    set_usbcam_input(npz_path)

    cam = BehaviorCam(recording_config=config, camera_index=0)
    cam.capture(show_video=False)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1

    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Correct columns
    assert reader.fieldnames == ["frame_index", "unix_time"]

    # Monotonically increasing frame index starting at 0
    indices = [int(r["frame_index"]) for r in rows]
    assert indices == list(range(len(rows)))

    # Timestamps should be monotonically non-decreasing
    times = [float(r["unix_time"]) for r in rows]
    assert all(t1 <= t2 for t1, t2 in zip(times, times[1:]))

def test_write_frame_count_matches_video(set_usbcam_input, tmp_path, monkeypatch):
    """Count write_frame calls and verify they match the video frame count."""
    call_count = {"ok": 0, "failed": 0}
    original = VideoWriter.write_frame

    def wrapped(self, frame):
        ok = original(self, frame)
        if ok:
            call_count["ok"] += 1
        else:
            call_count["failed"] += 1
        return ok

    monkeypatch.setattr(VideoWriter, "write_frame", wrapped, raising=True)

    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path)
    config.ntp_server = None

    npz_path = _make_npz(tmp_path / "count_test.npz", num_frames=30)
    set_usbcam_input(npz_path)

    cam = BehaviorCam(recording_config=config, camera_index=0)
    cam.capture(show_video=False)

    video_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
    assert len(video_files) == 1

    cap = cv2.VideoCapture(str(video_files[0]))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert call_count["failed"] == 0, f"write_frame had {call_count['failed']} failures"
    assert call_count["ok"] == video_frame_count, (
        f"write_frame calls ({call_count['ok']}) != video frames ({video_frame_count})"
    )

@pytest.mark.parametrize(
    "codec,expected_ext",
    [("h264", ".mp4"), ("libx264", ".mp4"), ("rawvideo", ".avi")],
)
def test_output_container_matches_codec(set_usbcam_input, tmp_path, codec, expected_ext):
    """Verify the correct container format is chosen for each codec."""
    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path)
    config.ntp_server = None
    config.codec = codec
    if codec == "libx264":
        config.backend = "skvideo"
        config.pix_fmt = "yuv420p"

    npz_path = _make_npz(tmp_path / "codec_test.npz", num_frames=5)
    set_usbcam_input(npz_path)

    cam = BehaviorCam(recording_config=config, camera_index=0)
    cam.capture(show_video=False)

    video_files = list(tmp_path.glob(f"*{expected_ext}"))
    assert len(video_files) == 1, (
        f"Expected 1 {expected_ext} file for codec={codec}, "
        f"found: {list(tmp_path.glob('*.*'))}"
    )

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


def test_frame_count_matches_csv(backend_config, set_usbcam_input, tmp_path):
    """Verify video frame count matches CSV row count for each backend.

    Known bug: skvideo backend may produce fewer video frames than CSV rows.
    """
    npz_path = _make_npz(tmp_path / "stress_input.npz", num_frames=STRESS_NUM_FRAMES)
    set_usbcam_input(npz_path)

    behavior_cam = BehaviorCam(recording_config=backend_config, camera_index=0)
    behavior_cam.capture(show_video=False)

    video_files = list(tmp_path.glob("*.mp4"))
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

@pytest.mark.parametrize("input_fps", [20, 19, 18, 15])
def test_frame_count_realtime(backend_config, set_usbcam_input, tmp_path, input_fps):
    """Verify frame count with realtime replay to simulate camera timing."""
    npz_path = _make_npz(tmp_path / "realtime_input.npz", num_frames=STRESS_NUM_FRAMES, fps=input_fps)
    set_usbcam_input(npz_path, realtime=True)

    behavior_cam = BehaviorCam(recording_config=backend_config, camera_index=0)
    behavior_cam.capture(show_video=False)

    video_files = list(tmp_path.glob("*.mp4"))
    csv_files = list(tmp_path.glob("*.csv"))

    assert len(video_files) == 1, f"Expected 1 video file, found {len(video_files)}"
    assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"

    with open(csv_files[0]) as f:
        csv_row_count = sum(1 for _ in csv.DictReader(f))

    reader = VideoReader(str(video_files[0]))
    video_frame_count = sum(1 for _ in reader.read_frames())
    reader.release()

    assert video_frame_count == csv_row_count, (
        f"Frame count mismatch ({backend_config.backend} backend, "
        f"input {input_fps}fps vs configured {TEST_FPS}fps): "
        f"video has {video_frame_count} frames but CSV has {csv_row_count} rows"
    )
