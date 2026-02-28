import csv
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mio.behavior_cam import BehaviorCam
from mio.io import VideoReader, VideoWriter
from mio.models.usbcam import USBCameraRecordingConfig

NUM_TEST_FRAMES = 10
TEST_WIDTH = 1280
TEST_HEIGHT = 720
TEST_FPS = 20


def _make_npz(path: Path, num_frames: int = NUM_TEST_FRAMES, fps: float = TEST_FPS) -> Path:
    """Generate a synthetic .npz matching elp-camera config."""
    frames = np.random.default_rng().integers(
        0, 255, size=(num_frames, TEST_HEIGHT, TEST_WIDTH, 3), dtype=np.uint8
    )
    timestamps = np.arange(num_frames, dtype=np.float64) / fps
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


@pytest.mark.timeout(60)
def test_capture_interrupt_produces_valid_output(set_usbcam_input, tmp_path):
    """Verify output is valid when capture is interrupted mid-recording."""
    # 600 frames at 20fps = 30s of realtime playback.
    # Timer fires at 15s, allowing for slow multiprocess startup on Windows CI (~5s).
    num_frames = 600
    npz_path = _make_npz(tmp_path / "interrupt_input.npz", num_frames=num_frames)
    set_usbcam_input(npz_path, realtime=True)

    config = USBCameraRecordingConfig.from_id("elp-camera")
    config.output_dir = str(tmp_path / "output")
    config.ntp_server = None

    cam = BehaviorCam(recording_config=config, camera_index=0)

    timer = threading.Timer(15.0, cam.terminate.set)
    timer.start()
    cam.capture(show_video=False)

    video_files = list(Path(config.output_dir).glob("*.mp4")) + list(
        Path(config.output_dir).glob("*.avi")
    )
    assert len(video_files) == 1

    # Confirm the interrupt actually fired
    assert cam.terminate.is_set(), "terminate event was never set — timer did not fire"

    # Verify the partial recording is readable (moov atom present)
    cap = cv2.VideoCapture(str(video_files[0]))
    assert cap.isOpened(), "Interrupted recording not readable — moov atom likely missing"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 0, "No frames in interrupted recording"
    assert frame_count < num_frames, (
        f"Expected partial recording ({num_frames} input frames at 20fps realtime) "
        f"but got all {frame_count} frames — interrupt may not have fired in time"
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("config_id", ["elp-camera", "elp-camera-skvideo"])
def test_sigint_produces_valid_output(tmp_path, config_id):
    """Test that a real signal (SIGINT / CTRL_BREAK) produces a valid video.

    Runs capture in a subprocess and sends it a signal to simulate Ctrl+C.
    """
    npz_path = tmp_path / "input.npz"
    _make_npz(npz_path, num_frames=600)
    output_dir = tmp_path / "output"

    script = textwrap.dedent(f"""\
        import os
        os.environ["BEHAVIORCAM_MOCKRUN"] = "1"
        os.environ["PYTEST_USBCAM_DATA_FILE"] = {str(npz_path)!r}
        os.environ["PYTEST_USBCAM_REALTIME"] = "1"
        from mio.behavior_cam import BehaviorCam
        from mio.models.usbcam import USBCameraRecordingConfig
        config = USBCameraRecordingConfig.from_id("{config_id}")
        config.output_dir = {str(output_dir)!r}
        config.ntp_server = None
        cam = BehaviorCam(recording_config=config, camera_index=0)
        cam.capture(show_video=False)
    """)

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)

    # Wait for capture to start producing frames
    time.sleep(15)

    # Send signal — like real Ctrl+C
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)

    proc.wait(timeout=30)

    videos = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.avi"))
    assert len(videos) == 1

    cap = cv2.VideoCapture(str(videos[0]))
    assert cap.isOpened(), "Video not readable after signal — moov atom likely missing"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 0, "No frames in recording after signal"
    assert frame_count < 600, (
        f"Expected partial recording but got {frame_count} frames — signal may not have fired in time"
    )


STRESS_NUM_FRAMES = 100


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


@pytest.mark.parametrize("input_fps", [20, 19, 18, 15])
def test_frame_count_realtime(backend_config, set_usbcam_input, tmp_path, input_fps):
    """Verify frame count with realtime replay to simulate real camera timing.

    Real cameras often deliver slightly fewer FPS than configured.
    Known bug: skvideo backend may produce fewer video frames than CSV rows.
    """
    npz_path = _make_npz(tmp_path / "realtime_input.npz", num_frames=STRESS_NUM_FRAMES, fps=input_fps)
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
        f"Frame count mismatch ({backend_config.backend} backend, "
        f"input {input_fps}fps vs configured {TEST_FPS}fps): "
        f"video has {video_frame_count} frames but CSV has {csv_row_count} rows"
    )
