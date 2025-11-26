"""
USB Camera device helper functions.
"""

import time
from typing import Dict, Optional

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

# Constants
MAX_CAMERA_INDEX = 5
CAMERA_INIT_DELAY_SECONDS = 0.1  # Delay after setting camera properties before reading
CAMERA_INIT_RETRY_ATTEMPTS = 3  # Number of retry attempts when reading initial frame


def determine_pix_fmt(codec: str, pix_fmt: Optional[str] = None) -> str:
    """
    Determine pixel format for video output based on codec.

    Args:
        codec: Video codec (e.g., "mjpeg", "rawvideo", "libx264")
        pix_fmt: Explicit pixel format override (if provided, returns this)

    Returns:
        Pixel format string for FFmpeg
    """
    if pix_fmt is not None:
        return pix_fmt
    elif codec.lower() == "mjpeg":
        return "yuvj420p"  # YUV color format for MJPEG
    elif codec.lower() == "rawvideo":
        return "gray"  # Grayscale for rawvideo
    else:
        return "yuv420p"  # Default YUV format for other codecs


def convert_frame_for_codec(frame: np.ndarray, codec: str) -> np.ndarray:
    """
    Convert frame color space based on codec requirements.

    Args:
        frame: Input frame (BGR from OpenCV)
        codec: Video codec (e.g., "mjpeg", "rawvideo", "libx264")

    Returns:
        Converted frame ready for video writer
    """
    if codec.lower() == "rawvideo":
        # Rawvideo expects grayscale
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return frame
    else:
        # Other codecs expect RGB
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


def open_camera(
    camera_index: int,
    frame_width: int,
    frame_height: int,
    fps: int,
) -> cv2.VideoCapture:
    """
    Open and configure a camera with the specified settings.

    Args:
        camera_index: Index of the camera to open
        frame_width: Desired frame width
        frame_height: Desired frame height
        fps: Desired frames per second

    Returns:
        Configured VideoCapture object

    Raises:
        RuntimeError: If camera cannot be opened or cannot read frames
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Give camera time to initialize after setting properties
    time.sleep(CAMERA_INIT_DELAY_SECONDS)

    # Verify camera is working by reading a test frame
    # Retry a few times as some cameras need a moment to start
    ret = False
    frame = None
    for _ in range(CAMERA_INIT_RETRY_ATTEMPTS):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(CAMERA_INIT_DELAY_SECONDS)

    if not ret:
        cap.release()
        raise RuntimeError(
            f"Camera at index {camera_index} opened but could not read initial frame "
            f"after {CAMERA_INIT_RETRY_ATTEMPTS} attempts. "
            "The camera may be in use by another application."
        )

    return cap


def format_camera_info(idx: int, info: Dict[str, str], prefix: str = "[") -> str:
    """
    Format camera information for display.

    Args:
        idx: Camera index
        info: Camera info dictionary
        prefix: Prefix for index (default: "[" for "[0]",
            use "Index " for "Index 0:" format)

    Returns:
        Formatted string for display
    """
    name = info.get("name", "Camera")
    resolution = info.get("resolution", "Unknown")
    fps = info.get("fps", "Unknown")
    index_str = f"[{idx}]" if prefix == "[" else f"{prefix}{idx}:"
    return f"{index_str} {name} - {resolution} @ {fps} fps"


def list_cameras() -> Dict[int, Dict[str, str]]:
    """
    List available cameras with name, resolution, and fps.

    .. note::
        **Windows Limitation**: Camera enumeration on Windows has known issues where
        checking one camera can interfere with detecting others. This function may
        only find one camera per run on Windows systems. If you have multiple cameras,
        you may need to check them individually or run the enumeration multiple times.

    Returns:
        Dictionary mapping camera index (0, 1, 2...) to camera info.
    """
    available_cameras: Dict[int, Dict[str, str]] = {}
    found_cameras: Dict[tuple[str, str], int] = {}  # (resolution, fps) -> index

    # Get camera names from cv2-enumerate-cameras
    enumerated_cameras: Dict[int, str] = {}
    try:
        for camera in enumerate_cameras():
            enumerated_cameras[camera.index] = camera.name
    except Exception:
        pass

    # Check standard indices (0-9) - prefer these over high backend indices
    for i in range(MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                resolution = f"{frame.shape[1]}x{frame.shape[0]}"
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                camera_key = (resolution, str(fps))

                # Try to find matching name from enumerated cameras
                name = f"Camera {i}"
                for enum_idx, enum_name in enumerated_cameras.items():
                    # Check if this standard index matches an enumerated camera
                    test_cap = cv2.VideoCapture(enum_idx)
                    if test_cap.isOpened():
                        test_ret, test_frame = test_cap.read()
                        test_cap.release()
                        if test_ret:
                            test_res = f"{test_frame.shape[1]}x{test_frame.shape[0]}"
                            test_fps = int(cv2.VideoCapture(enum_idx).get(cv2.CAP_PROP_FPS))
                            if test_res == resolution and str(test_fps) == str(fps):
                                name = enum_name
                                break

                # Check if we've already found this camera (duplicate)
                if camera_key not in found_cameras:
                    found_cameras[camera_key] = i
                    available_cameras[i] = {
                        "name": name,
                        "resolution": resolution,
                        "fps": str(fps),
                    }
            cap.release()
        else:
            # Stop checking after first failure (no more cameras)
            break

    return available_cameras
