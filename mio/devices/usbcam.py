"""
USB Camera device helper functions.
"""

import time
from typing import Dict, Literal, TypedDict

import cv2
import numpy as np

from mio.logging import init_logger

logger = init_logger("usbcam")

# Constants
MAX_CAMERA_INDEX = 5
CAMERA_INIT_DELAY_SECONDS = 0.1  # Delay after setting camera properties before reading
CAMERA_INIT_RETRY_ATTEMPTS = 3  # Number of retry attempts when reading initial frame


Codec = Literal["mjpeg", "libx264", "h264", "rawvideo"]


class CameraInfo(TypedDict):
    """Camera information from OpenCV discovery."""

    name: str
    resolution: str
    fps: int


def convert_frame_for_codec(frame: np.ndarray, codec: Codec) -> np.ndarray:
    """
    Convert frame color space based on codec requirements.

    Args:
        frame: Input frame (BGR from OpenCV)
        codec: Video codec for output encoding

    Returns:
        Converted frame ready for video writer
    """
    if codec == "rawvideo":
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
            raise ValueError(
                f"Expected BGR (3-channel) frame for codec '{codec}', "
                f"got shape {frame.shape}. Use 'rawvideo' for grayscale."
            )


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
    for _ in range(CAMERA_INIT_RETRY_ATTEMPTS):
        ret, _ = cap.read()
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


def format_camera_info(idx: int, info: CameraInfo) -> str:
    """
    Format camera information for display.

    Args:
        idx: Camera index
        info: Camera info from discovery

    Returns:
        Formatted string for display
    """
    return f"[{idx}] {info['name']} - {info['resolution']} @ {info['fps']} fps"


def list_cameras() -> Dict[int, CameraInfo]:
    """
    List available cameras with name, resolution, and fps.

    Returns:
        Dictionary mapping camera index (0, 1, 2...) to camera info.
    """
    available_cameras: Dict[int, CameraInfo] = {}

    logger.info(f"Scanning for cameras (indices 0-{MAX_CAMERA_INDEX - 1})...")
    for i in range(MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                resolution = f"{frame.shape[1]}x{frame.shape[0]}"
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_cameras[i] = {
                    "name": f"Camera {i}",
                    "resolution": resolution,
                    "fps": fps,
                }
            else:
                logger.debug(f"Camera at index {i} opened but failed to read frame")
            cap.release()
        else:
            logger.debug(f"No camera found at index {i}")

    return available_cameras
