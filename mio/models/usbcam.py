"""
Models for USB camera recording configuration.
"""

from typing import Literal, Optional

from pydantic import Field

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin

Codec = Literal["mjpeg", "libx264", "h264", "rawvideo"]


class USBCameraRecordingConfig(MiniscopeConfig, ConfigYAMLMixin):
    """
    Configuration for recording video from USB camera.
    """

    output_dir: str = Field(
        default="recordings", description="Directory to save the recorded video."
    )
    frame_width: int = Field(default=1920, description="Width of the recorded video.")
    frame_height: int = Field(default=1080, description="Height of the recorded video.")
    fps: int = Field(default=20, description="Frames per second of the recorded video.")
    format: Literal["MJPEG", "YUY2"] = Field(
        default="MJPEG",
        description="Video format for camera capture. "
        "Note: Output video encoding is handled by VideoWriter.",
    )
    codec: Codec = Field(
        default="libx264",
        description=(
            "Video codec for output file. "
            "Used by skvideo backend, mapped to fourcc for cv2 backend."
        ),
    )
    pix_fmt: Optional[str] = Field(
        default="yuv420p",
        description=(
            "Pixel format for video encoding (e.g., yuvj420p, yuv420p, gray). "
            "Only used by skvideo backend, ignored by cv2 backend."
        ),
    )
    backend: Literal["skvideo", "cv2"] = Field(
        default="skvideo",
        description=(
            "Video writer backend: 'skvideo' uses FFmpegWriter, " "'cv2' uses cv2.VideoWriter."
        ),
    )
    ntp_server: Optional[str] = Field(
        default=None,
        description="NTP server address for time synchronization check. "
        "If specified, the system time will be verified against this server before capture.",
    )
    ntp_max_offset_seconds: float = Field(
        default=0.01,
        description="Maximum allowed time offset in seconds for NTP synchronization check "
        "(default: 0.01 = 10ms).",
    )
