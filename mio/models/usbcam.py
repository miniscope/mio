"""
Models for USB camera recording configuration.
"""

from typing import Optional

from pydantic import Field

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin


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
    format: str = Field(
        default="MJPEG",
        description="Video format for camera capture (e.g., MJPEG, YUY2). "
        "Note: Output video encoding is handled by VideoWriter.",
    )
    codec: str = Field(
        default="mjpeg",
        description="Video codec for output file (e.g., mjpeg, libx264, rawvideo).",
    )
    pix_fmt: Optional[str] = Field(
        default=None,
        description="Pixel format for video encoding (e.g., yuvj420p, yuv420p, gray). "
        "If None, automatically determined from codec.",
    )
    vendor_id: int = Field(
        default=0x32E4,
        description="USB vendor ID of the camera (for reference/documentation).",
    )
    product_id: Optional[int] = Field(
        default=None,
        description="USB product ID of the camera (for reference/documentation).",
    )
