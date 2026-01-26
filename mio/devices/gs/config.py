# ruff: noqa: D100


import numpy as np
from pydantic import ConfigDict

from mio.models.stream import StreamDevConfig


class GSDevConfig(StreamDevConfig):
    """Device config for an unknown, mystery microscope"""

    pix_depth: int = 12 # 12 originally
    max_pixels_per_buffer: int = 10000

    model_config = ConfigDict(validate_default=True)

    @property
    def frame_width_input(self) -> int:
        """8 (12 bit) alignment columns removed from 320 rows of imaging data"""
        # return self.frame_width + 8
        return self.frame_width + 8

    @property
    def pix_depth_input(self) -> int:
        """12 bit raw processed to 10 bit pixel values"""
        # return self.pix_depth + 2
        return self.pix_depth + 2

    @property
    def buffer_npix(self) -> list[int]:
        """
        Number of pixels (not bytes!) present in each buffer within a frame.

        i.e. the list will be the length of the number of buffers per frame,
        and each item in the list is the number of pixels in that buffer
        """
        total_pixels = self.frame_width_input * self.frame_height
        buffer_npix = [self.max_pixels_per_buffer] * int(
            np.ceil(total_pixels / self.max_pixels_per_buffer)
        )
        remainder = total_pixels % self.max_pixels_per_buffer
        if remainder != 0:
            buffer_npix[-1] = remainder
        return buffer_npix
