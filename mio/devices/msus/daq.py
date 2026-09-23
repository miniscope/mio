# ruff: noqa: D100

import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from mio import init_logger
from mio.devices.msus.config import MSUSDevConfig
from mio.devices.msus.header import MSUSBufferHeader, MSUSBufferHeaderFormat
from mio.devices.stream import StreamDevice
from mio.io import BufferedCSVWriter, VideoWriter
from mio.plots.headers import StreamPlotter
from mio.types import ConfigSource


# testing here:
def format_frame(frame_data: list[np.ndarray], config: MSUSDevConfig) -> np.ndarray:
    """
    Convert a list of 1D pixel arrays into a full frame, stripping the leading "training" pixels
    """
    pixels = np.concatenate(frame_data)
    # frame = pixels.reshape((config.frame_height, config.frame_width_input)) # active full
    # frame = pixels.reshape(320, 328) #active full
    # frame = pixels.reshape(120, 120) # active when ROI
    frame = pixels.reshape(320, 328)  # active full

    # strip training pixels
    frame = frame[:, 8:]  # active full

    return frame


class MSUSStreamDevice(StreamDevice):
    """Mystery scope daq"""

    buffer_header_cls: ClassVar = MSUSBufferHeader

    def __init__(
        self,
        device_config: MSUSDevConfig | ConfigSource,
        header_fmt: MSUSBufferHeaderFormat | ConfigSource = "msus-buffer-header",
    ) -> None:
        """
        Constructer for the class.
        This parses configuration from the input yaml file.

        Parameters
        ----------
        config : StreamDevConfig | Path
            DAQ configurations imported from the input yaml file.
            Examples and required properties can be found in /mio/config/example.yml

            Passed either as the instantiated config object or a path to on-disk yaml configuration
        header_fmt : MetadataHeaderFormat, optional
            Header format used to parse information from buffer header,
            by default `MetadataHeaderFormat()`.
        """

        super().__init__(device_config, header_fmt)  # initiating parameters of the parent class
        self.logger = init_logger("GSStreamDaq")
        self.config = MSUSDevConfig.from_any(device_config)
        self.header_fmt = MSUSBufferHeaderFormat.from_any(header_fmt)

        self.preamble = self.config.preamble

        self._nbuffer_per_fm: int | None = None
        self._buffered_writer: BufferedCSVWriter | None = None
        self._header_plotter: StreamPlotter | None = None

    @property
    def buffer_npix(self) -> list[int]:
        """List of pixels per buffer for a frame includes unprocessed data"""
        if self._buffer_npix is None:
            self._buffer_npix = self.config.buffer_npix

        return self._buffer_npix

    def _format_frame_inner(self, frame_data: list[np.ndarray]) -> np.ndarray:
        return format_frame(frame_data, self.config)

    def _handle_frame(
        self,
        image: np.ndarray,
        header_list: list[MSUSBufferHeaderFormat],
        show_video: bool,
        writer: VideoWriter | None,
        show_metadata: bool,
        metadata: Path | None = None,
    ) -> None:
        """
        Inner handler for :meth:`.capture` to process the frames from the frame queue.

        .. todo::

            Further refactor to break into smaller pieces, not have to pass 100 args every time.

        """
        if show_metadata or metadata:
            for header in header_list:
                if show_metadata:
                    self.logger.debug("Plotting header metadata")
                    try:
                        self._header_plotter.update(header)
                    except Exception as e:
                        self.logger.exception(f"Exception plotting headers: \n{e}")
                if metadata:
                    self.logger.debug("Saving header metadata")
                    try:
                        self._buffered_writer.append(
                            list(header.model_dump(warnings=False).values()) + [time.time()]
                        )
                    except Exception as e:
                        self.logger.exception(f"Exception saving headers: \n{e}")
        if image is None or image.size == 0:
            self.logger.warning("Empty frame received, skipping.")
            return
        if show_video:
            try:
                cv2.imshow("image", image)
                cv2.waitKey(1)
            except cv2.error as e:
                self.logger.exception(f"Error displaying frame: {e}")
        if writer:
            try:
                picture = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # If your image is grayscale
                writer.write_frame(picture)
            except cv2.error as e:
                self.logger.exception(f"Exception writing frame: {e}")
