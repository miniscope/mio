"""
DAQ For use with FPGA streaming video sources.
"""

import json
import multiprocessing
import sys
from pathlib import Path

import cv2
import numpy as np
from noob import Tube
from noob.runner import SynchronousRunner

from mio.devices.base import Device
from mio.devices.stream.ber import prbs15_ber
from mio.devices.stream.config import StreamDevConfig
from mio.devices.stream.headers import StreamBufferHeader
from mio.io import BufferedCSVWriter, VideoWriter
from mio.models.process import FrequencyMaskingConfig
from mio.plots.headers import StreamPlotter
from mio.process.frame_helper import FrequencyMaskHelper
from mio.types import ConfigSource


class StreamDevice(Device):
    """
    A class for configuring and reading frames from an FPGA source.
    Supported devices and required inputs are described in StreamDevConfig model documentation.
    This function's entry point is the main function, which should be used from the
    stream_image_capture command installed with the package.
    Example configuration yaml files are stored in /mio/config/.

    Examples
    --------
    $ mio stream capture -c path/to/config.yml -o output_filename.avi
    Connected to XEM7310-A75
    Succesfully uploaded /mio/mio/interfaces/selected_bitfile.bit
    FrontPanel is supported

    .. todo::

        Make it fast and understandable.

    """

    config_cls = StreamDevConfig
    header_cls = StreamBufferHeader
    device_name = "stream"

    def __init__(self, config: StreamDevConfig | ConfigSource, tube_id: str = "stream") -> None:
        """
        Constructer for the class.
        This parses configuration from the input yaml file.

        Parameters
        ----------
        config : StreamDevConfig | Path
            DAQ configurations imported from the input yaml file.
            Examples and required properties can be found in /mio/config/example.yml

            Passed either as the instantiated config object or a path to on-disk yaml configuration
        """
        super().__init__(config)

        self.tube_id = tube_id

        self.terminate: multiprocessing.Event = multiprocessing.Event()

        self._buffer_npix: list[int] | None = None
        self._nbuffer_per_fm: int | None = None
        self._buffered_writer: BufferedCSVWriter | None = None
        self._header_plotter: StreamPlotter | None = None
        self._buffer_recv_index: int = 0

    def _ber_mode(
        self,
        serial_buffer_queue: multiprocessing.Queue,
        ber_output: Path | None,
    ) -> None:
        """
        BER-mode dispatch from :meth:`.capture`. Runs :meth:`.prbs15_ber`, logs the
        summary, and (optionally) writes the run's results as JSON to ``ber_output``.
        """
        target_buffers = self.config.runtime.ber_test_n_buffers
        result = prbs15_ber(
            serial_buffer_queue, self.config, self.logger, target_buffers, self.header_cls
        )
        aborted = result["aborted"]
        status = f"aborted ({aborted})" if aborted else "complete"
        self.logger.info(
            f"BER test {status}: "
            f"buffers={result['buffers']}/{result['expected_buffers']} "
            f"dropped={result['dropped_buffers']} errored={result['errored_buffers']} "
            f"bits={result['bits']} errors={result['errors']} "
            f"ber={result['ber']:.6g} per={result['per']:.6g}"
        )
        if ber_output:
            summary = {
                "prbs": "PRBS-15 (x^15+x^14+1, MSB-first), seed=(buffer_count & 0x7FFF) or 1",
                "target_buffers": target_buffers,
                "buffers_received": result["buffers"],
                "buffer_count_start": result["buffer_count_start"],
                "buffer_count_end": result["buffer_count_end"],
                "expected_buffers": result["expected_buffers"],
                "dropped_buffers": result["dropped_buffers"],
                "errored_buffers": result["errored_buffers"],
                "bits": result["bits"],
                "errors": result["errors"],
                "ber": result["ber"],
                "per": result["per"],
                "aborted": aborted,
                "windows": result["windows"],
            }
            with open(ber_output, "w") as f:
                json.dump(summary, f, indent=2, default=float)
            self.logger.info(f"BER results written to {ber_output}")

    def capture(
        self,
        video: Path | None = None,
        video_kwargs: dict | None = None,
        metadata: Path | None = None,
        binary: Path | None = None,
        show_video: bool | None = True,
        show_metadata: bool | None = False,
        freq_mask_config: FrequencyMaskingConfig | None = None,
        n_frames: int | None = None,
    ) -> None:
        """
        Entry point to start frame capture.

        Parameters
        ----------
        video: Path, optional
            If present, a path to an output video file
        video_kwargs: dict, optional
            kwargs passed to :meth:`.init_video`
        metadata: Path, optional
            Save metadata information during capture.
        binary: Path, optional
            Save raw binary directly from ``okDev`` to file, if present.
            Note that binary is captured in *append* mode, rather than rewriting an existing file.
        show_video: bool, optional
            If True, display the video in real-time.
        show_metadata: bool, optional
            If True, show metadata information during capture.
        mode: Literal["capture", "ber"], optional
            Capture mode. ``"capture"`` (default) is the main capture routine
            that outputs videos and metadata;
            ``"ber"`` runs a PRBS bit-error-rate test on the incoming data stream.
        ber_output: Path, optional
            When ``mode == "ber"``, JSON file to write the BER summary to.
        n_frames: int, optional
            If set, only capture n_frames from the source, then quit
        """
        tube = Tube.from_specification(
            self.tube_id,
            input={
                "config": self.config,
                "capture_binary": binary,
                "header_csv": metadata,
                "show_video": show_video,
                "show_metadata": show_metadata,
                "freq_mask_config": (
                    FrequencyMaskingConfig.from_any(freq_mask_config) if freq_mask_config else None
                ),
                "video_path": video,
            },
        )
        runner = SynchronousRunner(tube)
        with runner:
            try:
                runner.run(n_frames)
            except KeyboardInterrupt:
                # TODO: soft-stop, drain remaining frame queues
                self.logger.exception("Quitting capture")

    def _handle_frame(
        self,
        image: np.ndarray,
        header_list: list[StreamBufferHeader],
        show_video: bool,
        writer: VideoWriter | None,
        show_metadata: bool,
        metadata: Path | None = None,
        freq_mask_helper: FrequencyMaskHelper | None = None,
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
                        meta_row = header.model_dump()
                        self._buffered_writer.process(meta_row)
                    except Exception as e:
                        self.logger.exception(f"Exception saving headers: \n{e}")
        if image is None or image.size == 0:
            self.logger.warning("Empty frame received, skipping.")
            return
        if show_video:
            try:
                display_image = freq_mask_helper.process(image) if freq_mask_helper else image

                cv2.imshow("image", display_image)
                cv2.waitKey(1)
            except cv2.error as e:
                self.logger.exception(f"Error displaying frame: {e}")
        if writer:
            try:
                writer.write_frame(image)
            except cv2.error as e:
                self.logger.exception(f"Exception writing frame: {e}")


# DEPRECATION: v0.3.0
if __name__ == "__main__":
    import warnings

    warnings.warn(
        "Calling the device.py module directly is deprecated - use the `mio` cli. "
        "try:\n\n  mio stream capture --help",
        stacklevel=1,
    )
    sys.exit(1)
