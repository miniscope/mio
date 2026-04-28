"""
DAQ For use with FPGA streaming video sources.
"""

import json
import multiprocessing
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from mio.devices.base import Device
from mio.devices.stream.ber import prbs15_ber
from mio.devices.stream.config import StreamDevConfig
from mio.devices.stream.headers import StreamBufferHeader
from mio.devices.stream.nodes import buffer_to_frame, exact_iter, format_frame, fpga_recv
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

    def __init__(
        self,
        config: StreamDevConfig | ConfigSource,
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
        """
        super().__init__(config)

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
        self.logger.info(
            "BER test complete: buffers=%d bits=%d errors=%d ber=%.6g",
            result["buffers"],
            result["bits"],
            result["errors"],
            result["ber"],
        )
        if ber_output:
            summary = {
                "prbs": "PRBS-15 (x^15+x^14+1, MSB-first), " "seed=(buffer_count & 0x7FFF) or 1",
                "target_buffers": target_buffers,
                "buffers_received": result["buffers"],
                "buffer_count_start": result["buffer_count_start"],
                "buffer_count_end": result["buffer_count_end"],
                "bits": result["bits"],
                "errors": result["errors"],
                "ber": result["ber"],
                "windows": result["windows"],
            }
            with open(ber_output, "w") as f:
                json.dump(summary, f, indent=2, default=float)
            self.logger.info("BER results written to %s", ber_output)

    def capture(
        self,
        read_length: int | None = None,
        video: Path | None = None,
        video_kwargs: dict | None = None,
        metadata: Path | None = None,
        binary: Path | None = None,
        show_video: bool | None = True,
        show_metadata: bool | None = False,
        freq_mask_config: FrequencyMaskingConfig | None = None,
        mode: Literal["capture", "ber"] = "capture",
        ber_output: Path | None = None,
    ) -> None:
        """
        Entry point to start frame capture.

        Parameters
        ----------
        read_length : Optional[int], optional
            Passed to :func:`~mio.stream_daq.stream_daq.fpga_recv` when
            `source == "fpga"`, by default None.
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
        """
        self.terminate.clear()
        if mode not in ("capture", "ber"):
            raise ValueError(f"Mode must be either 'capture' or 'ber', got {mode}")

        shared_resource_manager = multiprocessing.Manager()
        serial_buffer_queue = shared_resource_manager.Queue(
            self.config.runtime.serial_buffer_queue_size
        )
        frame_buffer_queue = shared_resource_manager.Queue(
            self.config.runtime.frame_buffer_queue_size
        )
        imagearray = shared_resource_manager.Queue(self.config.runtime.image_buffer_queue_size)

        spawn_mode = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        ctx = multiprocessing.get_context(spawn_mode)

        procs = []
        self.logger.debug("Starting fpga capture process")
        p_recv = ctx.Process(
            target=fpga_recv,
            args=(serial_buffer_queue, self.config, read_length, True, binary),
            name="fpga_recv",
        )

        procs.append(p_recv)

        if freq_mask_config:
            freq_mask_helper = FrequencyMaskHelper(
                height=self.config.frame_height,
                width=self.config.frame_width,
                freq_mask_config=freq_mask_config,
            )
        else:
            freq_mask_helper = None

        writer = None
        if video:
            writer = VideoWriter(
                path=video,
                fps=self.config.fs,
                output_dict=video_kwargs,
            )

        if mode == "capture":
            p_buffer_to_frame = ctx.Process(
                target=buffer_to_frame,
                args=(serial_buffer_queue, frame_buffer_queue, self.config, self.header_cls),
                name="buffer_to_frame",
            )
            p_format_frame = ctx.Process(
                target=format_frame,
                args=(
                    frame_buffer_queue,
                    imagearray,
                    self.config,
                ),
                name="format_frame",
            )
            procs.append(p_buffer_to_frame)
            procs.append(p_format_frame)

        for p in procs:
            p.start()

        if show_metadata:
            self._header_plotter = StreamPlotter(
                header_keys=self.config.runtime.plot.keys,
                history_length=self.config.runtime.plot.history,
                update_ms=self.config.runtime.plot.update_ms,
            )

        if metadata:
            header_cols = StreamBufferHeader.csv_header_cols()
            self._buffered_writer = BufferedCSVWriter(
                metadata, header=header_cols, buffer_size=self.config.runtime.csvwriter.buffer
            )

        try:
            if mode == "ber":
                self._ber_mode(serial_buffer_queue, ber_output)
                return
            for image, header_list in exact_iter(imagearray.get, None):
                self._handle_frame(
                    image,
                    header_list,
                    show_video=show_video,
                    writer=writer,
                    show_metadata=show_metadata,
                    metadata=metadata,
                    freq_mask_helper=freq_mask_helper,
                )
        except KeyboardInterrupt:
            self.logger.exception(
                "Quitting capture, processing remaining frames. Ctrl+C again to force quit"
            )
            self.terminate.set()
            try:
                for image, header_list in exact_iter(lambda: imagearray.get(1), None):
                    self._handle_frame(
                        image,
                        header_list,
                        show_video=show_video,
                        writer=writer,
                        show_metadata=show_metadata,
                        metadata=metadata,
                    )
            except KeyboardInterrupt:
                self.logger.exception("Force quitting")
        except Exception as e:
            self.logger.exception(f"Error during capture: {e}")
            self.terminate.set()
        finally:
            if writer:
                writer.close()
                self.logger.debug("VideoWriter released")
            if show_video:
                cv2.destroyAllWindows()
                cv2.waitKey(100)
            if show_metadata:
                self._header_plotter.close_plot()
            if metadata:
                self._buffered_writer.close()

            # Join child processes with a timeout
            # Should never happen except during a force quit, as we wait for all
            # queues to drain, and if they don't do so on their own, it's a bug.
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    self.logger.warning(f"Termination timeout: force terminating process {p.name}.")
                    p.terminate()
                    p.join()
            self.logger.info("Child processes joined. End capture.")

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
                        meta_row = header.model_dump_all()
                        self._buffered_writer.append(meta_row)
                    except Exception as e:
                        self.logger.exception(f"Exception saving headers: \n{e}")
        if image is None or image.size == 0:
            self.logger.warning("Empty frame received, skipping.")
            return
        if show_video:
            try:
                display_image = freq_mask_helper.process_frame(image) if freq_mask_helper else image

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
