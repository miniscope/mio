"""
DAQ For use with FPGA and Uart streaming video sources.
"""

import logging
import multiprocessing
import os
import queue
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Generator, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from bitstring import BitArray, Bits

from mio import init_logger
from mio.bit_operation import BufferFormatter
from mio.devices.mocks import okDevMock
from mio.exceptions import EndOfRecordingException, StreamReadError
from mio.io import BufferedCSVWriter, VideoWriter
from mio.models.process import FreqencyMaskingConfig
from mio.models.stream import (
    RuntimeMetadata,
    StreamBufferHeader,
    StreamBufferHeaderFormat,
    StreamDevConfig,
)
from mio.plots.headers import StreamPlotter
from mio.process.frame_helper import FrequencyMaskHelper
from mio.types import ConfigSource

HAVE_OK = False
ok_error = None
BIT_PER_WORD = 32

try:
    from mio.devices.opalkelly import okDev

    HAVE_OK = True
except (ImportError, ModuleNotFoundError):
    module_logger = init_logger("streamDaq")
    module_logger.warning(
        "Could not import OpalKelly driver, you can't read from FPGA!\n"
        "Check out Opal Kelly's website for troubleshooting\n"
        "https://docs.opalkelly.com/fpsdk/getting-started/"
    )


def exact_iter(f: Callable, sentinel: Any) -> Generator[Any, None, None]:
    """
    A version of :func:`iter` that compares with `is` rather than `==`
    because truth value of numpy arrays is ambiguous.
    """
    while True:
        val = f()
        if val is sentinel:
            break
        else:
            yield val


class StreamDaq:
    """
    A combined class for configuring and reading frames from a UART and FPGA source.
    Supported devices and required inputs are described in StreamDevConfig model documentation.
    This function's entry point is the main function, which should be used from the
    stream_image_capture command installed with the package.
    Example configuration yaml files are stored in /mio/config/.

    Examples
    --------
    $ mio stream capture -c path/to/config.yml -o output_filename.avi
    Connected to XEM7310-A75
    Succesfully uploaded /mio/mio/devices/selected_bitfile.bit
    FrontPanel is supported

    .. todo::

        Make it fast and understandable.

    """

    def __init__(
        self,
        device_config: Union[StreamDevConfig, ConfigSource],
        header_fmt: Union[StreamBufferHeaderFormat, ConfigSource] = "stream-buffer-header",
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

        self.logger = init_logger("streamDaq")
        self.config = StreamDevConfig.from_any(device_config)
        self.header_fmt = StreamBufferHeaderFormat.from_any(header_fmt)
        if isinstance(header_fmt, str):
            self.header_fmt = StreamBufferHeaderFormat.from_id(header_fmt)
        elif isinstance(header_fmt, StreamBufferHeaderFormat):
            self.header_fmt = header_fmt
        else:
            raise TypeError(
                "header_fmt should be an instance of StreamBufferHeaderFormat or a config ID."
            )
        self.preamble = self.config.preamble
        self.terminate: multiprocessing.Event = multiprocessing.Event()

        self._buffer_npix: Optional[List[int]] = None
        self._nbuffer_per_fm: Optional[int] = None
        self._buffered_writer: Optional[BufferedCSVWriter] = None
        self._header_plotter: Optional[StreamPlotter] = None
        self._buffer_recv_index: int = 0

    @property
    def buffer_npix(self) -> List[int]:
        """List of pixels per buffer for a frame"""
        if self._buffer_npix is None:
            px_per_frame = self.config.frame_width * self.config.frame_height
            byte_per_word = np.iinfo(np.int32).bits / np.iinfo(np.int8).bits

            px_per_buffer = (
                self.config.buffer_block_length * self.config.block_size
                - self.config.header_len / np.iinfo(np.int8).bits
                - self.config.dummy_words * byte_per_word
            )
            quotient, remainder = divmod(px_per_frame, px_per_buffer)
            self._buffer_npix = [int(px_per_buffer)] * int(quotient) + [int(remainder)]
        return self._buffer_npix

    @property
    def nbuffer_per_fm(self) -> int:
        """
        Number of buffers per frame, computed from :attr:`.buffer_npix`
        """
        if self._nbuffer_per_fm is None:
            self._nbuffer_per_fm = len(self.buffer_npix)
        return self._nbuffer_per_fm

    def _trim(
        self,
        data: np.ndarray,
        expected_size_array: List[int],
        header: StreamBufferHeader,
        logger: logging.Logger,
    ) -> np.ndarray:
        """
        Trim or pad an array to match an expected size

        .. todo::
            Re-think about the timing to deal with dummy words.
            It feels cleaner to remove these dummy words right after the preamble detections.
            That way, all data we inject into later stages will be pure metadata and pixel data.
            This isn't critical and I don't want to slow down detection so skipping for now.
        """
        expected_payload_size = expected_size_array[0]
        expected_data_size = expected_size_array[header.frame_buffer_count]

        # This validation is temporary. More info in todo above.
        if data.shape[0] != expected_payload_size + self.config.dummy_words * 4:
            logger.warning(
                f"Frame {header.frame_num}; Buffer {header.buffer_count} "
                f"(#{header.frame_buffer_count} in frame)\n"
                f"Expected buffer data length: {expected_payload_size}, got data with shape "
                f"{data.shape}.\nPadding to expected length",
            )

        if data.shape[0] != expected_data_size:
            # trim if too long
            if data.shape[0] > expected_data_size:
                data = data[0:expected_data_size]
                header.runtime_metadata.black_padding_px = 0  # No padding, data was trimmed
            # pad if too short
            else:
                padding_amount = expected_data_size - data.shape[0]
                data = np.pad(data, (0, padding_amount))
                header.runtime_metadata.black_padding_px = padding_amount
        else:
            # No trimming or padding needed
            header.runtime_metadata.black_padding_px = 0

        return data

    def _init_okdev(self, BIT_FILE: Path, read_length: int) -> Union[okDev, okDevMock]:
        # FIXME: when multiprocessing bug resolved, remove this and just mock in tests
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("STREAMDAQ_MOCKRUN"):
            dev = okDevMock(read_length=read_length)
        else:
            dev = okDev(read_length=read_length)

        dev.upload_bit(str(BIT_FILE))
        dev.set_wire(0x00, 0b0010)
        time.sleep(0.01)
        dev.set_wire(0x00, 0b0)
        dev.set_wire(0x00, 0b1000)
        time.sleep(0.01)
        dev.set_wire(0x00, 0b0)
        return dev

    def _fpga_recv(
        self,
        serial_buffer_queue: multiprocessing.Queue,
        read_length: int = None,
        pre_first: bool = True,
        capture_binary: Optional[Path] = None,
    ) -> None:
        """
        Function to read bitstream from OpalKelly device and store buffer in `serial_buffer_queue`.

        The bits data are read in fixed chunks defined by `read_length`.
        Then we concatenate the chunks and try to look for `self.preamble` in the data.
        The data between every pair of `self.preamble` is considered to be a single buffer and
        stored in `serial_buffer_queue`.

        Parameters
        ----------
        serial_buffer_queue : multiprocessing.Queue[bytes]
            The queue holding the buffer data.
        read_length : int, optional
            Length of data to read in chunks (in number of bytes), by default None.
            If `None`, an optimal length is estimated so that it roughly covers a single buffer
            and is an integer multiple of 16 bytes (as recommended by OpalKelly).
        pre_first : bool, optional
            Whether preamble/header is returned at the beginning of each buffer, by default True.
        capture_binary: Path, optional
            save binary directly from the ``okDev`` to the supplied path, if present.

        Raises
        ------
        RuntimeError
            If the OpalKelly device library cannot be found
        """
        locallogs = init_logger("streamDaq.fpga_recv")
        if not HAVE_OK:
            serial_buffer_queue.put(None)
            raise RuntimeError(
                "Couldnt import OpalKelly device. Check the docs for install instructions!"
            )
        # determine length
        if read_length is None:
            read_length = int(max(self.buffer_npix) * self.config.pix_depth / 8 / 16) * 16

        # set up fpga devices
        BIT_FILE = self.config.bitstream
        if not BIT_FILE.exists():
            serial_buffer_queue.put(None)
            raise RuntimeError(f"Configured to use bitfile at {BIT_FILE} but no such file exists")

        # set up fpga devices
        dev = self._init_okdev(BIT_FILE, read_length)

        # read loop
        pre = Bits(self.preamble)
        if self.config.reverse_header_bits:
            pre = pre[::-1]

        locallogs.debug("Starting capture")
        try:
            for buf in iter_buffers(
                dev, preamble=pre, pre_first=pre_first, capture_binary=capture_binary
            ):
                try:
                    serial_buffer_queue.put(
                        buf,
                        block=True,
                        timeout=self.config.runtime.queue_put_timeout,
                    )
                except queue.Full:
                    locallogs.warning("Serial buffer queue full, skipping buffer.")

        finally:
            locallogs.debug("Quitting, putting sentinel in queue")
            try:
                serial_buffer_queue.put(
                    None, block=True, timeout=self.config.runtime.queue_put_timeout
                )
            except queue.Full:
                locallogs.error("Serial buffer queue full, Could not put sentinel.")

    def _parse_header(self, buffer: bytes) -> Tuple[StreamBufferHeader, np.ndarray]:
        """
        Function to parse header from each buffer.

        Parameters
        ----------
        buffer : bytes
            Input buffer.

        Returns
        -------
        Tuple[BufferHeader, ndarray]
            The returned header data and payload (uint8).
        """

        header, payload = BufferFormatter.bytebuffer_to_ndarrays(
            buffer=buffer,
            header_length_words=int(self.config.header_len / 32),
            preamble_length_words=int(len(Bits(self.config.preamble)) / 32),
            reverse_header_bits=self.config.reverse_header_bits,
            reverse_header_bytes=self.config.reverse_header_bytes,
            reverse_payload_bits=self.config.reverse_payload_bits,
            reverse_payload_bytes=self.config.reverse_payload_bytes,
        )

        runtime_metadata = RuntimeMetadata(
            buffer_recv_index=-1,  # will be set later in _buffer_to_frame for processed buffers
            buffer_recv_unix_time=time.time(),
        )
        header_data = StreamBufferHeader.from_format(
            header.astype(int), self.header_fmt, runtime_metadata=runtime_metadata
        )
        header_data.adc_scaling = self.config.adc_scale

        return header_data, payload

    def _buffer_to_frame(
        self,
        serial_buffer_queue: multiprocessing.Queue,
        frame_buffer_queue: multiprocessing.Queue,
    ) -> None:
        """
        Group buffers together to make frames.

        Pull out buffers in `serial_buffer_queue`, then get frame and buffer index by
        parsing headers in the buffer.
        The buffers belonging to the same frame are put in the same list at
        corresponding buffer index.
        The lists representing each frame are then put into `frame_buffer_queue`.

        Parameters
        ----------
        serial_buffer_queue : multiprocessing.Queue[bytes]
            Input buffer queue.
        frame_buffer_queue : multiprocessing.Queue[ndarray]
            Output frame queue.
        """
        locallogs = init_logger("streamDaq.buffer")

        cur_fm_num = -1  # Frame number

        frame_buffer_prealloc = [np.zeros(bufsize, dtype=np.uint8) for bufsize in self.buffer_npix]
        frame_buffer = frame_buffer_prealloc.copy()
        header_list = []

        try:
            for serial_buffer in exact_iter(serial_buffer_queue.get, None):
                header_data, serial_buffer = self._parse_header(serial_buffer)

                if cur_fm_num == -1 and header_data.frame_buffer_count != 0:
                    # discard until we see a buffer 0 to align to the start of a frame
                    continue

                # update buffer_recv_index only for processed buffers
                header_data.runtime_metadata.buffer_recv_index = self._buffer_recv_index
                self._buffer_recv_index += 1

                try:
                    serial_buffer = self._trim(
                        serial_buffer,
                        self.buffer_npix,
                        header_data,
                        locallogs,
                    )
                except IndexError:
                    locallogs.exception(
                        f"Frame {header_data.frame_num}; Buffer {header_data.buffer_count} "
                        f"(#{header_data.frame_buffer_count} in frame)\n"
                        f"Frame buffer count {header_data.frame_buffer_count} "
                        f"exceeds buffer number per frame {len(self.buffer_npix)}\n"
                        f"Discarding buffer.\n"
                        f"-- THERE IS AN ERROR IN YOUR CONFIGURATION CAUSING YOU TO LOSE DATA --\n"
                        f"If you are seeing this emitted on every frame, "
                        f"The device is sending more buffers per frame than expected based on "
                        f"the configured frame width, height, and buffer size. "
                        f"You must fix the configuration such that it matches the data being sent "
                        f"by the device."
                    )
                    continue

                # if first buffer of a frame
                if header_data.frame_num != cur_fm_num:
                    # push previous frame_buffer into frame_buffer queue if we had one
                    if cur_fm_num != -1:
                        try:
                            frame_buffer_queue.put(
                                (frame_buffer, header_list),
                                block=True,
                                timeout=self.config.runtime.queue_put_timeout,
                            )
                        except queue.Full:
                            locallogs.warning("Frame buffer queue full, skipping frame.")

                    # init new frame_buffer
                    frame_buffer = frame_buffer_prealloc.copy()
                    header_list = []

                    # update frame_num and index
                    cur_fm_num = header_data.frame_num

                    if header_data.frame_buffer_count != 0:
                        locallogs.warning(
                            f"Frame {cur_fm_num} started with buffer "
                            f"{header_data.frame_buffer_count}"
                        )

                # update data and record header for the current (possibly new) frame
                frame_buffer[header_data.frame_buffer_count] = serial_buffer
                header_list.append(header_data)
                locallogs.debug("----buffer #" + str(header_data.frame_buffer_count) + " stored")
        finally:
            try:
                # get remaining buffers.
                frame_buffer_queue.put(
                    (None, header_list), block=True, timeout=self.config.runtime.queue_put_timeout
                )
            except queue.Full:
                locallogs.warning("Frame buffer queue full, skipping frame.")

            try:
                frame_buffer_queue.put(
                    None, block=True, timeout=self.config.runtime.queue_put_timeout
                )
                locallogs.debug("Quitting, putting sentinel in queue")
            except queue.Full:
                locallogs.error("Frame buffer queue full, Could not put sentinel.")

    def _format_frame(
        self,
        frame_buffer_queue: multiprocessing.Queue,
        imagearray: multiprocessing.Queue,
    ) -> None:
        """
        Construct frame from grouped buffers.

        Each frame data is concatenated from a list of buffers in `frame_buffer_queue`
        according to `buffer_npix`.
        If there is any mismatch between the expected length of each buffer
        (defined by `buffer_npix`) and the actual length, then the buffer is either
        truncated or zero-padded at the end to make the length appropriate,
        and a warning is thrown.
        Finally, the concatenated buffer data are converted into a 1d numpy array with
        uint8 dtype and put into `imagearray` queue.

        Parameters
        ----------
        frame_buffer_queue : multiprocessing.Queue[list[bytes]]
            Input buffer queue.
        imagearray : multiprocessing.Queue[np.ndarray]
            Output image array queue.
        """
        locallogs = init_logger("streamDaq.frame")
        frame_index_counter = 0
        try:
            for frame_data, header_list in exact_iter(frame_buffer_queue.get, None):
                if not frame_data or len(frame_data) == 0:
                    try:
                        imagearray.put(
                            (None, header_list),
                            block=True,
                            timeout=self.config.runtime.queue_put_timeout,
                        )
                    except queue.Full:
                        locallogs.warning("Image array queue full, skipping frame.")
                    # Don't increment frame_index_counter for empty frames
                    continue
                frame_data = np.concatenate(frame_data, axis=0)

                try:
                    frame = np.reshape(
                        frame_data, (self.config.frame_width, self.config.frame_height)
                    )
                except ValueError as e:
                    expected_size = self.config.frame_width * self.config.frame_height
                    provided_size = frame_data.size
                    locallogs.exception(
                        "Frame size doesn't match: %s. "
                        " Expected size: %d, got size: %d."
                        "Replacing with zeros.",
                        e,
                        expected_size,
                        provided_size,
                    )
                    frame = np.zeros(
                        (self.config.frame_width, self.config.frame_height), dtype=np.uint8
                    )

                # Populate reconstructed_frame_index for all headers in this frame
                for header in header_list:
                    header.runtime_metadata.reconstructed_frame_index = frame_index_counter

                try:
                    imagearray.put(
                        (frame, header_list),
                        block=True,
                        timeout=self.config.runtime.queue_put_timeout,
                    )
                except queue.Full:
                    locallogs.warning("Image array queue full, skipping frame.")

                frame_index_counter += 1
        finally:
            locallogs.debug("Quitting, putting sentinel in queue")
            try:
                imagearray.put(None, block=True, timeout=self.config.runtime.queue_put_timeout)
            except queue.Full:
                locallogs.error("Image array queue full, Could not put sentinel.")

    def alive_processes(self) -> List[multiprocessing.Process]:
        """
        Return a list of alive processes.

        Returns
        -------
        List[multiprocessing.Process]
            List of alive processes.
        """

        raise NotImplementedError("Not implemented yet")
        return None

    def capture(
        self,
        source: Literal["uart", "fpga"],
        read_length: Optional[int] = None,
        video: Optional[Path] = None,
        video_kwargs: Optional[dict] = None,
        metadata: Optional[Path] = None,
        binary: Optional[Path] = None,
        show_video: Optional[bool] = True,
        show_metadata: Optional[bool] = False,
        freq_mask_config: Optional[FreqencyMaskingConfig] = None,
    ) -> None:
        """
        Entry point to start frame capture.

        Parameters
        ----------
        source : Literal[uart, fpga]
            Device source.
        read_length : Optional[int], optional
            Passed to :func:`~mio.stream_daq.stream_daq._fpga_recv` when
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

        Raises
        ------
        ValueError
            If `source` is not in `("uart", "fpga")`.
        """
        self.terminate.clear()

        shared_resource_manager = multiprocessing.Manager()
        serial_buffer_queue = shared_resource_manager.Queue(
            self.config.runtime.serial_buffer_queue_size
        )
        frame_buffer_queue = shared_resource_manager.Queue(
            self.config.runtime.frame_buffer_queue_size
        )
        imagearray = shared_resource_manager.Queue(self.config.runtime.image_buffer_queue_size)

        if source == "uart":
            self.logger.debug("Starting uart capture process")
            p_recv = multiprocessing.Process(
                target=self._uart_recv,
                args=(
                    serial_buffer_queue,
                    self.config["port"],
                    self.config["baudrate"],
                ),
            )
        elif source == "fpga":
            self.logger.debug("Starting fpga capture process")
            p_recv = multiprocessing.Process(
                target=self._fpga_recv,
                args=(serial_buffer_queue, read_length, True, binary),
                name="_fpga_recv",
            )
        else:
            raise ValueError(f"source can be one of uart or fpga. Got {source}")

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
            )

        p_buffer_to_frame = multiprocessing.Process(
            target=self._buffer_to_frame,
            args=(
                serial_buffer_queue,
                frame_buffer_queue,
            ),
            name="_buffer_to_frame",
        )
        p_format_frame = multiprocessing.Process(
            target=self._format_frame,
            args=(
                frame_buffer_queue,
                imagearray,
            ),
            name="_format_frame",
        )

        p_recv.start()
        p_buffer_to_frame.start()
        p_format_frame.start()

        if show_metadata:
            self._header_plotter = StreamPlotter(
                header_keys=self.config.runtime.plot.keys,
                history_length=self.config.runtime.plot.history,
                update_ms=self.config.runtime.plot.update_ms,
            )

        if metadata:
            header_cols = StreamBufferHeader.csv_header_cols(self.header_fmt)
            self._buffered_writer = BufferedCSVWriter(
                metadata, header=header_cols, buffer_size=self.config.runtime.csvwriter.buffer
            )

        try:
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
            for p in [p_recv, p_buffer_to_frame, p_format_frame]:
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
        writer: Optional[VideoWriter],
        show_metadata: bool,
        metadata: Optional[Path] = None,
        freq_mask_helper: Optional[FrequencyMaskHelper] = None,
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


def iter_buffers(
    source: Iterator[bytes],
    preamble: Bits,
    pre_first: bool = True,
    capture_binary: Optional[Path] = None,
) -> Generator[bytes, None, None]:
    """
    Given some iterator that yields bytes (like a camera device),
    yield buffers from that iterator as `bytes` objects
    split by the `preamble` delimiter.

    Args:
        source (Iterator[bytes]): The iterator that yields bytes
        preamble (Bits): The delimiter bit series to split buffers by
        pre_first (bool | None): Whether preamble/header is returned
            at the beginning of each buffer, by default True.
        capture_binary (Path | None): save binary directly from the ``okDev`` to the supplied path,
            if present.
    """
    logger = init_logger("streamDaq.iter_buffers")
    cur_buffer = BitArray()
    while True:
        try:
            buf = next(source)
        except (EndOfRecordingException, KeyboardInterrupt, StopIteration):
            logger.debug("Got end of recording exception, breaking")
            return
        except StreamReadError:
            logger.exception("Read failed, continuing")
            # It might be better to choose continue or break with a continuous flag
            continue

        if capture_binary:
            with open(capture_binary, "ab") as file:
                file.write(buf)

        dat = BitArray(buf)
        cur_buffer = cur_buffer + dat
        pre_pos = list(cur_buffer.findall(preamble))
        for buf_start, buf_stop in zip(pre_pos[:-1], pre_pos[1:]):
            if not pre_first:
                buf_start, buf_stop = (
                    buf_start + len(preamble),
                    buf_stop + len(preamble),
                )
            yield cur_buffer[buf_start:buf_stop].tobytes()

        if pre_pos:
            cur_buffer = cur_buffer[pre_pos[-1] :]


# DEPRECATION: v0.3.0
if __name__ == "__main__":
    import warnings

    warnings.warn(
        "Calling the stream_daq.py module directly is deprecated - use the `mio` cli. "
        "try:\n\n  mio stream capture --help",
        stacklevel=1,
    )
    sys.exit(1)
