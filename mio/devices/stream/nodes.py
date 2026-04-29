"""
Separable processing operations for streaming devices
"""

import logging
import multiprocessing
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any, Union

import numpy as np
from bitstring import BitArray, Bits

from mio import init_logger
from mio.devices.stream import StreamBufferHeader, StreamDevConfig
from mio.exceptions import EndOfRecordingException, StreamReadError
from mio.interfaces.mocks import okDevMock

HAVE_OK = False
ok_error = None
BIT_PER_WORD = 32
okDev = None  # Set if OpalKelly driver is available

try:
    from mio.interfaces.opalkelly import okDev

    HAVE_OK = True
except (ImportError, ModuleNotFoundError):
    pass  # okDev stays None; error raised when actually trying to use FPGA


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


def iter_buffers(
    source: Iterator[bytes],
    preamble: Bits,
    pre_first: bool = True,
    capture_binary: Path | None = None,
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


def init_okdev(BIT_FILE: Path, read_length: int) -> Union["okDev", okDevMock]:
    """Create a connection to an :class:`.okDev` device"""
    # FIXME: when multiprocessing bug resolved, remove this and just mock in tests
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("STREAMDAQ_MOCKRUN"):
        dev = okDevMock(read_length=read_length)
    else:
        if not HAVE_OK:
            raise ImportError(
                "OpalKelly driver not available. Cannot read from FPGA.\n"
                "See: https://docs.opalkelly.com/fpsdk/getting-started/"
            )
        dev = okDev(read_length=read_length)

    dev.upload_bit(str(BIT_FILE))
    dev.set_wire(0x00, 0b0010)
    time.sleep(0.01)
    dev.set_wire(0x00, 0b0)
    dev.set_wire(0x00, 0b1000)
    time.sleep(0.01)
    dev.set_wire(0x00, 0b0)
    return dev


def fpga_recv(
    serial_buffer_queue: multiprocessing.Queue,
    config: StreamDevConfig,
    terminate: mp.Event,
    read_length: int = None,
    pre_first: bool = True,
    capture_binary: Path | None = None,
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
        read_length = int(max(config.buffer_npix) * config.pix_depth / 8 / 16) * 16

    # set up fpga interfaces
    BIT_FILE = config.bitstream
    if not BIT_FILE.exists():
        serial_buffer_queue.put(None)
        raise RuntimeError(f"Configured to use bitfile at {BIT_FILE} but no such file exists")

    # set up fpga interfaces
    dev = init_okdev(BIT_FILE, read_length)

    # read loop
    pre = Bits(config.preamble)
    if config.reverse_header_bits:
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
                    timeout=config.runtime.queue_put_timeout,
                )
            except queue.Full:
                locallogs.warning("Serial buffer queue full, skipping buffer.")
            if terminate.is_set():
                break

    except Exception as e:
        locallogs.exception(f"Exception in fpga_recv: {e}")

    finally:
        locallogs.debug("Quitting, putting sentinel in queue")
        try:
            serial_buffer_queue.put(None, block=True, timeout=config.runtime.queue_put_timeout)
        except queue.Full:
            locallogs.error("Serial buffer queue full, Could not put sentinel.")


def buffer_to_frame(
    serial_buffer_queue: multiprocessing.Queue,
    frame_buffer_queue: multiprocessing.Queue,
    config: StreamDevConfig,
    header_cls: type[StreamBufferHeader],
    terminate: mp.Event,
    buffer_idx_start: int = 0,
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

    frame_buffer_prealloc = [np.zeros(bufsize, dtype=np.uint8) for bufsize in config.buffer_npix]
    frame_buffer = frame_buffer_prealloc.copy()
    header_list = []
    buffer_recv_index = buffer_idx_start

    try:
        for serial_buffer in exact_iter(serial_buffer_queue.get, None):
            header_data, serial_buffer = header_cls.from_buffer(serial_buffer, config)

            if cur_fm_num == -1 and header_data.frame_buffer_count != 0:
                # discard until we see a buffer 0 to align to the start of a frame
                continue

            # update buffer_recv_index only for processed buffers
            header_data.buffer_recv_index = buffer_recv_index
            buffer_recv_index += 1

            try:
                serial_buffer = _trim(
                    serial_buffer,
                    config,
                    config.buffer_npix,
                    header_data,
                    locallogs,
                )
            except IndexError:
                locallogs.exception(
                    f"Frame {header_data.frame_num}; Buffer {header_data.buffer_count} "
                    f"(#{header_data.frame_buffer_count} in frame)\n"
                    f"Frame buffer count {header_data.frame_buffer_count} "
                    f"exceeds buffer number per frame {len(config.buffer_npix)}\n"
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
                            timeout=config.runtime.queue_put_timeout,
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
            if terminate.is_set():
                break

    except Exception as e:
        locallogs.exception(f"Exception in buffer_to_frame: {e}")

    finally:
        try:
            # get remaining buffers.
            frame_buffer_queue.put(
                (None, header_list), block=True, timeout=config.runtime.queue_put_timeout
            )
        except queue.Full:
            locallogs.warning("Frame buffer queue full, skipping frame.")

        try:
            frame_buffer_queue.put(None, block=True, timeout=config.runtime.queue_put_timeout)
            locallogs.debug("Quitting, putting sentinel in queue")
        except queue.Full:
            locallogs.error("Frame buffer queue full, Could not put sentinel.")


def format_frame(
    frame_buffer_queue: multiprocessing.Queue,
    imagearray: multiprocessing.Queue,
    config: StreamDevConfig,
    terminate: mp.Event,
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
                        timeout=config.runtime.queue_put_timeout,
                    )
                except queue.Full:
                    locallogs.warning("Image array queue full, skipping frame.")
                # Don't increment frame_index_counter for empty frames
                continue
            frame_data = np.concatenate(frame_data, axis=0)

            try:
                frame = np.reshape(frame_data, (config.frame_width, config.frame_height))
            except ValueError as e:
                expected_size = config.frame_width * config.frame_height
                provided_size = frame_data.size
                locallogs.exception(
                    "Frame size doesn't match: %s. "
                    " Expected size: %d, got size: %d."
                    "Replacing with zeros.",
                    e,
                    expected_size,
                    provided_size,
                )
                frame = np.zeros((config.frame_width, config.frame_height), dtype=np.uint8)

            # Populate reconstructed_frame_index for all headers in this frame
            for header in header_list:
                header.reconstructed_frame_index = frame_index_counter

            try:
                imagearray.put(
                    (frame, header_list),
                    block=True,
                    timeout=config.runtime.queue_put_timeout,
                )
            except queue.Full:
                locallogs.warning("Image array queue full, skipping frame.")

            if terminate.is_set():
                break

            frame_index_counter += 1
    except Exception as e:
        locallogs.exception(f"Exception in format_frame: {e}")

    finally:
        locallogs.debug("Quitting, putting sentinel in queue")
        try:
            imagearray.put(None, block=True, timeout=config.runtime.queue_put_timeout)
        except queue.Full:
            locallogs.error("Image array queue full, Could not put sentinel.")


def _trim(
    data: np.ndarray,
    config: StreamDevConfig,
    expected_size_array: list[int],
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
    if data.shape[0] != expected_payload_size + config.dummy_words * 4:
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
            header.black_padding_px = 0  # No padding, data was trimmed
        # pad if too short
        else:
            padding_amount = expected_data_size - data.shape[0]
            data = np.pad(data, (0, padding_amount))
            header.black_padding_px = padding_amount
    else:
        # No trimming or padding needed
        header.black_padding_px = 0

    return data
