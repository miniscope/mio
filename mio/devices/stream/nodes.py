"""
Separable processing operations for streaming devices
"""

import os
import time
from collections.abc import Callable, Generator
from functools import cached_property
from itertools import count
from pathlib import Path
from typing import Annotated as A
from typing import Any, TypeVar, Union
from typing import Literal as L

import numpy as np
from bitstring import BitArray, Bits
from noob import Name, Node, NoEventable
from noob.event import MetaSignal
from pydantic import PrivateAttr

from mio import init_logger
from mio.devices.stream import StreamBufferHeader, StreamDevConfig
from mio.exceptions import DeviceConfigurationError
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


def iter_fpga(config: StreamDevConfig) -> Generator[A[bytes, Name("chunk")], None, None]:
    """
    Iterate a raw binary stream from the FPGA in chunks
    (not necessarily split into buffers)
    """
    # set up fpga interfaces
    dev = init_okdev(config.bitstream, config.read_length)

    yield from dev


class SplitBuffers(Node):
    """
    Collect raw chunks from the FPGA, yield buffers

    The data are passed in fixed chunks.
    Then we concatenate the chunks and try to look for {attr}`.SplitBuffers.preamble` in the data.
    The data between every pair of {attr}`.SplitBuffers.preamble`
    is considered to be a single buffer and yielded.
    """

    config: StreamDevConfig
    _buffer: BitArray = PrivateAttr(default_factory=BitArray)

    @cached_property
    def preamble(self) -> Bits:
        """
        The preamble that indicates the start of a buffer.

        If ``config.reverse_header_bits`` is true, the preamble is bit-flipped
        """
        pre = Bits(self.config.preamble)
        if self.config.reverse_header_bits:
            pre = pre[::-1]
        return pre

    def process(self, chunk: bytes) -> A[list[bytes] | L[MetaSignal.NoEvent], Name("chunks")]:
        """
        Append the chunk to the buffer, return any split buffers, if found.
        """
        self._buffer += BitArray(chunk)
        pos = list(self._buffer.findall(self.preamble))
        buffers = [self._buffer[start:stop].tobytes() for start, stop in zip(pos[:-1], pos[1:])]
        if buffers:
            self._buffer = self._buffer[pos[-1] :]
            return buffers
        else:
            return MetaSignal.NoEvent

    def deinit(self) -> None:
        """Clear the internal buffer"""
        self._buffer = BitArray()


_THeader = TypeVar("_THeader", bound=StreamBufferHeader)


class ParseHeader(Node):
    """
    Parse a raw binary chunk into its header and 1-dimensional pixel buffer array

    If a counter is provided, mark :attr:`StreamBufferHeader.buffer_recv_index`
    to track the count of buffers received by mio,
    which might differ from the on-device count.
    """

    stateful: bool = True

    config: StreamDevConfig
    header_cls: type[_THeader] = StreamBufferHeader

    _seen_start_buffer: bool = False
    """
    Whether or not we have seen a buffer with a 0 index, the start of a frame.
    Drop buffers/headers until we do. 
    """

    def process(
        self,
        chunk: bytes,
        counter: count | None = None,
    ) -> tuple[
        A[NoEventable[_THeader], Name("header")], A[NoEventable[np.ndarray], Name("buffer")]
    ]:
        header_data, buffer = self.header_cls.from_buffer(chunk, self.config)

        if not self._seen_start_buffer:
            if header_data.frame_buffer_count != 0:
                return MetaSignal.NoEvent, MetaSignal.NoEvent
            else:
                self._seen_start_buffer = True

        if counter is not None:
            header_data.buffer_recv_index = next(counter)
        init_logger("parse_header").debug("HEADER: %s", header_data)
        return header_data, buffer


class CombineBuffers(Node):
    """Collect buffers until we the header tells us that we're in a new frame"""

    config: StreamDevConfig

    _buffers: list[np.ndarray] = PrivateAttr(default_factory=list)
    _buffers_prealloc: list[np.ndarray] = PrivateAttr(default_factory=list)
    _current_frame: int = -1
    _frame_idx: int = 0

    def process(
        self, buffer: np.ndarray, header: StreamBufferHeader
    ) -> tuple[A[NoEventable[np.ndarray], Name("frame")], A[int, Name("frame_idx")]]:
        # when starting, wait for the start of a new frame
        if self._current_frame == -1:
            if header.frame_buffer_count != 0:
                return MetaSignal.NoEvent, self._frame_idx
            else:
                self._current_frame = header.frame_num

        if header.frame_num != self._current_frame:
            # return the completed, previous frame - this is a new frame!
            buffers = self._buffers

            # stash this buffer and prepare for next iteration
            self._buffers = [None for _ in range(len(self._buffers_prealloc))]
            self._buffers[header.frame_buffer_count] = buffer
            self._current_frame = header.frame_num
            self._frame_idx += 1

            # fill in missing buffers with zeros (the header csv will show this as a missing buffer)
            for i in range(len(buffers)):
                if buffers[i] is None:
                    buffers[i] = self._buffers_prealloc[i]
            try:
                frame = np.concatenate(buffers, axis=0).reshape(
                    (self.config.frame_width, self.config.frame_height)
                )
            except ValueError as e:
                raise DeviceConfigurationError(
                    f"Could not reshape frame, "
                    f"expected ({self.config.frame_width}, {self.config.frame_height}), "
                    f"{self.config.frame_width * self.config.frame_height}px, "
                    f"but buffers were {sum(len(b) for b in buffers)} pixels"
                ) from e

            return frame, self._frame_idx
        else:
            self._buffers[header.frame_buffer_count] = buffer
            return MetaSignal.NoEvent, self._frame_idx

    def init(self) -> None:
        self._buffers_prealloc = [
            np.zeros(bufsize, dtype=np.uint8) for bufsize in self.config.buffer_npix
        ]
        self._buffers = [None for _ in range(len(self._buffers_prealloc))]

    def deinit(self) -> None:
        """
        Clear mutable state *except* for the buffer index,
        which should continue incrementing across stop/start cycles.
        """
        self._buffers = []
        self._current_frame = -1


def imshow(frame: np.ndarray, window: str = "image") -> None:
    """
    Show an image with opencv imshow.

    (Need to wrap the function because the c extension doesn't ``inspect`` correctly)
    """
    import cv2

    cv2.imshow(window, frame)


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


def init_okdev(BIT_FILE: Path, read_length: int) -> Union["okDev", okDevMock]:
    """
    Create a connection to an :class:`.okDev` device

    Writes to the FPGA to reset its state:

    * put the manchester decoder at reset mode
    * sleep 0.01
    * un-reset all components
    * put the clock generator/multiplier at reset
    * sleep 0.01
    * un-reset all components (presumably the system is ready at this point).

    """
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


def trim_or_pad(
    buffer: np.ndarray, header: StreamBufferHeader, config: StreamDevConfig
) -> tuple[A[NoEventable[np.ndarray], Name("buffer")], A[StreamBufferHeader, Name("header")]]:
    """
    Trim or pad an array to match an expected size

    .. todo::
        Re-think about the timing to deal with dummy words.
        It feels cleaner to remove these dummy words right after the preamble detections.
        That way, all data we inject into later stages will be pure metadata and pixel data.
        This isn't critical and I don't want to slow down detection so skipping for now.
    """
    try:
        expected_data_size = config.buffer_npix[header.frame_buffer_count]
    except IndexError:
        logger = init_logger("stream.trim_or_pad")
        logger.exception(
            f"Frame {header.frame_num}; Buffer {header.buffer_count} "
            f"(#{header.frame_buffer_count} in frame)\n"
            f"Frame buffer count {header.frame_buffer_count} "
            f"exceeds buffer number per frame {len(config.buffer_npix)}\n"
            f"Discarding buffer.\n"
            f"-- THERE IS AN ERROR IN YOUR CONFIGURATION CAUSING YOU TO LOSE DATA --\n"
            f"If you are seeing this emitted on every frame, "
            f"The device is sending more buffers per frame than expected based on "
            f"the configured frame width, height, and buffer size. "
            f"You must fix the configuration such that it matches the data being sent "
            f"by the device."
        )
        return MetaSignal.NoEvent, header

    if buffer.shape[0] != expected_data_size:
        header.black_padding_px = expected_data_size - buffer.shape[0]
        if buffer.shape[0] > expected_data_size:
            buffer = buffer[0:expected_data_size]
        else:
            buffer = np.pad(buffer, (0, header.black_padding_px))
    else:
        header.black_padding_px = 0

    return buffer, header
