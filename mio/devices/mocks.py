"""
Hardware mocks for devices.

Used in testing, but kept in-package since for now some devices
need modifications to their source (and we can't import from tests)

Not to be considered part of the public interface of mio <3
"""

# ruff: noqa: D102

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np

from mio.exceptions import EndOfRecordingException


class okDevMock:
    """
    Mock class for :class:`~mio.devices.opalkelly.okDev`
    """

    DATA_FILE: Optional[Path] = None
    """
    Recorded data file to use for simulating read.
    
    Set as class variable so that it can be monkeypatched in tests that
    require different source data files.
    
    Can be set using the ``PYTEST_OKDEV_DATA_FILE`` environment variable if 
    this mock is to be used within a separate process.
    """

    def __init__(
        self,
        read_length: int,
        serial_id: str = "",
    ):
        self.read_length = read_length
        self.serial_id = serial_id
        self.bit_file: Optional[Path] = None

        self._wires: Dict[int, int] = {}
        self._buffer_position = 0

        # preload the data file to a byte array
        if self.DATA_FILE is None:
            if os.environ.get("PYTEST_OKDEV_DATA_FILE") is not None:
                # need to get file from env variables here because on some platforms
                # the default method for creating a new process is "spawn" which creates
                # an entirely new python session instead of "fork" which would preserve
                # the classvar
                data_file: str = os.environ.get("PYTEST_OKDEV_DATA_FILE")  # type: ignore

                self.DATA_FILE = Path(data_file)
                okDevMock.DATA_FILE = Path(data_file)
            else:
                raise RuntimeError("DATA_FILE class attr must be set before using the mock")

        with open(self.DATA_FILE, "rb") as dfile:
            self._buffer = bytearray(dfile.read())

    def upload_bit(self, bit_file: str) -> None:
        assert Path(bit_file).exists()
        self.bit_file = Path(bit_file)

    def read_data(
        self, length: Optional[int] = None, addr: int = 0xA0, blockSize: int = 16
    ) -> bytearray:
        if length is None:
            length = self.read_length

        if self._buffer_position >= len(self._buffer):
            # Error if called after we have returned the last data
            raise EndOfRecordingException("End of sample buffer")

        end_pos = min(self._buffer_position + length, len(self._buffer))
        data = self._buffer[self._buffer_position : end_pos]
        self._buffer_position = end_pos
        return data

    def set_wire(self, addr: int, val: int) -> None:
        self._wires[addr] = val

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> bytes:
        try:
            return self.read_data()
        except EndOfRecordingException as e:
            raise StopIteration() from e


class USBCamMock:
    """
    Mock class for :class:`cv2.VideoCapture`.

    Replays frames from a ``.npz`` file recorded with ``mio usbcam record --binary_export``
    with keys ``frames`` (N, H, W, C) uint8 and ``timestamps`` (N,) float64.

    Set as class variable so that it can be monkeypatched in tests that
    require different source data files.

    Can be set using the ``PYTEST_USBCAM_DATA_FILE`` environment variable if
    this mock is to be used within a separate process.
    """

    DATA_FILE: Optional[Path] = None
    REALTIME: bool = False
    """If True, sleep between frames to match recorded timestamps.

    Can also be set via ``PYTEST_USBCAM_REALTIME=1`` env var for multiprocessing.
    """

    def __init__(self) -> None:
        if self.DATA_FILE is None:
            if os.environ.get("PYTEST_USBCAM_DATA_FILE") is not None:
                # need to get file from env variables here because on some platforms
                # the default method for creating a new process is "spawn" which creates
                # an entirely new python session instead of "fork" which would preserve
                # the classvar
                data_file: str = os.environ.get("PYTEST_USBCAM_DATA_FILE")  # type: ignore
                self.DATA_FILE = Path(data_file)
                USBCamMock.DATA_FILE = Path(data_file)
            else:
                raise RuntimeError("DATA_FILE class attr must be set before using USBCamMock")

        if not self.REALTIME and os.environ.get("PYTEST_USBCAM_REALTIME"):
            self.REALTIME = True  # Not sure why this is needed but following the OKDevMock for now
            USBCamMock.REALTIME = True

        data = np.load(self.DATA_FILE)
        self._frames = data["frames"]
        self._timestamps = data["timestamps"]
        self._position = 0
        self._opened = True
        self._props: Dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - match cv2.VideoCapture API
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._position >= len(self._frames):
            raise EndOfRecordingException("End of recorded frames")

        if self.REALTIME and self._position > 0:
            dt = self._timestamps[self._position] - self._timestamps[self._position - 1]
            if dt > 0:
                time.sleep(dt)

        frame = self._frames[self._position]
        self._position += 1
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        self._props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self._props.get(prop, 0.0)

    def release(self) -> None:
        self._opened = False
