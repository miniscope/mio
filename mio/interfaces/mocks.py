"""
Hardware mocks for interfaces.

Used in testing, but kept in-package since for now some interfaces
need modifications to their source (and we can't import from tests)

Not to be considered part of the public interface of mio <3
"""

# ruff: noqa: D102

import os
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from mio.exceptions import EndOfRecordingException


class okDevMock:
    """
    Mock class for :class:`~mio.interfaces.opalkelly.okDev`
    """

    DATA_FILE: Path | None = None
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
        self.bit_file: Path | None = None

        self._wires: dict[int, int] = {}
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
        self, length: int | None = None, addr: int = 0xA0, blockSize: int = 16
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
