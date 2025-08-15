# ruff: noqa: D100

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


import numpy as np

from mio.models.stream import StreamBufferHeader, StreamBufferHeaderFormat

if TYPE_CHECKING:
    from mio.devices.gs.config import GSDevConfig


def buffer_to_array(buffer: bytes) -> np.ndarray:
    """
    Given the GS's "12-bit" pixel format,
    where 10-bit pixels are flanked by two pad values
    e.g. (``1xxxxxxxxxx0``)

    Strip the pads, and return a 16-bit ndarray
    """
    # convert to a binary array 8 at a time
    binary_data = np.unpackbits(np.frombuffer(buffer, dtype=np.uint8))

    # reshape to a n x 12
    pixel_cols = binary_data.reshape((-1, 12))

    # remove padding pixels (12 bit x n --> 10 bit x n)
    stripped = pixel_cols[:, 1:-1]

    # Cast to 16 bit ndarray
    # padded = np.pad(stripped, ((0, 0), (6, 0)), mode="constant", constant_values=0)
    # packed_16bit = np.packbits(padded, axis=1).view(np.uint16).byteswap()
    # return packed_16bit.flatten()

    # cast to an 8 bit ndarray
    stripped_8bit = stripped[:, :-2] # current
    packed_8bit =  np.packbits(stripped_8bit, axis=1).flatten()

    return packed_8bit



class GSBufferHeader(StreamBufferHeader):
    """
    Header at the start of GS buffers -
    Dummy [0-11 32 bit words]
    Preamble [12th 32 bit word] ~ 0x12345678 (LSB = 0x78563412)
    Header [12th 32 bit word]
    Full Data Buffer [3750 32 bit words]
    Partial Data Buffer [1860 32 bit words]

    formatted by :class:`.GSBufferHeaderFormat`
    (...hemal describe data structure...)
    NEC Camera: 328 columns x 320 rows x 12 bit unprocessed pixel
    Processing:
    First 8 columns are alignment buffers
    12 bit unprocessed pixel includes [1][10 bit processed pixel][0]
    Final:
    NE Camera: 320 columns x 320 rows x 10 bit processed pixel
    """

    @classmethod
    def from_buffer(
        cls, buffer: bytes, header_fmt: "GSBufferHeaderFormat", config: "GSDevConfig"
    ) -> tuple[Self, np.ndarray]:
        """Split buffer into a :class:`.GSBufferHeader` and a 1D, 16-bit pixel array."""
        header_start = len(config.preamble)
        header_end = header_start + ((header_fmt.header_length) * 4)  # = 44 ((384-32)/32)  = 11
        header_array = np.frombuffer(buffer[header_start:header_end], dtype=np.uint32)
        header = cls.from_format(header_array, header_fmt, construct=True)
        dummy_len = config.dummy_words * 4
        payload = buffer_to_array(
            buffer[header_end:-dummy_len]
        )  # ignoring the last 384 bits, can change after dummy is detected
        return header, payload


class GSBufferHeaderFormat(StreamBufferHeaderFormat):
    """Positions of header fields in GS headers"""

    pass
