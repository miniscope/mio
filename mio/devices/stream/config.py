"""Config for stream devices"""

from pathlib import Path
from typing import Literal

from pydantic import field_validator

from mio.const import INTERFACES_DIR
from mio.devices.stream import StreamDevRuntime
from mio.devices.stream.headers import ADCScaling
from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin


class StreamDevConfig(MiniscopeConfig, ConfigYAMLMixin):
    """
    Format model used to parse DAQ configuration yaml file (examples are in ./config)
    The model attributes are key-value pairs needed for reconstructing frames from data streams.

    Parameters
    ----------
    device: str
        Interface hardware used for receiving data.
        Current options are "OK" (Opal Kelly XEM 7310)
        Only "OK" is supported at the moment.
    bitstream: str, optional
        Required when device is "OK".
        The configuration bitstream file to upload to the Opal Kelly board.
        This uploads a Manchester decoder HDL and different bitstream files are required
        to configure different data rates and bit polarity.
        This is a binary file synthesized using Vivado,
        and details for generating this file will be provided in later updates.
    frame_width: int
        Frame width of transferred image. This is used to reconstruct image.
    frame_height: int
        Frame height of transferred image. This is used to reconstruct image.
    fs: int
        Framerate of acquired stream
    preamble: str
        32-bit preamble used to locate the start of each buffer.
        The header and image data follows this preamble.
        This is used as a hex but imported as a string because yaml doesn't support hex format.
    header_len : int, optional
        Length of header in bits. (For 32-bit words, 32 * number of words)
        This is useful when not all the variable/words in the header are defined in
        :class:`.MetadataHeaderFormat`.
        The user is responsible to ensure that `header_len` is larger than the largest bit
        position defined in :class:`.MetadataHeaderFormat`
        otherwise unexpected behavior might occur.
    pix_depth : int, optional
        Bit-depth of each pixel, by default 8.
    buffer_block_length: int
        Defines the data buffer structure. This value needs to match the Miniscope firmware.
        Number of blocks per each data buffer.
        This is required to calculate the number of pixels contained in one data buffer.
    block_size: int
        Defines the data buffer structure. This value needs to match the Miniscope firmware.
        Number of 32-bit words per data block.
        This is required to calculate the number of pixels contained in one data buffer.
    num_buffers: int
        Defines the data buffer structure. This value needs to match the Miniscope firmware.
        This is the number of buffers that the source microcontroller cycles around.
        This isn't strictly required for data reconstruction but useful for debugging.
    reverse_header_bits : bool, optional
        If True, reverse the bits within each byte of the header.
        Default is False.
    reverse_header_bytes : bool, optional
        If True, reverse the byte order within each 32-bit word of the header.
        This is used for handling endianness in systems where the byte order needs to be swapped.
        Default is False.
    reverse_payload_bits : bool, optional
        If True, reverse the bits within each byte of the payload.
        Default is False.
    reverse_payload_bytes : bool, optional
        If True, reverse the byte order within each 32-bit word of the payload.
        This is used for handling endianness in systems where the byte order needs to be swapped.
        Default is False.
    dummy_words : int, optional
        Number of 32-bit dummy words in the header.
        This is used to stabilize clock recovery in FPGA Manchester decoder.
        This value does not have a meaning for image recovery.
    """

    device: Literal["OK"] = "OK"
    bitstream: Path | None = None
    frame_width: int
    frame_height: int
    fs: int = 20
    preamble: bytes
    header_len: int
    pix_depth: int = 8
    buffer_block_length: int
    block_size: int
    num_buffers: int
    reverse_header_bits: bool = False
    reverse_header_bytes: bool = False
    reverse_payload_bits: bool = False
    reverse_payload_bytes: bool = False
    dummy_words: int = 0
    adc_scale: ADCScaling | None = ADCScaling()
    runtime: StreamDevRuntime = StreamDevRuntime()

    _px_per_buffer: int = None

    @field_validator("preamble", mode="before")
    def preamble_to_bytes(cls, value: str | bytes | int) -> bytes:
        """
        Cast ``preamble`` to bytes.

        Args:
            value (str, bytes, int): Recast from `str` (in yaml like ``preamble: "0x12345"`` )
                or `int` (in yaml like `preamble: 0x12345`

        Returns:
            bytes
        """
        if isinstance(value, str):
            return bytes.fromhex(value)
        elif isinstance(value, int):
            return bytes.fromhex(hex(value)[2:])
        else:
            return value

    @field_validator("bitstream", mode="after")
    def resolve_relative(cls, value: Path) -> Path:
        """
        If we are given a relative path to a bitstream, resolve it relative to
        the device path
        """
        if not value.is_absolute():
            value = INTERFACES_DIR / value
        return value

    @field_validator("bitstream", mode="after")
    def ensure_exists(cls, value: Path | None) -> Path | None:
        """If a bitstream file has been provided, ensure it exists"""
        if isinstance(value, Path):
            assert (
                value.exists()
            ), f"Configured to use bitstream file {value}, but it does not exist"
        return value

    @property
    def px_per_buffer(self) -> int:
        """
        Number of pixels per buffer
        """

        px_per_word = 32 / self.pix_depth
        if self._px_per_buffer is None:
            self._px_per_buffer = (
                self.buffer_block_length * self.block_size
                - self.header_len / self.pix_depth
                - px_per_word * self.dummy_words
            )
        return self._px_per_buffer

    @property
    def buffer_npix(self) -> list[int]:
        """
        List of pixel counts per buffer for a complete frame.

        A frame is split across multiple buffers. This returns a list where each element
        is the number of pixels in that buffer. The last buffer may have fewer pixels
        (the remainder).
        """
        px_per_frame = self.frame_width * self.frame_height
        # Payload size in bytes (= pixels when pix_depth=8)
        byte_per_word = 4  # 32 bits / 8 bits
        payload_bytes = int(
            self.buffer_block_length * self.block_size
            - self.header_len / 8
            - self.dummy_words * byte_per_word
        )
        quotient, remainder = divmod(px_per_frame, payload_bytes)
        return [payload_bytes] * int(quotient) + ([int(remainder)] if remainder else [])
