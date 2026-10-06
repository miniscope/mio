"""Headers & metadata for stream devices"""

from __future__ import annotations

import struct
import time
import zlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandera.pandas as pa
from bitstring import Bits
from pydantic import BaseModel, Field, computed_field

from mio.bit_operation import BufferFormatter
from mio.devices.base.headers import BufferHeader
from mio.models import MiniscopeConfig
from mio.models.models import Table

if TYPE_CHECKING:
    from mio.devices.stream.config import StreamDevConfig

from typing import Self

VERSION_RECORD_LENGTH = 32
VERSION_RECORD_MAGIC = 0xA5


class ADCScaling(MiniscopeConfig):
    """
    Configuration for the ADC scaling factors
    """

    ref_voltage: float = Field(
        1.1,
        description="Reference voltage of the ADC",
    )
    bitdepth: int = Field(
        8,
        description="Bit depth of the ADC",
    )
    battery_div_factor: float = Field(
        5.0,
        description="Voltage divider factor for the battery voltage",
    )
    vin_div_factor: float = Field(
        11.3,
        description="Voltage divider factor for the Vin voltage",
    )

    def scale_battery_voltage(self, voltage_raw: float) -> float:
        """
        Scale raw input ADC voltage to Volts

        Args:
            voltage_raw: Voltage as output by the ADC

        Returns:
            float: Scaled voltage
        """
        return voltage_raw / 2**self.bitdepth * self.ref_voltage * self.battery_div_factor

    def scale_input_voltage(self, voltage_raw: float) -> float:
        """
        Scale raw input ADC voltage to Volts

        Args:
            voltage_raw: Voltage as output by the ADC

        Returns:
            float: Scaled voltage
        """
        return voltage_raw / 2**self.bitdepth * self.ref_voltage * self.vin_div_factor


class RuntimeMetadata(MiniscopeConfig):
    """
    Runtime metadata for data streams.
    """


class StreamBufferHeader(BufferHeader):
    """
    Refinements of :class:`.BufferHeader` for
    :class:`~mio.devices.stream.StreamDevice`

    Additional runtime keys not specified in ``POSITIONS`` must be provided
    when instantiating the object as ``kwargs`` to the ``from_sequence`` method.
    """

    POSITIONS: ClassVar[dict[str, int]] = {
        "linked_list": 0,
        "frame_num": 1,
        "buffer_count": 2,
        "frame_buffer_count": 3,
        "write_buffer_count": 4,
        "dropped_buffer_count": 5,
        "timestamp": 6,
        "write_timestamp": 8,
        "pixel_count": 7,
        "battery_voltage_raw": 9,
        "input_voltage_raw": 10,
    }

    pixel_count: int
    battery_voltage_raw: int
    input_voltage_raw: int

    # runtime metadata
    buffer_recv_index: int = Field(
        -1,
        description=(
            "Index of the buffer received since the start of the stream data acquisition. "
            "Note: This is different from the device's internal buffer index, "
            "which counts buffers from device boot. "
            "buffer index -1 shouldn't exist in the output data as this value should always be set."
        ),
    )
    buffer_recv_unix_time: float = Field(
        -1.0,
        description="Unix time when the buffer was received",
    )
    black_padding_px: int = Field(
        -1,
        description="Number of black padding pixels added to the end of each buffer",
    )
    reconstructed_frame_index: int = Field(
        -1,
        description=(
            "Index of the frame since the start of stream data acquisition. "
            "This value matches the frame index in the output video file. "
            "Note: This is different from the device's internal frame_index, "
            "which counts frames from device boot, "
            "and also counts frames that failed to be reconstructed. "
            "If the buffer is not part of a valid frame, this will be -1."
        ),
    )
    header_crc_ok: bool | None = Field(
        None,
        description=(
            "Whether the header CRC sent by the firmware matched. "
            "None if the config does not enable ``header_crc``."
        ),
    )
    version_byte: int | None = Field(
        None,
        description=(
            "Byte ``buffer_count % 32`` of the firmware version record, "
            "the first byte of the header CRC word (see :class:`.FirmwareVersionRecord`). "
            "None if the config does not enable ``header_crc``."
        ),
    )

    _adc_scaling: ADCScaling = None

    @property
    def adc_scaling(self) -> ADCScaling | None:
        """
        :class:`.ADCScaling` applied to voltage readings
        """
        return self._adc_scaling

    @adc_scaling.setter
    def adc_scaling(self, scaling: ADCScaling) -> None:
        self._adc_scaling = scaling

    @computed_field
    def battery_voltage(self) -> float:
        """
        Scaled battery voltage in Volts.
        """
        if self._adc_scaling is None:
            return self.battery_voltage_raw
        else:
            return self._adc_scaling.scale_battery_voltage(self.battery_voltage_raw)

    @computed_field
    def input_voltage(self) -> float:
        """
        Scaled input voltage in Volts.
        """
        if self._adc_scaling is None:
            return self.input_voltage_raw
        else:
            return self._adc_scaling.scale_input_voltage(self.input_voltage_raw)

    @classmethod
    def from_buffer(cls, buffer: bytes, config: StreamDevConfig) -> tuple[Self, np.ndarray]:
        """
        Parse a header and its payload from the raw buffer from the hardware
        """

        header = BufferFormatter.bytebuffer_to_header(
            buffer=buffer,
            header_length_words=int(config.header_len / 32),
            preamble_length_words=int(len(Bits(config.preamble)) / 32),
            reverse_header_bits=config.reverse_header_bits,
            reverse_header_bytes=config.reverse_header_bytes,
        )

        runtime_metadata = dict(
            buffer_recv_index=-1,  # will be set later in buffer_to_frame for processed buffers
            buffer_recv_unix_time=time.time(),
        )
        if config.header_crc:
            # the last 3 header bytes are the low 24 bits of a CRC-32 of the bytes before them
            header_bytes = header.view(np.uint8)
            crc = int.from_bytes(header_bytes[-3:], "little")
            runtime_metadata["header_crc_ok"] = (zlib.crc32(header_bytes[:-3]) & 0xFFFFFF) == crc
            # the first byte of the CRC word
            runtime_metadata["version_byte"] = int(header_bytes[-4])

        if runtime_metadata.get("header_crc_ok") is False:
            # the buffer is dropped downstream, so skip unpacking its pixels
            payload = np.empty(0, dtype=np.uint8)
        else:
            payload = BufferFormatter.bytebuffer_to_payload(
                buffer=buffer,
                header_length_words=int(config.header_len / 32),
                reverse_payload_bits=config.reverse_payload_bits,
                reverse_payload_bytes=config.reverse_payload_bytes,
            )

        header_data = StreamBufferHeader.from_sequence(header.astype(int), **runtime_metadata)
        header_data.adc_scaling = config.adc_scale
        return header_data, payload


class FirmwareVersionRecord(BaseModel):
    """
    Firmware version record, sent one byte per buffer in the first byte of the header CRC word
    (byte index = ``buffer_count % 32``), so it can be assembled from any 32 consecutive buffers.

    Byte layout: 0 magic 0xA5, 1 record format, 2-4 firmware version major, minor, patch,
    5-8 git hash (little endian), 9 header layout version, 10 flags, 11 device id,
    12-17 image width, height, black reference pixels (uint16, little endian), 18 frame rate,
    19 number of buffers, 20 buffer block length, 21 git tree dirty, 22-30 reserved,
    31 checksum (all 32 bytes sum to 0 modulo 256).
    """

    format: int
    fw_version: str
    git_hash: str
    git_dirty: bool
    header_layout: int
    flags: int
    device_id: int
    image_width: int
    image_height: int
    blackref_px: int
    frame_rate: int
    num_buffers: int
    buffer_block_length: int

    @classmethod
    def from_bytes(cls, record: bytes) -> Self | None:
        """Decode a complete record, or ``None`` if its magic byte or checksum is wrong"""
        if record[0] != VERSION_RECORD_MAGIC or sum(record) % 256 != 0:
            return None
        major, minor, patch = record[2:5]
        width, height, blackref = struct.unpack_from("<HHH", record, 12)
        return cls(
            format=record[1],
            fw_version=f"{major}.{minor}.{patch}",
            git_hash=f"{struct.unpack_from('<I', record, 5)[0]:08x}",
            git_dirty=bool(record[21]),
            header_layout=record[9],
            flags=record[10],
            device_id=record[11],
            image_width=width,
            image_height=height,
            blackref_px=blackref,
            frame_rate=record[18],
            num_buffers=record[19],
            buffer_block_length=record[20],
        )


class StreamBufferTable(Table):
    """
    Table form of the stream
    """

    _RECORD_MODEL = StreamBufferHeader

    linked_list: int = pa.Field(ge=0, coerce=True)
    frame_num: int = pa.Field(ge=0, coerce=True)
    buffer_count: int = pa.Field(ge=0, coerce=True)
    frame_buffer_count: int = pa.Field(ge=0, coerce=True)
    write_buffer_count: int = pa.Field(ge=0, coerce=True)
    dropped_buffer_count: int = pa.Field(ge=0, coerce=True)
    timestamp: int = pa.Field(ge=0, coerce=True)
    pixel_count: int = pa.Field(ge=0, coerce=True)
    write_timestamp: int = pa.Field(ge=0, coerce=True)
    battery_voltage_raw: float = pa.Field(ge=0, coerce=True)
    input_voltage_raw: float = pa.Field(ge=0, coerce=True)
    buffer_recv_index: int = pa.Field(ge=0, coerce=True)
    buffer_recv_unix_time: float = pa.Field(ge=0, coerce=True)
    black_padding_px: int = pa.Field(ge=0, coerce=True)
    reconstructed_frame_index: int = pa.Field(ge=0, coerce=True)
    header_crc_ok: bool | None = pa.Field(nullable=True, coerce=True)
    version_byte: int | None = pa.Field(nullable=True, coerce=True)
