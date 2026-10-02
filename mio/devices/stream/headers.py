"""Headers & metadata for stream devices"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandera.pandas as pa
from bitstring import Bits
from pydantic import Field, computed_field

from mio.bit_operation import BufferFormatter
from mio.devices.base.headers import BufferHeader
from mio.models import MiniscopeConfig
from mio.models.models import Table

if TYPE_CHECKING:
    from mio.devices.stream.config import StreamDevConfig

from typing import Self


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

        header, payload = BufferFormatter.bytebuffer_to_ndarrays(
            buffer=buffer,
            header_length_words=int(config.header_len / 32),
            preamble_length_words=int(len(Bits(config.preamble)) / 32),
            reverse_header_bits=config.reverse_header_bits,
            reverse_header_bytes=config.reverse_header_bytes,
            reverse_payload_bits=config.reverse_payload_bits,
            reverse_payload_bytes=config.reverse_payload_bytes,
        )

        runtime_metadata = dict(
            buffer_recv_index=-1,  # will be set later in buffer_to_frame for processed buffers
            buffer_recv_unix_time=time.time(),
        )
        header_data = StreamBufferHeader.from_sequence(header.astype(int), **runtime_metadata)
        header_data.adc_scaling = config.adc_scale
        return header_data, payload


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
