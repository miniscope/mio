"""Headers & metadata for stream devices"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from bitstring import Bits
from pydantic import Field, computed_field

from mio.bit_operation import BufferFormatter
from mio.devices.base.headers import BufferHeader
from mio.models import MiniscopeConfig
from mio.models.sinks import CSVWriterConfig, StreamPlotterConfig

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if TYPE_CHECKING:
    from mio.devices.stream.config import StreamDevConfig


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


class StreamBufferHeader(BufferHeader):
    """
    Refinements of :class:`.BufferHeader` for
    :class:`~mio.devices.stream.StreamDevice`
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
    _adc_scaling: ADCScaling = None

    runtime_metadata: RuntimeMetadata = Field(default_factory=lambda: RuntimeMetadata())

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

    def model_dump_all(self, warning: bool = False) -> dict:
        """
        Return a dictionary of the model values, including runtime metadata if available.

        Returns:
            dict: Dictionary of model values
        """
        meta_row = self.model_dump(warnings=warning)
        if "runtime_metadata" in meta_row and meta_row["runtime_metadata"]:
            runtime_data = meta_row.pop("runtime_metadata")
            meta_row.update(runtime_data)

        return meta_row

    @classmethod
    def from_sequence(
        cls,
        vals: Sequence,
        construct: bool = False,
        runtime_metadata: RuntimeMetadata = None,
    ) -> Self:
        """
        Instantiate a stream buffer header from linearized values (eg. in an ndarray or list),
        an associated format that tells us what index the model values are found in that data,
        and runtime metadata container.

        Args:
            vals (list, :class:`numpy.ndarray` ): Indexable values to cast to the header model
            construct (bool): If ``True`` , use :meth:`~pydantic.BaseModel.model_construct`
                to create the model instance (ie. without validation, but faster).
                Default: ``False``
            runtime_metadata (:class:`.RuntimeMetadata`, optional): Runtime metadata
             to attach to the header.

        Returns:
            :class:`.StreamBufferHeader`
        """
        header = super().from_sequence(vals=vals, construct=construct)
        if runtime_metadata is not None:
            header.runtime_metadata = runtime_metadata
        return header

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

        runtime_metadata = RuntimeMetadata(
            buffer_recv_index=-1,  # will be set later in buffer_to_frame for processed buffers
            buffer_recv_unix_time=time.time(),
        )
        header_data = StreamBufferHeader.from_sequence(
            header.astype(int), runtime_metadata=runtime_metadata
        )
        header_data.adc_scaling = config.adc_scale
        return header_data, payload

    @classmethod
    def csv_header_cols(cls) -> list[str]:
        """
        Return the standardized column names for CSV output.

        This ensures consistent column ordering across all StreamBufferHeader instances
        when writing to CSV files.

        Args:
            header_format: The StreamBufferHeaderFormat instance to get column ordering from

        Returns:
            list[str]: Column names in the order they should appear in CSV output
        """
        # Get the base header format columns (excluding internal fields)
        header_items = sorted(cls.POSITIONS.items(), key=lambda x: x[1])
        base_cols = [name for name, _ in header_items]

        # Add runtime metadata fields from the class's own runtime_metadata attribute
        runtime_fields = list(cls.model_fields["runtime_metadata"].annotation.model_fields.keys())

        return base_cols + runtime_fields


class StreamDevRuntime(MiniscopeConfig):
    """
    Runtime configuration for :class:`.StreamDevice`

    Included within :class:`.StreamDevConfig` to separate config that is not
    unique to the device, but how that device is controlled at runtime.
    """

    serial_buffer_queue_size: int = Field(
        10,
        description="Buffer length for serial data reception in streamDaq",
    )
    frame_buffer_queue_size: int = Field(
        5,
        description="Buffer length for storing frames in streamDaq",
    )
    image_buffer_queue_size: int = Field(
        5,
        description="Buffer length for storing images in streamDaq",
    )
    queue_put_timeout: int = Field(
        5,
        description="Timeout for putting data into the queue",
    )
    plot: StreamPlotterConfig | None = Field(
        StreamPlotterConfig(
            keys=["timestamp", "buffer_count", "frame_buffer_count"], update_ms=1000, history=500
        ),
        description="Configuration for plotting header data as it is collected. "
        "If ``None``, use the default params in StreamPlotter. "
        "Note that this does *not* control whether header metadata is plotted during capture, "
        "for enabling/disabling, use the ``show_metadata`` kwarg in the capture method",
    )
    csvwriter: CSVWriterConfig | None = Field(
        CSVWriterConfig(buffer=100),
        description="Default configuration for writing header data to a CSV file. "
        "If ``None``, use the default params in BufferedCSVWriter. "
        "Note that this does *not* control whether header metadata is written during capture, "
        "for enabling/disabling, use the ``metadata`` kwarg in the capture method.",
    )
    ntp_server: str | None = Field(
        default=None,
        description="NTP server address for time synchronization check. "
        "If specified, the system time will be verified against this server before capture.",
    )
    ntp_max_offset_seconds: float = Field(
        default=0.01,
        description="Maximum allowed time offset in seconds "
        "for NTP synchronization check (default: 0.01 = 10ms).",
    )
    ber_test_n_buffers: int = Field(
        32767,
        description="Number of buffers to consume when running BER test mode. "
        "Default is 2^15 - 1, one full cycle of the PRBS-15 seed space.",
    )
