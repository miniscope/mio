"""
Data model for configuring an SD card. Will be instantiated in the constants module with
specific values. This allows for the model to be reused across different miniscopes, and
for consuming code to use a consistent, introspectable API
"""

from typing import ClassVar

from mio.devices.base.headers import BufferHeader
from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin


class SectorConfig(MiniscopeConfig):
    """
    Configuration of sector layout on the SD card.

    For each sector, one can retrieve the position with the attribute ``_pos``,

    Examples:

        >>> sectors = SectorConfig(header=1023, config=1024, data=1025, size=512)
        >>> sectors.header
        1023
        >>> # should be 1023 * 512
        >>> sectors.header_pos
        523776


    """

    header: int = 1023
    """
    Holds user settings to configure Miniscope and recording
    """
    config: int = 1024
    """
    Holds final settings of the actual recording
    """
    data: int = 1025
    """
    Recording data starts here
    """
    size: int = 512
    """
    The size of an individual sector
    """

    def __getattr__(self, item: str) -> int:
        """
        Get positions by multiplying by sector size
        (__getattr__ is only called if the name can't be found, so we don't need to handle
        the base case of the existing attributes)
        """
        split = item.split("_")
        if len(split) == 2 and split[1] == "pos":
            return getattr(self, split[0]) * self.size
        else:
            raise AttributeError()


class ConfigPositions(MiniscopeConfig):
    """
    Image acquisition configuration positions
    """

    width: int = 0
    height: int = 1
    fs: int = 2
    buffer_size: int = 3
    n_buffers_recorded: int = 4
    n_buffers_dropped: int = 5


class SDHeaderPositions(MiniscopeConfig):
    """
    Positions in the header for the whole SD card
    """

    gain: int = 4
    led: int = 5
    ewl: int = 6
    record_length: int = 7
    fs: int = 8
    """Frame rate"""
    delay_start: int | None = None
    battery_cutoff: int | None = None


class SDLayout(MiniscopeConfig, ConfigYAMLMixin):
    """
    Data layout of an SD Card.

    Used by the :class:`.io.SDCardDevice` class to tell it how data on the SD card is laid out.
    """

    sectors: SectorConfig
    write_key0: int = 0x0D7CBA17
    write_key1: int = 0x0D7CBA17
    write_key2: int = 0x0D7CBA17
    write_key3: int = 0x0D7CBA17
    """
    These don't seem to actually be used in the existing reading/writing code, 
    but we will leave them here for continuity's sake :)
    """
    word_size: int = 4
    """
    I'm actually not sure what this is, but 4 is hardcoded a few times in the 
    existing notebook and it appears to be used as a word size when
    reading from the SD card.
    """

    header: SDHeaderPositions = SDHeaderPositions()
    config: ConfigPositions = ConfigPositions()


class SDConfig(MiniscopeConfig):
    """
    The configuration of a recording taken on this SD card.

    Read from the locations given in :class:`.ConfigPositions`
    for an SD card with a given :class:`.SDLayout`
    """

    width: int
    height: int
    fs: int
    buffer_size: int
    n_buffers_recorded: int
    n_buffers_dropped: int


class SDBufferHeader(BufferHeader):
    """
    Header data at the start of each frame
    """

    POSITIONS: ClassVar[dict[str, int]] = {
        "length": 0,
        "linked_list": 1,
        "frame_num": 2,
        "buffer_count": 3,
        "frame_buffer_count": 4,
        "write_buffer_count": 5,
        "dropped_buffer_count": 6,
        "timestamp": 7,
        "data_length": 8,
    }

    length: int
    write_buffer_count: int
    dropped_buffer_count: int
    data_length: int
    write_timestamp: int | None = None
    battery_voltage: int | None = None
