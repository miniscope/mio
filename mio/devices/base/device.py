"""Base device class."""

from typing import ClassVar

from mio.devices.base.config import DeviceConfig
from mio.devices.base.headers import BufferHeader
from mio.types import ConfigSource


class Device:
    """Abstract device parent class"""

    header_cls: ClassVar[type[BufferHeader]] = BufferHeader
    config_cls: ClassVar[type[DeviceConfig]] = DeviceConfig

    def __init__(self, config: DeviceConfig | ConfigSource):
        if isinstance(config, self.config_cls):
            self.config = config
        else:
            self.config = self.config_cls.from_any(config)