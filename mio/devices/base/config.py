from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin


class DeviceConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Base class for device configurations"""
