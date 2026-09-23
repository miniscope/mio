"""
Models for device update batch configuration.
"""

from typing import Literal

from pydantic import Field

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin

PermittedKey = Literal["LED", "GAIN", "ROI_X", "ROI_Y", "SUBSAMPLE"]


class DeviceUpdateEntry(MiniscopeConfig):
    """
    One device's update specification.

    Optional `device_id` and `port` may override CLI defaults for this entry.
    """

    device_id: int = Field(description="Target device ID")
    port: str | None = Field(default=None, description="Serial port (optional)")
    updates: dict[PermittedKey, int] = Field(default_factory=dict)


class UpdateBatch(MiniscopeConfig, ConfigYAMLMixin):
    """
    Batch update configuration. Only list form is supported:

        devices: [DeviceUpdateEntry, ...]
    """

    devices: list[DeviceUpdateEntry]

    def iter_updates(
        self, default_port: str | None
    ) -> list[tuple[str | None, int, PermittedKey, int]]:
        """
        Yield (port, device_id, key, value) tuples for all updates in this batch.
        """
        results: list[tuple[str | None, int, PermittedKey, int]] = []

        if not self.devices:
            raise ValueError("devices list must not be empty in UpdateBatch")

        for entry in self.devices:
            entry_port = entry.port if entry.port is not None else default_port
            entry_device_id = entry.device_id
            for k, v in entry.updates.items():
                results.append((entry_port, entry_device_id, k, v))

        return results
