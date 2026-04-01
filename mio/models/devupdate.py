"""
Models for device update commands.
"""

from enum import Enum

import serial.tools.list_ports
from pydantic import BaseModel, field_validator, model_validator


class DeviceCommand(Enum):
    """Commands for device."""

    REBOOT = 200


class UpdateCommandDefinitions:
    """
    Definitions of Bit masks and headers for remote update commands.
    """

    # Header to indicate key/value.
    # It probably won't be used in other places so defined here.
    id_header = 0b00000000
    key_header = 0b11000000
    LSB_header = 0b01000000
    MSB_header = 0b10000000
    LSB_value_mask = 0b000000111111  # value below 12-bit
    MSB_value_mask = 0b111111000000  # value below 12-bit
    reset_byte = 0b11111111


class UpdateKey(int, Enum):
    """
    Keys to update. Needs to be under 6-bit.
    """

    LED = 0
    GAIN = 1
    ROI_X = 2
    ROI_Y = 3
    SUBSAMPLE = 4
    """
    ROI_WIDTH = 4  # not implemented
    ROI_HEIGHT = 5  # not implemented
    EWL = 6  # not implemented
    """
    DEVICE = 50  # for device commands


class DevUpdateCommand(BaseModel):
    """
    Command to update device configuration.
    """

    device_id: int
    port: str
    key: UpdateKey
    value: int

    @model_validator(mode="after")
    def validate_values(cls, values: dict) -> dict:
        """
        Validate values based on key.
        """
        key = values.key
        value = values.value

        if key == UpdateKey.LED:
            assert 0 <= value <= 100, "For LED, value must be between 0 and 100"
        elif key == UpdateKey.GAIN:
            assert value in [1, 2, 4], "For GAIN, value must be 1, 2, or 4"
        elif key == UpdateKey.DEVICE:
            assert value in [DeviceCommand.REBOOT.value], "For DEVICE, value must be in [200]"
        elif key == UpdateKey.SUBSAMPLE:
            assert value in [0, 1], "For SUBSAMPLE, value must be in [0, 1]"
        elif key in [UpdateKey.ROI_X, UpdateKey.ROI_Y]:
            # validation not implemented
            pass
        elif key in UpdateKey:
            raise NotImplementedError()
        else:
            raise ValueError(f"{key} is not a valid update key," "need an instance of UpdateKey")
        return values

    @field_validator("port")
    def validate_port(cls, value: str) -> str:
        """
        Validate port.

        Args:
            value: Port to validate.

        Returns:
            Validated port.

        Raises:
            ValueError: If no serial ports found or port not found.
        """
        portlist = list(serial.tools.list_ports.comports())

        if len(portlist) == 0:
            raise ValueError("No serial ports found")
        if value not in [port.device for port in portlist]:
            raise ValueError(f"Port {value} not found")
        return value

    @field_validator("key", mode="before")
    def validate_key(cls, value: str) -> UpdateKey:
        """
        Validate and convert key string to UpdateKey Enum type.

        Args:
            value (str): Key to validate.

        Returns:
            UpdateKey: Validated key as UpdateKey.

        Raises:
            ValueError: If key not found.
        """
        try:
            return UpdateKey[value]
        except KeyError as e:
            raise ValueError(
                f"Key {value} not found, must be a member of UpdateKey:"
                f" {list(UpdateKey.__members__.keys())}."
            ) from e
