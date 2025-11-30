"""
Update miniscope device configuration, such as LED, GAIN, etc.
"""

import time
from typing import Optional

import serial
import serial.tools.list_ports

from mio.logging import init_logger
from mio.models.devupdate import DevUpdateCommand, UpdateCommandDefinitions

logger = init_logger(name="device_update")
FTDI_VENDOR_ID = 0x0403
FTDI_PRODUCT_ID = 0x6001


def device_update(
    key: str,
    value: int,
    device_id: int,
    port: Optional[str] = None,
) -> None:
    """
    Remote update of device configuration.

    Args:
        device_id: ID of the device. 0 will update all devices.
        port: Serial port to which the device is connected.
        key: What to update on the device (e.g., LED, GAIN).
        value: Value to which the key should be updated.

    Returns:
        None
    """

    if port:
        logger.debug(f"Using port {port}")
    else:
        ftdi_port_list = find_ftdi_device()
        if len(ftdi_port_list) == 0:
            raise ValueError("No FTDI devices found.")
        if len(ftdi_port_list) > 1:
            raise ValueError("Multiple FTDI devices found. Please specify the port.")
        if len(ftdi_port_list) == 1:
            port = ftdi_port_list[0]
            logger.debug(f"Using port {port}")

    command = DevUpdateCommand(device_id=device_id, port=port, key=key, value=value)
    logger.info(f"Updating device id {device_id} with {key}={value}")

    try:
        serial_port = serial.Serial(port=command.port, baudrate=2400, timeout=5, stopbits=2)
    except Exception as e:
        logger.exception(e)
        raise e
    logger.debug("Open serial port")

    try:
        id_command = (command.device_id + UpdateCommandDefinitions.id_header) & 0xFF
        serial_port.write(id_command.to_bytes(1, "big"))
        logger.debug(f"Command: {format(id_command, '08b')}; Device ID: {command.device_id}")
        time.sleep(0.1)

        key_command = (command.key.value + UpdateCommandDefinitions.key_header) & 0xFF
        serial_port.write(key_command.to_bytes(1, "big"))
        logger.debug(f"Command: {format(key_command, '08b')}; Key: {command.key.name}")
        time.sleep(0.1)

        value_LSB_command = (
            (command.value & UpdateCommandDefinitions.LSB_value_mask)
            + UpdateCommandDefinitions.LSB_header
        ) & 0xFF
        serial_port.write(value_LSB_command.to_bytes(1, "big"))
        logger.debug(f"Command: {format(value_LSB_command, '08b')}; Value: {command.value} (LSB)")
        time.sleep(0.1)

        value_MSB_command = (
            ((command.value & UpdateCommandDefinitions.MSB_value_mask) >> 6)
            + UpdateCommandDefinitions.MSB_header
        ) & 0xFF
        serial_port.write(value_MSB_command.to_bytes(1, "big"))
        logger.debug(f"Command: {format(value_MSB_command, '08b')}; Value: {command.value} (MSB)")
        time.sleep(0.1)

        serial_port.write(UpdateCommandDefinitions.reset_byte.to_bytes(1, "big"))

    finally:
        serial_port.close()
        logger.debug("Closed serial port")


def find_ftdi_device() -> list[str]:
    """
    Find FTDI devices connected to the computer.
    """
    ports = serial.tools.list_ports.comports()
    ftdi_ports = []

    for port in ports:
        if port.vid == FTDI_VENDOR_ID and port.pid == FTDI_PRODUCT_ID:
            ftdi_ports.append(port.device)

    return ftdi_ports
