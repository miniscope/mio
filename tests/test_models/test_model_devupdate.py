from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mio.models.devupdate import DeviceCommand, DevUpdateCommand, UpdateKey


def mock_comports():
    class Port:
        def __init__(self, device):
            self.device = device

    return [Port("COM1"), Port("COM2")]


@pytest.fixture
def mock_serial_ports():
    with patch("serial.tools.list_ports.comports", side_effect=mock_comports):
        yield


def test_valid_led_update(mock_serial_ports):
    cmd = DevUpdateCommand(device_id=1, port="COM1", key="LED", value=50)
    assert cmd.key == UpdateKey.LED
    assert cmd.value == 50


def test_valid_gain_update(mock_serial_ports):
    cmd = DevUpdateCommand(device_id=1, port="COM2", key="GAIN", value=2)
    assert cmd.key == UpdateKey.GAIN
    assert cmd.value == 2


def test_invalid_led_value(mock_serial_ports):
    with pytest.raises(ValidationError):
        DevUpdateCommand(device_id=1, port="COM1", key="LED", value=150)


def test_invalid_gain_value(mock_serial_ports):
    with pytest.raises(ValidationError):
        DevUpdateCommand(device_id=1, port="COM1", key="GAIN", value=3)


def test_invalid_key(mock_serial_ports):
    with pytest.raises(ValueError):
        DevUpdateCommand(device_id=1, port="COM1", key="FAKEDEVICE", value=10)


def test_invalid_port():
    with patch("serial.tools.list_ports.comports", return_value=mock_comports()):
        with pytest.raises(ValidationError):
            DevUpdateCommand(device_id=1, port="COM3", key="LED", value=50)


def test_device_command(mock_serial_ports):
    cmd = DevUpdateCommand(device_id=1, port="COM2", key="DEVICE", value=DeviceCommand.REBOOT.value)
    assert cmd.value == DeviceCommand.REBOOT.value
