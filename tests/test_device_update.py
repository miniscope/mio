from unittest.mock import MagicMock, call, patch

import pytest
import serial
from pydantic import ValidationError

from mio.device_update import device_update, find_ftdi_device
from mio.models.devupdate import UpdateCommandDefinitions, UpdateKey


@pytest.fixture
def mock_serial_fixture(request):
    device_list = request.param
    with patch("serial.Serial") as mock_serial, patch(
        "serial.tools.list_ports.comports"
    ) as mock_comports:
        mock_serial_instance = mock_serial.return_value
        mock_comports.return_value = [
            MagicMock(vid=device["vid"], pid=device["pid"], device=device["device"])
            for device in device_list
        ]
        yield mock_serial, mock_comports, mock_serial_instance


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [
            {"vid": 0x0403, "pid": 0x6001, "device": "COM3"},
            {"vid": 0x0111, "pid": 0x6111, "device": "COM2"},
        ],
    ],
    indirect=True,
)
def test_devupdate_with_device_connected(mock_serial_fixture):
    """
    Test device_update function with a device connected.
    """
    mock_serial, mock_comports, mock_serial_instance = mock_serial_fixture
    key = "LED"
    value = 2
    device_id = 1
    port = "COM3"

    device_update(key, value, device_id, port)

    mock_serial.assert_called_once_with(port=port, baudrate=2400, timeout=5, stopbits=2)

    id_command = (device_id + UpdateCommandDefinitions.id_header) & 0xFF
    key_command = (UpdateKey.LED.value + UpdateCommandDefinitions.key_header) & 0xFF
    value_LSB_command = (
        (value & UpdateCommandDefinitions.LSB_value_mask) + UpdateCommandDefinitions.LSB_header
    ) & 0xFF
    value_MSB_command = (
        ((value & UpdateCommandDefinitions.MSB_value_mask) >> 6)
        + UpdateCommandDefinitions.MSB_header
    ) & 0xFF
    reset_command = UpdateCommandDefinitions.reset_byte

    expected_calls = [
        call(id_command.to_bytes(1, "big")),
        call(key_command.to_bytes(1, "big")),
        call(value_LSB_command.to_bytes(1, "big")),
        call(value_MSB_command.to_bytes(1, "big")),
        call(reset_command.to_bytes(1, "big")),
    ]

    assert mock_serial_instance.write.call_count == len(expected_calls)
    mock_serial_instance.write.assert_has_calls(expected_calls, any_order=False)


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [],
    ],
    indirect=True,
)
def test_devupdate_without_device_connected(mock_serial_fixture):
    """
    Test device_update function without a device connected.
    """

    key = "GAIN"
    value = 2
    device_id = 0

    with pytest.raises(ValueError, match="No FTDI devices found."):
        device_update(key, value, device_id)


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [
            {"vid": 0x0403, "pid": 0x6001, "device": "COM3"},
            {"vid": 0x0111, "pid": 0x6111, "device": "COM2"},
        ],
    ],
    indirect=True,
)
def test_find_ftdi_device(mock_serial_fixture):
    """
    Test find_ftdi_device function.
    """
    result = find_ftdi_device()

    assert result == ["COM3"]


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [
            {"vid": 0x0403, "pid": 0x6001, "device": "COM3"},
            {"vid": 0x0111, "pid": 0x6111, "device": "COM2"},
        ],
    ],
    indirect=True,
)
def test_invalid_key_raises_error(mock_serial_fixture):
    """
    Test that an invalid key raises an error.
    """

    key = "RANDOM_STRING"
    value = 50
    device_id = 1
    port = "COM3"

    with pytest.raises(ValidationError, match="Key RANDOM_STRING not found"):
        device_update(key, value, device_id, port)


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [
            {"vid": 0x0403, "pid": 0x6001, "device": "COM3"},
            {"vid": 0x0111, "pid": 0x6111, "device": "COM2"},
        ],
    ],
    indirect=True,
)
def test_invalid_led_value_raises_error(mock_serial_fixture):
    """
    Test that an invalid LED value raises an error.
    """
    mock_serial, mock_comports, mock_serial_instance = mock_serial_fixture

    key = "LED"
    value = 150  # LED value should be between 0 and 100
    device_id = 1
    port = "COM3"

    with pytest.raises(ValidationError, match="For LED, value must be between 0 and 100"):
        device_update(key, value, device_id, port)


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [
            {"vid": 0x0403, "pid": 0x6001, "device": "COM3"},
            {"vid": 0x0403, "pid": 0x6001, "device": "COM2"},
        ],
    ],
    indirect=True,
)
def test_devupdate_with_multiple_ftdi_devices(mock_serial_fixture):
    """
    Test that multiple FTDI devices raise an error.
    """

    key = "GAIN"
    value = 5
    device_id = 1

    with pytest.raises(ValueError, match="Multiple FTDI devices found. Please specify the port."):
        device_update(key, value, device_id)


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [{"vid": 0x0403, "pid": 0x6001, "device": "COM3"}],
    ],
    indirect=True,
)
def test_devupdate_serial_exception_handling(mock_serial_fixture):
    """
    Test exception handling when serial port cannot be opened.
    Might be too obvious so it might be better to re-think this test.
    """
    mock_serial, mock_comports, mock_serial_instance = mock_serial_fixture

    mock_serial.side_effect = serial.SerialException("Serial port error")

    key = "LED"
    value = 50
    device_id = 1
    port = "COM3"

    with pytest.raises(serial.SerialException):
        device_update(key, value, device_id, port)


@pytest.mark.parametrize(
    "mock_serial_fixture",
    [
        [{"vid": 0x0413, "pid": 0x6111, "device": "COM2"}],
    ],
    indirect=True,
)
def test_specified_port_not_ftdi_device(mock_serial_fixture):
    """
    Test with a specified port not corresponding to an FTDI device.
    """
    mock_serial, mock_comports, mock_serial_instance = mock_serial_fixture

    key = "GAIN"
    value = 10
    device_id = 1

    with pytest.raises(ValueError, match="No FTDI devices found."):
        device_update(key, value, device_id)
