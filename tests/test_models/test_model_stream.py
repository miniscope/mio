import pytest

from mio.const import DEVICE_DIR
from mio.models.stream import ADCScaling, StreamDevConfig, StreamBufferHeader

from ..conftest import CONFIG_DIR


@pytest.mark.parametrize(
    "config",
    [
        "preamble_hex.yml",
    ],
)
def test_preamble_hex_parsing(config):
    """
    Test that a hexadecimal string is correctly parsed to a byte string
    from a string or a hex integer
    """
    config_file = CONFIG_DIR / config

    instance = StreamDevConfig.from_yaml(config_file)
    assert instance.preamble == b"\x124Vx"


def test_absolute_bitstream():
    """
    Relative paths should be resolved relative to the devices dir
    """
    example = CONFIG_DIR / "wireless_example.yml"

    instance = StreamDevConfig.from_yaml(example)
    assert instance.bitstream.is_absolute()
    assert str(instance.bitstream).startswith(str(DEVICE_DIR))


_default_adc_scale = {
    "ref_voltage": 1.1,
    "bitdepth": 8,
    "battery_div_factor": 5,
    "vin_div_factor": 11.3,
}


@pytest.mark.parametrize("scale", [None, 1, 2, _default_adc_scale["ref_voltage"]])
def test_adc_scaling(scale, config_override):
    """
    Test that the ADC scaling factors are correctly parsed,
    and that :class:`.ADCScaling` methods are correctly applied to their relevant values
    """
    if scale is None:
        adc_scale = None
    else:
        adc_scale = _default_adc_scale.copy()
        adc_scale.update({"ref_voltage": scale})

    example = config_override(CONFIG_DIR / "stream_daq_test_200px.yml", {"adc_scale": adc_scale})
    instance_config = StreamDevConfig.from_yaml(example)

    battery_voltage_raw = 200
    input_voltage_raw = 250

    instance_header = StreamBufferHeader(
        linked_list=0,
        frame_num=0,
        buffer_count=0,
        frame_buffer_count=0,
        write_buffer_count=0,
        dropped_buffer_count=0,
        timestamp=0,
        pixel_count=0,
        write_timestamp=0,
        battery_voltage_raw=battery_voltage_raw,
        input_voltage_raw=input_voltage_raw,
    )
    instance_header.adc_scaling = instance_config.adc_scale

    if scale is None:
        assert instance_header.battery_voltage == battery_voltage_raw
        assert instance_header.input_voltage == input_voltage_raw

    else:
        adcscale = ADCScaling(**adc_scale)
        assert instance_header.battery_voltage == adcscale.scale_battery_voltage(
            battery_voltage_raw
        )
        assert instance_header.input_voltage == adcscale.scale_input_voltage(input_voltage_raw)
