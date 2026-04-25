# stream

This module is a data acquisition module that captures video streams from Miniscopes based on the `Miniscope-SAMD-Framework` firmware. The firmware repository will be published in future updates but is currently under development and private.

## Command
After [installation](../../guide/installation.md) and customizing [device configurations](stream-dev-config) and {class}`mio.devices.stream.config.StreamDevConfig <runtime configuration>` if necessary, run the command described in [CLI Usage](../../cli/index).

One example of this command is the following:
```bash
$ mio stream capture -c .path/to/device/config.yml -o output_filename.avi -m
```
A window displaying the image transferred from the Miniscope and a graph plotting metadata (`-m` option) should pop up. Additionally, the indexes of captured frames and their statuses will be logged in the terminal. The `MIO_STREAM_HEADER_PLOT_KEY` defines plotted header fields (see `.env.sample`).

## Prerequisites
- **Data capture hardware:** Opal Kelly XEM7310-A75 FPGA board (connected via USB)
- **Supported Operating Systems:** MacOS or Ubuntu PC (To do: specify version)
- **Imaging hardware:** Miniscope based on the `Miniscope-SAMD-Framework` firmware. Hardware modules for feeding in data into the data capture hardware are also needed but these will be specified in future updates.

(stream-dev-config)=
## Device configuration
A YAML file is used to configure Stream DAQ based on the device configuration. The device configuration needs to match the imaging and data capture hardware for proper operation. This file is used to set up hardware, define data formats, and set data preambles. The contents of this YAML file will be parsed into a {class}`mio.devices.stream.config.StreamDevConfig <model>`, which then configures the Stream DAQ.

### FPGA (Opal Kelly) configuration
The `bitstream` field in the device configuration yaml file specifies the image that will be uploaded to the opal kelly board. This file needs to be placed in `mio.devices`.


#### Bitstream file nomenclature
Name format of the bitstream files and directory:
`[DEVICE_DIR]/USBInterface-[CLOCK_FREQUENCY]-[INPUT_PIN]_[IO_VOLTAGE]-[ENCODING_POLARITY]`
- **DEVICE_DIR**: FPGA module name. The current package supports XEM7310-A75 (Opal kelly).
- **CLOCK_FREQUENCY**: Manchester encoding clock frequency. For the current FPGA (XEM7310-A75), this frequency can be configured to 100 / *i* MHz where *i* is an integer.
- **INPUT_PIN**: Signal input pin. `J2_2` means signal goes into second pin of J2 pin headers in the hardware module.
- **IO_VOLTAGE**: I/O voltage of the FPGA `INPUT_PIN`. The current package supports 3.3V input.
- **ENCODING_POLARITY**: Manchester encoding convention, which corresponds to bit polarity. The current package supports IEEE convention Manchester encoding.

### Example device configuration
Below is an example configuration YAML file. More examples can be found in `mio.data.config`.

```yaml
# capture device. "OK" (Opal Kelly) or "UART"
device: "OK"

# bitstream file to upload to Opal Kelly board
bitstream: "XEM7310-A75/USBInterface-8_33mhz-J2_2-3v3-IEEE.bit"

# COM port and baud rate is only required for UART mode
port: null
baudrate: null

# Preamble for each data buffer.
preamble: 0x12345678

# Image format. StreamDevice will calculate buffer size, etc. based on these parameters
frame_width: 200
frame_height: 200
pix_depth: 8

# Buffer data format. These have to match the firmware value
header_len: 384 # 12 * 32 (in bits)
buffer_block_length: 10
block_size: 512
num_buffers: 32
dummy_words: 10

# Flags to flip bit/byte order when recovering headers and data. See model document for details.
reverse_header_bits: True
reverse_header_bytes: True
reverse_payload_bits: True
reverse_payload_bytes: True

adc_scale:
  ref_voltage: 1.1
  bitdepth: 8
  battery_div_factor: 5
  vin_div_factor: 11.3

runtime:
  serial_buffer_queue_size: 10
  frame_buffer_queue_size: 5
  image_buffer_queue_size: 5
  csv:
    buffer: 100
  plot:
    keys:
      - timestamp
      - buffer_count
      - frame_buffer_count
      - battery_voltage
      - input_voltage
    update_ms: 1000
    history: 500
```

```{eval-rst}
.. automodule:: mio.devices.stream
    :members:
    :undoc-members:
```