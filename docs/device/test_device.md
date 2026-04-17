# Example stream device

**Under Construction:** This section will be populated when devices are released.

## Buffer Structure
- **Contents**: The buffer consists of a concatenation of 32-bit dummy words, 32-bit header data, and 8-bit pixel data. The dummy words are for stabilizing the clock recovery function in the Manchester decoder FPGA, and the header data is used for recovering images and detecting device status.
- **Payload**: A single image is split and stored in a circulating buffer within the device. The [`num_buffers`](../api/devices/stream) should match the number of circulating buffers in the device.

## Header Values and Expected Transitions
See following docs for the basic structure.
- `mio.models.buffer.BufferHeaderFormat`
- `mio.models.stream.StreamBufferHeaderFormat`

Device specific notes are listed below.
- **`preamble`**: 32-bit preamble for detecting the beginning of each buffer. The [`preamble`](../api/devices/stream) in the device config needs to match the preamble defined in firmware.
- **`dropped_buffer_count`**: Currently not used and should always be zero.
- **`write_timestamp`**: Currently not used and should always be zero.
- **`battery_voltage`**: Currently not used and should always be zero.
- **`ewl_pos`**: Currently not used and should always be zero.