from collections import defaultdict

import numpy as np
import pytest
from bitstring import Bits
from mio.stream_daq import iter_buffers

from mio.devices.msus.config import MSUSDevConfig
from mio.devices.msus.daq import format_frame
from mio.devices.msus.header import MSUSBufferHeader, MSUSBufferHeaderFormat
from mio.devices.msus.testing import (
    frame_to_naneye_buffers,
    patterned_frame,
)
from mio.utils import file_iter

from ..conftest import DATA_DIR


def test_format_frames():
    """
    Our format frame method should receive the 1-dimensional pixel buffers
    processed by parsing the headers (tested separately in `test_header`),
    and reassemble it to the original frame.
    """
    format = MSUSBufferHeaderFormat.from_id("msus-buffer-header")
    config = MSUSDevConfig.from_id("MSUS")

    frame = patterned_frame(
        width=config.frame_width, height=config.frame_height, pattern="sequential"
    )
    buffers = frame_to_naneye_buffers(frame)

    processed = [
        MSUSBufferHeader.from_buffer(buf, header_fmt=format, config=config) for buf in buffers
    ]
    pixels = [p[1] for p in processed]

    reconstructed = format_frame(pixels, config)
    assert np.array_equal(frame, reconstructed)


def test_format_headers_raw(msus_raw_buffers):
    """
    Use the fixtures and previously recorded .bin files to test the format_headers method.
    """
    format = MSUSBufferHeaderFormat.from_id("msus-buffer-header")
    config = MSUSDevConfig.from_id("MSUS")

    frame_buffers = defaultdict(list)
    for buffer in msus_raw_buffers:
        # this extracts header and pixels
        header, pixels = MSUSBufferHeader.from_buffer(buffer, header_fmt=format, config=config)
        # add the pixels value to a list of buffers
        frame_buffers[header.frame_num].append(pixels)

    # discard first and last which may be incomplete in the sample
    frames = list(frame_buffers.keys())
    del frame_buffers[frames[0]]
    del frame_buffers[frames[-1]]

    for pixel_arrays in frame_buffers.values():
        reconstructed = format_frame(pixel_arrays, config)
        assert reconstructed.shape == (config.frame_height, config.frame_width)


@pytest.mark.parametrize(
    "binary_input,thresh_low,thresh_high", [(DATA_DIR / "gs_test_raw_15_brightDark.bin", 300, 900)]
)
def test_format_frame_with_known_input(binary_input, thresh_low, thresh_high):
    """
    Assuming the preceding steps work (tested elsewhere),
    `format_frame` correctly reconstructs a 16-bit frame from a set of 1D pixel arrays.

    We use a raw sample from the device where the sensor is xposed to bright light
    for the first few frames, and then covered in the last few
    to generate "known input,"
    since the device is not capable of generating a test pattern.

    This test does not test the general correctness of `format_frame`,
    like its error handling, correctness of shape, etc.
    Here we are just testing the *values* of the frames - whether we get
    correct pixel values (or as close as we can verify with such a coarse notion of known input)
    """
    format = MSUSBufferHeaderFormat.from_id("msus-buffer-header")
    config: MSUSDevConfig = MSUSDevConfig.from_id("MSUS")

    iterator = file_iter(binary_input, 2048)
    frame_buffers = defaultdict(list)

    # collect pixel buffers by frame
    for buffer in iter_buffers(iterator, Bits(config.preamble)):
        header, pixels = MSUSBufferHeader.from_buffer(buffer, header_fmt=format, config=config)
        header: MSUSBufferHeader
        frame_buffers[header.frame_num].append(pixels)

    # delete the first and last, we assume they are incomplete
    del frame_buffers[min(frame_buffers.keys())]
    del frame_buffers[max(frame_buffers.keys())]

    frames = []
    for frame_n in sorted(frame_buffers.keys()):
        frames.append(format_frame(frame_buffers[frame_n], config))

    # first frames should be bright, last frames should be dark
    # this should be stricter, but the input data is not very clean
    # (dark is not very dark)
    # so we use 75% quantile for bright, and median for dark
    bright_qts = np.quantile(frames[0].flatten(), (0.25, 0.5, 0.75))
    dark_qts = np.quantile(frames[-1].flatten(), (0.25, 0.5, 0.75))

    # 75% of pixels are brighter than high thresh, vice versa for low
    assert bright_qts[0] > thresh_high
    assert dark_qts[1] < thresh_low
